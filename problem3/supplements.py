"""Run Code1 and align its results and the acoustic reference with the Q3 endpoints."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from models import BEHAVIORS, CALLS, COMBOS


def run_code1(data, output, mode, rscript=None, profile="published"):
    tmp = Path(tempfile.gettempdir())
    local_r = tmp / "apmcm-q2-r-runtime/R-4.4.3/bin/Rscript.exe"
    candidate = rscript or os.environ.get("RSCRIPT") or shutil.which("Rscript") or local_r
    executable = shutil.which(str(candidate))
    if not executable:
        raise FileNotFoundError("Rscript unavailable. Set --rscript to the installed Rscript executable.")
    runner = Path(executable).resolve()
    env = os.environ.copy()
    if os.name=="nt" and env.get("LC_ALL")=="C.UTF-8":
        env["LC_ALL"]="English_United States.utf8"
    if runner == local_r.resolve():
        library = tmp / "apmcm-q3-r-library"
        if library.is_dir():
            env.setdefault("R_LIBS_USER", str(library))
        tools = tmp / "apmcm-q3-build"
        compiler = tools / "x86_64-w64-mingw32.static.posix"
        shell = shutil.which("git")
        if (tools / "usr/bin/make.exe").exists() and shell:
            git_shell = Path(shell).resolve().parents[1] / "usr/bin"
            paths = os.pathsep.join(p.as_posix() for p in [compiler/"bin", tools/"usr/bin", git_shell])
            env.setdefault("R_CUSTOM_TOOLS_SOFT", compiler.as_posix())
            env.setdefault("R_CUSTOM_TOOLS_PATH", paths)
            env["PATH"] = paths + os.pathsep + env["PATH"]
    env.setdefault("Q3_STAN_CACHE", str(tmp / "apmcm-q3-stan-cache"))
    original = data.with_name("Code1_AfricanPenguins_playbacks_analyses.R")
    wrapper = Path(__file__).with_name("code1_baseline.R")
    iterations = ["20000", "5000"] if profile == "published" else ["2000", "1000"]
    exports = (["analysis_data", "baseline_counts"] if mode == "prepare" else
               ["baseline_parameters", "baseline_diagnostics", "baseline_predictions"])
    # Stage relative ASCII filenames because older Windows R locales corrupt Unicode arguments.
    with tempfile.TemporaryDirectory(prefix="q3-code1-") as work:
        stage = Path(work)
        for source in [original, wrapper, data]:
            shutil.copyfile(source, stage/source.name)
        subprocess.run([str(runner), "--vanilla", wrapper.name, data.name, original.name, ".",
                        mode, *iterations], cwd=stage, env=env, check=True)
        for export in exports:
            name = f"r_{export}.csv"
            shutil.copyfile(stage/name, output/name)


def analysis_data(data, output):
    d = pd.read_csv(output / "r_analysis_data.csv")
    raw = pd.read_csv(data)
    pd.testing.assert_frame_equal(d[raw.columns], raw, check_dtype=False)
    d["code"] = d.ResponseType.replace({"none": "000"}).map(dict(zip(COMBOS, range(8))))
    if d.code.isna().any():
        raise ValueError("Unknown Code1 ResponseType")
    counts = pd.read_csv(output/"r_baseline_counts.csv").set_index("Type").sort_index()
    columns = ["ResponseYesNo", *(h+"YesNo" for h in BEHAVIORS)]
    expected = raw.groupby("Type")[columns].sum()
    expected.columns = ["Responses", *BEHAVIORS]
    pd.testing.assert_frame_equal(counts, expected, check_dtype=False)
    return d


def published_parameters(pdf):
    content = subprocess.check_output(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"])
    sections = {"HeadsUpDuration": "HeadsUp", "VocalNcalls": "Vocal", "ApproachYesNo": "Approach"}
    term_pattern = re.compile(r"(?:hu_)?(?:Intercept|Context\w+(?::CallType\w+)?|CallType\w+|scaleDistance|scaleNpenguins|VocalNcallsBefore)")
    rows = []
    behavior = None
    for page, text in enumerate(content.decode("utf-8").split("\f"), 1):
        for line in text.splitlines():
            words = line.split()
            if words[:1] == ["Formula:"]:
                behavior = sections[words[1]]
            if behavior and len(words) == 8 and term_pattern.fullmatch(words[0]):
                rows.append([behavior, words[0], page, *map(float, [words[1], words[3], words[4]])])
    return pd.DataFrame(rows, columns=["behavior","term","pdf_page","published_mean","published_lo95","published_hi95"])


def model_comparisons(data, output, intensity, latency):
    published = published_parameters(data.with_name("Supplementary3_Modelling_analyses_on_playback_experiments.pdf"))
    fit = pd.read_csv(output/"r_baseline_parameters.csv")
    params = published.merge(fit, on=["behavior","term"], how="outer", validate="one_to_one", indicator=True)
    if not params._merge.eq("both").all():
        raise ValueError("Supplementary 3 and Code1 fixed-effect names disagree")
    params = params.drop(columns="_merge")
    params["refit_minus_published"] = params["mean"] - params.published_mean
    keys = ["CallType","Distance","Context","Npenguins","behavior"]
    measures = {"HeadsUpDuration":"HeadsUp","VocalNcalls":"Vocal"}
    positive = intensity[intensity.measure.isin(measures)].copy()
    positive["behavior"] = positive.measure.map(measures)
    approach = latency[(latency.Context!="standardized") & (latency.behavior=="Approach")]
    def probabilities(frame, col):
        return frame[keys+[col,col+"_lo",col+"_hi"]].set_axis(keys+["q3_probability","q3_lo95","q3_hi95"],axis=1)
    primary = pd.concat([probabilities(positive,"p_positive"), probabilities(approach,"p_event_60")])
    preds = pd.read_csv(output/"r_baseline_predictions.csv").merge(primary, on=keys, how="outer",
                                                                   validate="one_to_one", indicator=True)
    if not preds._merge.eq("both").all() or len(preds)!=108:
        raise ValueError("Incomplete Code1 / Q3 comparison grid")
    preds["endpoint"] = preds.behavior.map(dict(zip(BEHAVIORS, ["HeadsUpDuration > 0","VocalNcalls > 0","ApproachYesNo = 1"])))
    preds["q3_minus_r_pp"] = 100*(preds.q3_probability-preds.r_probability)
    return params, preds.drop(columns="_merge")


def acoustic_comparison(project, latency, q2_results=None):
    if q2_results is None:
        paths = list(project.glob("*/results/gam_predictions.csv"))
        if len(paths) != 1:
            raise ValueError("Set --q2-results to the directory containing the Code2 GAM exports.")
        q2_results = paths[0].parent
    audit = pd.read_csv(q2_results/"supplementary4_check.csv")
    if (len(audit)!=63 or audit.duplicated(["call","metric","d"]).any() or
        not np.array_equal(audit.valid, audit.wide_n) or
        not np.allclose(audit[["mean","sd"]], audit[["wide_mean","wide_sd"]], atol=1e-8, rtol=0)):
        raise ValueError("Supplementary 4 source-data cross-check failed")
    gam = pd.read_csv(q2_results/"gam_predictions.csv")
    gam = gam[(gam.hold==0) & gam.d.isin([50,100,150])].rename(columns={"call":"CallType","d":"Distance"})
    keys = ["CallType","Distance"]
    metrics = {"envelope","spectrum","spectrogram"}
    if set(gam.metric)!=metrics or len(gam)!=27:
        raise ValueError("Expected three complete acoustic curves per natural call")
    sound = gam.pivot(index=keys,columns="metric",values=["estimate","lower90","upper90"])
    sound.columns = [f"{metric}_{stat}" for stat,metric in sound.columns]
    typical = latency[(latency.Context=="standardized") & latency.CallType.isin(CALLS[1:])]
    outcomes = ["p_event_60","p_event_60_lo","p_event_60_hi","median_if_event_s",
                "median_if_event_s_lo","median_if_event_s_hi","events_exact"]
    behavior = typical.pivot(index=keys+["Npenguins"],columns="behavior",values=outcomes)
    behavior.columns = [f"{h}_{stat}" for stat,h in behavior.columns]
    support = typical[keys+["n_exact","outside_distance_range"]].drop_duplicates()
    both = behavior.reset_index().merge(support, on=keys, validate="one_to_one").merge(
        sound.reset_index(), on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(both)!=9 or not both._merge.eq("both").all() or both[sound.columns].isna().any().any():
        raise ValueError("Incomplete acoustic / behavior comparison")
    return both.drop(columns="_merge")
