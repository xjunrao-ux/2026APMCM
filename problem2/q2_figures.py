"""Figures for the propagation analysis: observed channels, fidelity, threshold
inversion and ranking reversal. Marker positions refer to measured distances."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"apmcm-q2-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = ["#366691", "#BB5927", "#137B74"]
NAMES = ["Flat", "Modulated", "Two-voice"]
MARKERS = ["o", "s", "^"]
STYLES = ["-", "--", "-."]


def draw_figures(out, distance, observed, fitted, sweep):
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"Arial","font.size":8,"svg.fonttype":"none",
                         "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.linewidth":0.7,"legend.frameon":False})

    def finish(fig, name, multipanel=False):
        if multipanel:
            # Optional external QA integration; data and plots remain portable.
            qa = Path.home()/".codex"/"skills"/"nature-figure"/"scripts"
            if (qa/"audit_panel_alignment.py").is_file():
                sys.path.insert(0,str(qa))
                from audit_panel_alignment import require_matplotlib_panel_alignment
                require_matplotlib_panel_alignment(fig, json_out=folder/(name+".alignment.json"),
                                                   strict=True)
        fig.savefig(folder / f"{name}.pdf")
        fig.savefig(folder / f"{name}.svg")
        fig.savefig(folder / f"{name}.png",dpi=300)
        plt.close(fig)

    def dist_axis(ax):
        assert np.all(distance > 0)
        ax.set_xscale("log",base=2)
        ax.set_xticks(distance,labels=[str(int(d)) for d in distance])
        ax.set_xlim(3.6,285)
        ax.set_xlabel("Distance (m)")
        ax.set_ylim(.05,1)

    legend = [Line2D([],[],color=c,ls=s,marker=m,markerfacecolor="white",label=n)
              for c,s,m,n in zip(COLORS,STYLES,MARKERS,NAMES)]
    fig, axes = plt.subplots(1,3,figsize=(7.3,2.65),sharey=True)
    fig.subplots_adjust(left=.084,right=.985,bottom=.23,top=.74,wspace=.19)
    for j,(ax,title) in enumerate(zip(axes,["Spectrogram","Amplitude spectrum","Amplitude envelope"])):
        for i in range(3):
            ax.plot(distance,fitted[i,j],ls=STYLES[i],color=COLORS[i],lw=1.35)
            ax.scatter(distance,observed[i,j],s=15,marker=MARKERS[i],
                       facecolors="white",edgecolors=COLORS[i],zorder=3,linewidths=.85)
        dist_axis(ax)
        ax.set_title(title,fontsize=9,pad=6)
    axes[0].set_ylabel("Correlation with 1 m reference")
    fig.legend(handles=legend,loc="upper center",ncol=3,bbox_to_anchor=(.52,1))
    finish(fig,"01_channels",multipanel=True)

    fig, ax = plt.subplots(figsize=(5.7,3.4))
    fig.subplots_adjust(left=.14,bottom=.18,right=.97,top=.85)
    for level in (.5,.6,.7):
        ax.axhline(level,ls=":",color="#A0A7AB",lw=.8,zorder=0)
    for i in range(3):
        ax.plot(distance,fitted[i].mean(axis=0),ls=STYLES[i],color=COLORS[i])
        ax.scatter(distance,observed[i].mean(axis=0),marker=MARKERS[i],facecolors="white",
                   edgecolors=COLORS[i],s=27,zorder=3)
    dist_axis(ax)
    ax.set_ylabel("Equal-weight acoustic fidelity")
    fig.legend(handles=legend,loc="upper center",ncol=3)
    finish(fig,"02_fidelity")

    fig, ax = plt.subplots(figsize=(5.7,3.4))
    fig.subplots_adjust(left=.14,bottom=.18,right=.97,top=.85)
    for level in (.5,.6,.7):
        ax.axvline(level,ls=":",color="#A0A7AB",lw=.8,zorder=0)
    for i,c in enumerate(("flat","mod","tv")):
        curve = sweep[(sweep.call==c)&(sweep.status=="crossing")]
        ax.plot(curve.threshold,curve.distance_m,ls=STYLES[i],color=COLORS[i])
        focus = curve[curve.threshold.isin([.5,.6,.7])]
        ax.scatter(focus.threshold,focus.distance_m,marker=MARKERS[i],s=27,
                   facecolors="white",edgecolors=COLORS[i],zorder=3)
    ax.set(xlabel="Required acoustic fidelity",ylabel="Threshold distance (m)",
           xlim=(.15,.9),ylim=(0,265))
    fig.legend(handles=legend,loc="upper center",ncol=3)
    finish(fig,"03_threshold_distance")

    fig, ax = plt.subplots(figsize=(5.7,3.2))
    fig.subplots_adjust(left=.15,bottom=.19,right=.97,top=.85)
    w = np.linspace(0,1,101)
    ax.axvline(1/3,color="#A0A7AB",ls=":",lw=.9)
    for i in range(3):
        curve = (1-w)*observed[i,:2,-1].mean()+w*observed[i,2,-1]
        ax.plot(w,curve,ls=STYLES[i],color=COLORS[i],label=NAMES[i])
    ax.set(xlabel="Weight assigned to the amplitude envelope",
           ylabel="Observed fidelity at 256 m",xlim=(0,1),ylim=(.12,.36))
    fig.legend(handles=legend,loc="upper center",ncol=3)
    finish(fig,"04_weight_at_256m")
