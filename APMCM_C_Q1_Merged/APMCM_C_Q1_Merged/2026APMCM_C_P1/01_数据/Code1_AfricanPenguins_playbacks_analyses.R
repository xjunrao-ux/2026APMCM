### packages ####
library(brms)
library(bayesplot)
library(parallel)
library(future)
library(rstanarm)
library(loo)
library(reshape2)
library(ggplot2)
library(dplyr)

### load data ####
dat<-read.csv(file="Data_AfricanPenguins_playbacks_data.csv")

### explore data ####
length(unique(dat$expID)) # 139 experiments

table(dat$CallType,dat$Context)
#          after commuting searching
# control    36        65        38
# flat       12        25        11
# mod         9        15         9
# tv         15        25        18
unique(dat$PlaybackTrack) # 11 flat, 10 tv, 7 mod

# overall responses
data.frame(
  Type = c("control","call"),
  Responses=c(sum(dat$ResponseYesNo[dat$Type=="control"]>0),sum(dat$ResponseYesNo[dat$Type=="call"]>0)),
  HeadsUp=c(sum(dat$HeadsUpYesNo[dat$Type=="control"]>0),sum(dat$HeadsUpYesNo[dat$Type=="call"]>0)),
  Vocal=c(sum(dat$VocalYesNo[dat$Type=="control"]>0),sum(dat$VocalYesNo[dat$Type=="call"]>0)),
  Approach=c(sum(dat$ApproachYesNo[dat$Type=="control"]>0),sum(dat$ApproachYesNo[dat$Type=="call"]>0))
)

# per context
contexts<-unique(dat$Context)
types<-unique(dat$Type)
for (c in (1:length(contexts))){
  for (t in (1:length(types))){
    print(contexts[c])
    print(types[t])
    tab<-dat[dat$Context==contexts[c]&dat$Type==types[t],]
    print(paste("N",dim(tab)[1])) 
    print(paste("sum R",sum(tab$ResponseYesNo)) )
    print(paste("% H",(sum(tab$HeadsUpYesNo)/sum(tab$ResponseYesNo))*100) )
    print(paste("% V",(sum(tab$VocalYesNo)/sum(tab$ResponseYesNo))*100 ))
    print(paste("% A",(sum(tab$ApproachYesNo)/sum(tab$ResponseYesNo))*100 ))
  }
}

# stats on latency and duration
for (c in (1:length(contexts))){
  for (t in (1:length(types))){
    print(contexts[c])
    print(types[t])
    tab<-dat[dat$Context==contexts[c]&dat$Type==types[t],]
    print("H latency")
    print(c(summary(tab$HeadsUpLatency[tab$HeadsUpYesNo>0]), sd(tab$HeadsUpLatency[tab$HeadsUpYesNo>0]))) 
    print("H duration")
    print(c(summary(tab$HeadsUpDuration[tab$HeadsUpYesNo>0]), sd(tab$HeadsUpDuration[tab$HeadsUpYesNo>0])))
    print("V latency")
    print(c(summary(tab$VocalLatency[tab$VocalYesNo>0]), sd(tab$VocalLatency[tab$VocalYesNo>0]))) 
    print("V N calls ")
    print(c(summary(tab$VocalNcalls[tab$VocalYesNo>0]), sd(tab$VocalNcalls[tab$VocalYesNo>0]))) 
    print("A latency")
    print(c(summary(tab$ApproachLatency[tab$ApproachYesNo>0]), sd(tab$ApproachLatency[tab$ApproachYesNo>0]))) 
    print("A duration")
    print(c(summary(tab$ApproachDuration[tab$ApproachYesNo>0]), sd(tab$ApproachDuration[tab$ApproachYesNo>0])))
  }
}


### figure % of responses ####
dat <- dat %>%
  mutate(ResponseType = case_when(
    HeadsUpYesNo == 1 & ApproachYesNo == 0 & VocalYesNo == 0 ~ "H",
    HeadsUpYesNo == 0 & ApproachYesNo == 0 & VocalYesNo == 1 ~ "V",
    HeadsUpYesNo == 0 & ApproachYesNo == 1 & VocalYesNo == 0 ~ "A",
    HeadsUpYesNo == 1 & ApproachYesNo == 0 & VocalYesNo == 1 ~ "HV",
    HeadsUpYesNo == 1 & ApproachYesNo == 1 & VocalYesNo == 0 ~ "HA",
    HeadsUpYesNo == 0 & ApproachYesNo == 1 & VocalYesNo == 1 ~ "VA",
    HeadsUpYesNo == 1 & ApproachYesNo == 1 & VocalYesNo == 1 ~ "HVA",
    TRUE ~ "none"))

tot <- dat %>%
  group_by(Type,Context) %>%
  summarise(N = n())

tab <- dat %>%
  group_by(Context, Type, ResponseType) %>%
  summarise( R = sum(ResponseYesNo)
             ) %>%
  filter(ResponseType != "none")

tab <- tab %>%
  left_join(tot, by = c("Type","Context")) %>%
  mutate(Percentage = (R/N)*100)

tab$Type<-factor(tab$Type,levels=c("control","call"))
tab$Context<-factor(tab$Context,levels=c("commuting","searching","after"))
tab$ResponseType<-factor(tab$ResponseType,levels=c("A","HA","HVA","VA","HV","V","H"))

colors <- c("H" = "#000000","A" = "#F0E442","V" = "#0072B2",
              "HA" = "#E69F00","HV" = "#009E73","VA" = "#56B4E9",
              "HVA" = "#D55E00")

# tiff(filename='Percentage_response.tiff', width=2400, height=2400, res=300)
png(filename='Percentage_response.png', width=720, height=480)
ggplot(tab, aes(x = interaction(Type, Context), y = Percentage, fill = ResponseType)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = colors) +
  ylim(0,100)+
  labs(x="",y = "Percentage of response (%)",fill = "Response") +
  facet_wrap("Context",strip.position="top",nrow=1,scales="free_x",
             labeller=labeller(Context=c("commuting"="Commuting","searching"="Searching","after"="After feeding")))+
  scale_x_discrete(labels = c("Control","Call"))+
  theme_bw(base_size=20) 
dev.off()


### N penguins ####
summary(dat$Npenguins)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1.00    3.00    8.00   10.76   15.00   40.00 
png(filename='Npenguins.png', width=480, height=480)
ggplot(data=dat[dat$Type=="call",], aes(x=Npenguins)) +
  geom_histogram(binwidth=2, color="black",fill="black") +
  labs(x = "N penguins", y = 'Count (N experiments)') +
  theme_bw(base_size = 20)
dev.off()

### Distances ####
summary(dat$Distance)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 5.00   50.00   80.00   80.32  100.00  200.00

(sum(dat$HeadsUpYesNo==1)/length(unique(dat$expID)))*100
# 49% of positive responses overall including call and control
(sum(dat$VocalYesNo==1)/length(unique(dat$expID)))*100
# 58% of positive responses overall including call and control
(sum(dat$ApproachYesNo==1)/length(unique(dat$expID)))*100
# 17% of positive responses overall including call and control

### HEADS UP ####

# data zero inflated
png(filename='Headsup_hist.png', width=480, height=240)
ggplot(dat,aes(x=HeadsUpDuration))+ 
  geom_histogram(binwidth=2,color='black',fill='black') +
  labs(x = "Heads up Duration", y = 'Count (N observations)') +
  theme_bw(base_size = 15)
dev.off()

(sum(dat$HeadsUpDuration>0)/dim(dat)[1])*100
# 24% of positive responses

# data overdispersed
mean(dat$HeadsUpDuration) # 5.856115
var(dat$HeadsUpDuration) # 163.2283
# var >> mean so we use the negative binomial distribution

table(dat$HeadsUpYesNo,dat$CallType)
#   control flat mod  tv
# 0     126   27  18  39
# 1      13   21  15  19
#     9%   44%  45%  33%

#### run models ####
set.seed(54984)
options(mc.cores=4)

H_brm_hurdle_normal1 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                          hu~CallType),
                      data = dat, family = hurdle_negbinomial(),
                      prior=c(prior(normal(0,1), class=b)),
                      warmup = 5000, iter = 20000, chains = 4,
                      control = list(adapt_delta = 0.99),
                      save_pars=save_pars(all=T))

H_brm_hurdle_normal15 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                      hu~CallType),
                      data = dat, family = hurdle_negbinomial(),
                      prior=c(prior(normal(0,1.5), class=b)),
                      warmup = 5000, iter = 20000, chains = 4,
                      control = list(adapt_delta = 0.99),
                      save_pars=save_pars(all=T))

H_brm_hurdle_normal2 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                   hu~CallType),
                               data = dat, family = hurdle_negbinomial(),
                               prior=c(prior(normal(0,2), class=b)),
                               warmup = 5000, iter = 20000, chains = 4,
                               control = list(adapt_delta = 0.99),
                              save_pars=save_pars(all=T))

H_brm_hurdle_student11 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_negbinomial(),
                                prior=c(prior(student_t(1,0,1), class=b)),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

H_brm_hurdle_student12 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_negbinomial(),
                                prior=c(prior(student_t(1,0,2), class=b)),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

H_brm_hurdle_student31 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                  hu~CallType),
                              data = dat, family = hurdle_negbinomial(),
                              prior=c(prior(student_t(3,0,1), class=b)),
                              warmup = 5000, iter = 20000, chains = 4,
                              control = list(adapt_delta = 0.99),
                              save_pars=save_pars(all=T))

H_brm_hurdle_student32 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_negbinomial(),
                                prior=c(prior(student_t(3,0,2), class=b)),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

H_brm_hurdle_student51 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_negbinomial(),
                                prior=c(prior(student_t(5,0,1), class=b)),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

H_brm_hurdle_student52 <- brm ( bf (HeadsUpDuration ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_negbinomial(),
                                prior=c(prior(student_t(5,0,2), class=b)),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

save(H_brm_hurdle_normal1,H_brm_hurdle_normal15,H_brm_hurdle_normal2,
     H_brm_hurdle_student11,H_brm_hurdle_student12,H_brm_hurdle_student31,
     H_brm_hurdle_student32,H_brm_hurdle_student51,H_brm_hurdle_student52,
     file="H_brm_models.Rdata")

#### comparison and selection ####
options(future.globals.maxSize= 4000 * 1024^2)

Hloo_n1<-loo(H_brm_hurdle_normal1,moment_match=T,reloo=T,cores=4)
Hloo_n15<-loo(H_brm_hurdle_normal15,moment_match=T,reloo=T,cores=4)
Hloo_n2<-loo(H_brm_hurdle_normal2,moment_match=T,reloo=T,cores=4)
Hloo_s11<-loo(H_brm_hurdle_student11,moment_match=T,reloo=T,cores=4)
Hloo_s12<-loo(H_brm_hurdle_student12,moment_match=T,reloo=T,cores=4)
Hloo_s31<-loo(H_brm_hurdle_student31,moment_match=T,reloo=T,cores=4)
Hloo_s32<-loo(H_brm_hurdle_student32,moment_match=T,reloo=T,cores=4)
Hloo_s51<-loo(H_brm_hurdle_student51,moment_match=T,reloo=T,cores=4)
Hloo_s52<-loo(H_brm_hurdle_student52,moment_match=T,reloo=T,cores=4)

loo_compare(Hloo_n1,Hloo_n15,Hloo_n2,Hloo_s11,Hloo_s12,
            Hloo_s31,Hloo_s32,Hloo_s51,Hloo_s52)
# elpd_diff se_diff
# H_brm_hurdle_student11  0.0       0.0   
# H_brm_hurdle_student31 -0.5       0.3   
# H_brm_hurdle_student51 -0.5       0.5   
# H_brm_hurdle_normal1   -0.6       0.5   
# H_brm_hurdle_normal15  -1.3       0.6   
# H_brm_hurdle_normal2   -1.4       0.8   
# H_brm_hurdle_student52 -1.5       0.9   
# H_brm_hurdle_student12 -1.6       0.7   
# H_brm_hurdle_student32 -2.2       0.8

# model selection: using prior student_t(1,0,1)
H_brm<-H_brm_hurdle_student11


#### diagnostics ####
plot(H_brm)

png(filename='H_brm_ppcheck.png', width=240, height=240)
pp_check(H_brm,ndraws=100,type='ecdf_overlay')
dev.off()

png(filename='H_brm_res.png', width=240, height=240)
hist(residuals(H_brm), main = "Residual histogram")
dev.off()

png(filename='H_brm_res_hu.png', width=240, height=240)
hist(residuals(H_brm,part="hu"), main = "Residual histogram")
dev.off()


#### results ####
summary(H_brm)

# probability of positive response from hurdle
# control
(1-plogis(fixef(H_brm)["hu_Intercept",1]))*100
# flat call
(1-plogis(fixef(H_brm)["hu_Intercept",1] + fixef(H_brm)["hu_CallTypeflat",1]))*100
# modulated call
(1-plogis(fixef(H_brm)["hu_Intercept",1] + fixef(H_brm)["hu_CallTypemod",1]))*100
# two-voice call
(1-plogis(fixef(H_brm)["hu_Intercept",1] + fixef(H_brm)["hu_CallTypetv",1]))*100


# extract estimates from model using conditional_effects function
ce<-conditional_effects(H_brm)
fitH<-ce[["Context:CallType"]]
fitH # to extract estimate values reported in main text
fitH$Context<-factor(fitH$Context,levels=c("commuting","searching","after"))

datsumH <- dat %>%
  group_by(Context, CallType) %>%
  summarise(n=n(),
            mean = mean(HeadsUpDuration),
            sd=sd(HeadsUpDuration),
            se=sd/sqrt(n),
            low=max(mean - 1.96*se,0),
            up=mean + 1.96*se)
datsumH$Context<-factor(datsumH$Context,levels=c("commuting","searching","after"))

tiff(filename='HeadsUp_res.tiff', width=4800, height=2400, res=300)
# png(filename='Headsup_res.png', width=960, height=480)
ggplot() +
  geom_point(data=datsumH,aes(x=CallType, y=mean),size=8, shape=19,color="black",
             position=position_nudge(x=-0.05)) +
  geom_errorbar(data=datsumH,aes(x=CallType,ymin=low,ymax=up),linewidth=1,color="black",
                position=position_nudge(x=-0.05))+
  geom_point(data=fitH,aes(x=CallType,y=estimate__),size=8, shape=18,color="darkgrey",
             position=position_nudge(x=+0.05)) +
  geom_errorbar(data=fitH,aes(x=CallType,ymin=lower__,ymax=upper__),linewidth=1,color="darkgrey",
                position=position_nudge(x=+0.05))+
  facet_wrap(~Context,labeller=labeller(Context=c("commuting"="Commuting","searching"="Searching","after"="After feeding")))+
  labs(x = "Call type per context", y = 'Heads up duration (s)') +
  scale_x_discrete(labels = c("control" = "Control","flat" = "Flat","mod" = "Modulated","tv" = "Two-voice")) +
  theme_bw(base_size = 20)+
  theme(legend.position = "none",axis.text.x = element_text(size = 14))+
  geom_text(data = datsumH, aes(x = CallType, y = max(datsumH$up)+1 , label = n),size = 6, vjust = 0)
dev.off()

rm(ce,fitH,datsumH)

### VOCAL ####

# data zero inflated
png(filename='Vocal_hist.png', width=480, height=240)
ggplot(dat,aes(x=VocalNcalls))+ 
  geom_histogram(binwidth=2,color='black',fill='black') +
  labs(x = "Vocal activity (N calls)", y = 'Count (N observations)') +
  theme_bw(base_size = 15)
dev.off()

(sum(dat$VocalNcalls>0)/dim(dat)[1])*100
# 31% of positive responses

# not as overdispersed as Heads up
mean(dat$VocalNcalls) # 1.151079
var(dat$VocalNcalls) # 6.265901


table(dat$VocalYesNo,dat$CallType)
#     control flat mod  tv
# 0     134   22  17  24
# 1       5   26  16  34
#     4%   54%  48%  59%

#### run models ####
set.seed(54984)
options(mc.cores=4)

V_brm_hurdle_normal1 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                          hu~CallType),
                      data = dat, family = hurdle_poisson(),
                      prior=c(prior(normal(0,1),class=b)),
                      warmup = 5000, iter = 20000, chains = 4,
                      control = list(adapt_delta = 0.99),
                      save_pars=save_pars(all=T))

V_brm_hurdle_normal15 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                           hu~CallType),
                       data = dat, family = hurdle_poisson(),
                       prior=c(prior(normal(0,1.5),class=b)),
                       warmup = 5000, iter = 20000, chains = 4,
                       control = list(adapt_delta = 0.99),
                       save_pars=save_pars(all=T))

V_brm_hurdle_normal2 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                           hu~CallType),
                       data = dat, family = hurdle_poisson(),
                       prior=c(prior(normal(0,2),class=b)),
                       warmup = 5000, iter = 20000, chains = 4,
                       control = list(adapt_delta = 0.99),
                       save_pars=save_pars(all=T))

V_brm_hurdle_normal3 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                  hu~CallType),
                              data = dat, family = hurdle_poisson(),
                              prior=c(prior(normal(0,3),class=b)),
                              warmup = 5000, iter = 20000, chains = 4,
                              control = list(adapt_delta = 0.99),
                              save_pars=save_pars(all=T))

V_brm_hurdle_student11 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_poisson(),
                                prior=c(prior(student_t(1,0,1),class="b")),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

V_brm_hurdle_student12 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_poisson(),
                                prior=c(prior(student_t(1,0,2),class="b")),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

V_brm_hurdle_student31 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_poisson(),
                                prior=c(prior(student_t(3,0,1),class="b")),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

V_brm_hurdle_student32 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_poisson(),
                                prior=c(prior(student_t(3,0,2),class="b")),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

V_brm_hurdle_student51 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_poisson(),
                                prior=c(prior(student_t(5,0,1),class="b")),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

V_brm_hurdle_student52 <- brm ( bf (VocalNcalls ~ Context * CallType + VocalNcallsBefore + scale(Npenguins) + scale(Distance) + (1| CallType:PlaybackTrack) + (1|expID),
                                    hu~CallType),
                                data = dat, family = hurdle_poisson(),
                                prior=c(prior(student_t(5,0,2),class="b")),
                                warmup = 5000, iter = 20000, chains = 4,
                                control = list(adapt_delta = 0.99),
                                save_pars=save_pars(all=T))

#### comparison and selection ####
options(future.globals.maxSize= 4000 * 1024^2)

Vloo_n1<-loo(V_brm_hurdle_normal1,moment_match=T,reloo=T,cores=4)
Vloo_n15<-loo(V_brm_hurdle_normal15,moment_match=T,reloo=T,cores=4)
Vloo_n2<-loo(V_brm_hurdle_normal2,moment_match=T,reloo=T,cores=4)
Vloo_n3<-loo(V_brm_hurdle_normal3,moment_match=T,reloo=T,cores=4)
Vloo_s11<-loo(V_brm_hurdle_student11,moment_match=T,reloo=T,cores=4)
Vloo_s12<-loo(V_brm_hurdle_student12,moment_match=T,reloo=T,cores=4)
Vloo_s31<-loo(V_brm_hurdle_student31,moment_match=T,reloo=T,cores=4)
Vloo_s32<-loo(V_brm_hurdle_student32,moment_match=T,reloo=T,cores=4)
Vloo_s51<-loo(V_brm_hurdle_student51,moment_match=T,reloo=T,cores=4)
Vloo_s52<-loo(V_brm_hurdle_student52,moment_match=T,reloo=T,cores=4)
Vloo_l05<-loo(V_brm_hurdle_laplace05,moment_match=T,reloo=T,cores=4)
Vloo_l1<-loo(V_brm_hurdle_laplace1,moment_match=T,reloo=T,cores=4)

loo_compare(Vloo_n1,Vloo_n15,Vloo_n2,Vloo_n3,
            Vloo_s11,Vloo_s12,Vloo_s31,Vloo_s32,Vloo_s51,Vloo_s52)
# elpd_diff se_diff
# V_brm_hurdle_normal2    0.0       0.0   
# V_brm_hurdle_student32 -0.2       0.3   
# V_brm_hurdle_normal3   -0.3       0.6   
# V_brm_hurdle_student52 -0.3       0.3   
# V_brm_hurdle_student12 -1.1       0.5   
# V_brm_hurdle_normal15  -1.6       1.1   
# V_brm_hurdle_student51 -2.0       1.7   
# V_brm_hurdle_student11 -2.0       1.5   
# V_brm_hurdle_normal1   -2.0       1.8   
# V_brm_hurdle_student31 -2.2       1.7  

# model selection: using prior normal(0,2)
V_brm<-V_brm_hurdle_normal2

#### diagnostics ####
plot(V_brm)

png(filename='V_brm_ppcheck.png', width=240, height=240)
pp_check(V_brm,ndraws=100,type='ecdf_overlay')
dev.off()

png(filename='V_brm_res.png', width=240, height=240)
hist(residuals(V_brm), main = "Residual histogram")
dev.off()

#### results ####
summary(V_brm)

# probability of positive response from hurdle
# control
(1-plogis( fixef(V_brm)["hu_Intercept",1]))*100
# flat call
(1-plogis(fixef(V_brm)["hu_Intercept",1] + fixef(V_brm)["hu_CallTypeflat",1]))*100
# modulated call
(1-plogis(fixef(V_brm)["hu_Intercept",1] + fixef(V_brm)["hu_CallTypemod",1]))*100
# two-voice call
(1-plogis(fixef(V_brm)["hu_Intercept",1] + fixef(V_brm)["hu_CallTypetv",1]))*100


# extract estimates from model using conditional_effects function
ce<-conditional_effects(V_brm)
fitV<-ce[["Context:CallType"]]
fitV[,c(1,2,14,16,17)]
# compare with real data 
dat %>% group_by(Context, CallType) %>% summarise(mean(VocalNcalls), .groups="drop")

fitV$Context<-factor(fitV$Context,levels=c("commuting","searching","after"))

datsumV <- dat %>%
  group_by(Context, CallType) %>%
  summarise(n=n(),
            mean = mean(VocalNcalls),
            sd=sd(VocalNcalls),
            se=sd/sqrt(n),
            low=max(mean - 1.96*se,0),
            up=mean + 1.96*se)
datsumV$Context<-factor(datsumV$Context,levels=c("commuting","searching","after"))

tiff(filename='Vocal_res.tiff', width=4800, height=2400, res=300)
# png(filename='Vocal_res.png', width=960, height=480)
ggplot() +
  geom_point(data=datsumV,aes(x=CallType, y=mean),size=8, shape=19,color="black",
             position=position_nudge(x=-0.05)) +
  geom_errorbar(data=datsumV,aes(x=CallType,ymin=low,ymax=up),linewidth=1,color="black",
                position=position_nudge(x=-0.05))+
  geom_point(data=fitV,aes(x=CallType,y=estimate__),size=8, shape=18,color="darkgrey",
             position=position_nudge(x=+0.05)) +
  geom_errorbar(data=fitV,aes(x=CallType,ymin=lower__,ymax=upper__),linewidth=1,color="darkgrey",
                position=position_nudge(x=+0.05))+
  facet_wrap(~Context,labeller=labeller(Context=c("commuting"="Commuting","searching"="Searching","after"="After feeding")))+
  labs(x = "Call type per context", y = 'Vocal activity (N calls)') +
  scale_x_discrete(labels = c("control" = "Control","flat" = "Flat","mod" = "Modulated","tv" = "Two-voice")) +
  scale_y_continuous(breaks=seq(0,10,by=2))+
  theme_bw(base_size = 20)+
  theme(legend.position = "none",axis.text.x = element_text(size = 14))+
  geom_text(data = datsumV, aes(x = CallType, y = max(datsumV$up)+0.5 , label = n),size = 6, vjust = 0) 
dev.off()

rm(fitV,ce,datsumV)


### APPROACH ####
(sum(dat$ApproachDuration>0)/dim(dat)[1])*100
# 8% of positive responses

table(dat$ApproachYesNo,dat$CallType)
# control flat mod  tv
# 0     139   44  25  47
# 1       0    4   8  11
#     0%   8%  24%  19%

table(dat$ApproachYesNo,dat$PlaybackTrack)

# dat$ApproachYesNo  = 0 or 1 (binomial response)
hist(dat$ApproachDuration,breaks=30) # data zero inflated
mean(dat$HeadsUpDuration) # 5.856115
var(dat$HeadsUpDuration) # 163.2283
# var >> mean so we use the negative binomial distribution

#### run models ####
set.seed(54984)
options(mc.cores=4)

A_brm_binom_normal1 <- brm ( formula = ApproachYesNo ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1|expID),
                         data = dat, family = bernoulli(),
                         prior=c(prior(normal(0,1),class="b")),
                         warmup = 5000, iter = 20000, chains = 4,
                         control = list(adapt_delta = 0.99),
                         save_pars=save_pars(all=T))

A_brm_binom_normal15 <- brm ( formula = ApproachYesNo ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1|expID),
                             data = dat, family = bernoulli(),
                             prior=c(prior(normal(0,1.5),class="b")),
                             warmup = 5000, iter = 20000, chains = 4,
                             control = list(adapt_delta = 0.99),
                             save_pars=save_pars(all=T))

A_brm_binom_student11 <- brm ( formula = ApproachYesNo ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1|expID),
                               data = dat, family = bernoulli(),
                               prior=c(prior(student_t(1,0,1),class="b")),
                               warmup = 5000, iter = 20000, chains = 4,
                               control = list(adapt_delta = 0.99),
                               save_pars=save_pars(all=T))

A_brm_binom_student31 <- brm ( formula = ApproachYesNo ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1|expID),
                      data = dat, family = bernoulli(),
                      prior=c(prior(student_t(3,0,1),class="b")),
                      warmup = 5000, iter = 20000, chains = 4,
                      control = list(adapt_delta = 0.99),
                      save_pars=save_pars(all=T))

A_brm_binom_student51 <- brm ( formula = ApproachYesNo ~ Context * CallType + scale(Npenguins) + scale(Distance) + (1|expID),
                               data = dat, family = bernoulli(),
                               prior=c(prior(student_t(5,0,1),class="b")),
                               warmup = 5000, iter = 20000, chains = 4,
                               control = list(adapt_delta = 0.99),
                               save_pars=save_pars(all=T))

#### comparison and selection ####
options(future.globals.maxSize= 4000 * 1024^2)

Aloo_n1<-loo(A_brm_binom_normal1,moment_match=T,reloo=T,cores=4)
Aloo_n15<-loo(A_brm_binom_normal15,moment_match=T,reloo=T,cores=4)
Aloo_s11<-loo(A_brm_binom_student11,moment_match=T,reloo=T,cores=4)
Aloo_s31<-loo(A_brm_binom_student31,moment_match=T,reloo=T,cores=4)
Aloo_s51<-loo(A_brm_binom_student51,moment_match=T,reloo=T,cores=4)

loo_compare(Aloo_n1,Aloo_n15,Aloo_s11,Aloo_s31,Aloo_s51)
# elpd_diff se_diff
# A_brm_binom_student11  0.0       0.0   
# A_brm_binom_normal15  -0.9       0.7   
# A_brm_binom_student31 -1.0       0.5   
# A_brm_binom_student51 -1.2       0.7   
# A_brm_binom_normal1   -1.7       1.0    

# model selection: using prior student_t(1,0,1)
A_brm<-A_brm_binom_student11

#### diagnostics ####
plot(A_brm)

png(filename='A_brm_ppcheck.png', width=240, height=240)
pp_check(A_brm,ndraws=100,type='ecdf_overlay')
dev.off()

png(filename='A_brm_res.png', width=240, height=240)
hist(residuals(A_brm), main = "Residual histogram")
dev.off()

#### results ####
summary(A_brm)

# figure

# extract estimates from model using conditional_effects function
ce<-conditional_effects(A_brm)
fitA<-ce[["Context:CallType"]]
fitA[,c(1,2,12,14,15)]
fitA$Context<-factor(fitA$Context,levels=c("commuting","searching","after"))

datsumA <- dat %>%
  group_by(Context, CallType) %>%
  summarise(n=n(),
            s=sum(ApproachYesNo),
            mean=s/n,
            low=binom.test(s,n)$conf.int[1],
            up=binom.test(s,n)$conf.int[2])
datsumA$Context<-factor(datsumA$Context,levels=c("commuting","searching","after"))

tiff(filename='Approach_res.tiff', width=4800, height=2400, res=300)
# png(filename='Approach_res.png', width=960, height=480)
ggplot() +
  geom_point(data=datsumA,aes(x=CallType, y=mean),size=8, shape=19,color="black",
             position=position_nudge(x=-0.05)) +
  geom_errorbar(data=datsumA,aes(x=CallType,ymin=low,ymax=up),linewidth=1,color="black",
                position=position_nudge(x=-0.05))+
  geom_point(data=fitA,aes(x=CallType,y=estimate__),size=7, shape=18,color="darkgrey",
             position=position_nudge(x=+0.05)) +
  geom_errorbar(data=fitA,aes(x=CallType,ymin=lower__,ymax=upper__),linewidth=1,color="darkgrey",
                position=position_nudge(x=+0.05))+
  facet_wrap(~Context,labeller=labeller(Context=c("commuting"="Commuting","searching"="Searching","after"="After feeding")))+
  labs(x = "Call type per context", y = 'Probability of approach') +
  scale_x_discrete(labels = c("control" = "Control","flat" = "Flat","mod" = "Modulated","tv" = "Two-voice")) +
  ylim(0,1)+
  theme_bw(base_size = 20)+
  theme(legend.position = "none",axis.text.x = element_text(size = 14))+
  geom_text(data = datsumA, aes(x = CallType, y = 1, label = n),size = 6, vjust = 0) 
dev.off()

rm(ce,fitA,datsumA)

### TYPE OF CALL #######

##### commuting ####
commuting<-dat[dat$Context=="commuting"&dat$Type=="call"&dat$VocalYesNo>0,]
tabc <- melt(table(commuting$CallType,commuting$VR1))

tiff(filename='Transition_commuting.tiff', width=3600, height=2400, res=300)
# png(filename='Transition_commuting.png', width=720, height=480)
ggplot(data = tabc, aes(x=Var2, y=Var1, fill=value)) +
  geom_tile(color="white") + 
  geom_text(aes(label=value), color=ifelse(tabc$value >= max(tabc$value)-2, "white", "black"), size=12) +  
  scale_fill_gradient(low="white", high="black") + 
  labs(title=paste("Commuting (N=",sum(tabc$value),")",sep=""),x="Response call type", y="Playback call type", fill="Count") +
  scale_x_discrete(labels=c("Flat","Modulated","Two-voice"),position="top")+
  scale_y_discrete(limits=rev(levels(tabc$Var1)),labels=c("Two-voice","Modulated","Flat"))+
  theme_minimal(base_size = 20)+
  theme(legend.text=element_text(size=22),axis.text=element_text(size=20),axis.text.y=element_text(angle=90,hjust=0.5))
dev.off()

##### searching ####
searching<-dat[dat$Context=="searching"&dat$Type=="call"&dat$VocalYesNo>0,]
tabs <- melt(table(searching$CallType,searching$VR1))

tiff(filename='Transition_searching.tiff', width=3600, height=2400, res=300)
# png(filename='Transition_searching.png', width=720, height=480)
ggplot(data = tabs, aes(x=Var2, y=Var1, fill=value)) +
  geom_tile(color="white") + 
  geom_text(aes(label=value), color=ifelse(tabs$value >= max(tabs$value)-2, "white", "black"), size=8) +  
  scale_fill_gradient(low="white", high="black",breaks=seq(0,10,by=2)) + 
  labs(title=paste("Searching (N=",sum(tabs$value),")",sep=""),x="Response call type", y="Playback call type", fill="Count") +
  scale_x_discrete(labels=c("Flat","Modulated","Two-voice"),position="top")+
  scale_y_discrete(limits=rev(levels(tabs$Var1)),labels=c("Two-voice","Modulated","Flat"))+
  theme_minimal(base_size = 20)+
  theme(legend.text=element_text(size=22),axis.text=element_text(size=20),axis.text.y=element_text(angle=90,hjust=0.5))
dev.off()

##### after feeding ####
after<-dat[dat$Context=="after"&dat$Type=="call"&dat$VocalYesNo>0,]
taba <- melt(table(after$CallType,after$VR1))

tiff(filename='Transition_after.tiff', width=3600, height=2400, res=300)
png(filename='Transition_after.png', width=720, height=480)
ggplot(data = taba, aes(x=Var2, y=Var1, fill=value)) +
  geom_tile(color="white") + 
  geom_text(aes(label=value), color=ifelse(taba$value >= max(taba$value)-1, "white", "black"), size=8) +  
  scale_fill_gradient(low="white", high="black") + 
  labs(title=paste("After feeding (N=",sum(taba$value),")",sep=""),x="Response call type", y="Playback call type", fill="Count") +
  scale_x_discrete(labels=c("Flat","Modulated","Two-voice"),position="top")+
  scale_y_discrete(limits=rev(levels(taba$Var1)),labels=c("Two-voice","Modulated","Flat"))+
  theme_minimal(base_size = 20)+
  theme(legend.text=element_text(size=22),axis.text=element_text(size=20),axis.text.y=element_text(angle=90,hjust=0.5))
dev.off()


