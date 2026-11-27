library(ggplot2)

####### Figure 5 in main text of article #####
##### load data #####
flat.spectro.corr1m<-read.csv(file="Data_correlation1m_flat_call_spectrogram.csv")
mod.spectro.corr1m<-read.csv(file="Data_correlation1m_modulated_call_spectrogram.csv")
tv.spectro.corr1m<-read.csv(file="Data_correlation1m_twovoice_call_spectrogram.csv")

##### calculate mean sd ####
# distances for all 
d<-c(4,8,16,32,64,128,256)

###### flat ######
flat.spectro<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),rep(64,300),rep(128,300),rep(256,300))
flat.spectro<-cbind(flat.spectro,
                    c(flat.spectro.corr1m[,1],flat.spectro.corr1m[,2],flat.spectro.corr1m[,3],flat.spectro.corr1m[,4],
                      flat.spectro.corr1m[,5],flat.spectro.corr1m[,6],flat.spectro.corr1m[,7]))
colnames(flat.spectro)<-c("dist","corr")
flat.spectro<-as.data.frame(flat.spectro)
flat.spectro<-na.omit(flat.spectro)

m<-c(mean(flat.spectro.corr1m[,1],na.rm=T),mean(flat.spectro.corr1m[,2],na.rm=T),
     mean(flat.spectro.corr1m[,3],na.rm=T),mean(flat.spectro.corr1m[,4],na.rm=T),
     mean(flat.spectro.corr1m[,5],na.rm=T),mean(flat.spectro.corr1m[,6],na.rm=T),
     mean(flat.spectro.corr1m[,7],na.rm=T))
s<-c(sd(flat.spectro.corr1m[,1],na.rm=T),sd(flat.spectro.corr1m[,2],na.rm=T),
     sd(flat.spectro.corr1m[,3],na.rm=T),sd(flat.spectro.corr1m[,4],na.rm=T),
     sd(flat.spectro.corr1m[,5],na.rm=T),sd(flat.spectro.corr1m[,6],na.rm=T),
     sd(flat.spectro.corr1m[,7],na.rm=T))

flatTRO<-as.data.frame(cbind(d,m,s))
rm(m,s)

###### mod ######
mod.spectro<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),rep(64,300),rep(128,300),rep(256,300))
mod.spectro<-cbind(mod.spectro,
                 c(mod.spectro.corr1m[,1],mod.spectro.corr1m[,2],mod.spectro.corr1m[,3],mod.spectro.corr1m[,4],
                   mod.spectro.corr1m[,5],mod.spectro.corr1m[,6],mod.spectro.corr1m[,7]))
colnames(mod.spectro)<-c("dist","corr")
mod.spectro<-as.data.frame(mod.spectro)
mod.spectro<-na.omit(mod.spectro)

m<-c(mean(mod.spectro.corr1m[,1],na.rm=T),mean(mod.spectro.corr1m[,2],na.rm=T),
     mean(mod.spectro.corr1m[,3],na.rm=T),mean(mod.spectro.corr1m[,4],na.rm=T),
     mean(mod.spectro.corr1m[,5],na.rm=T),mean(mod.spectro.corr1m[,6],na.rm=T),
     mean(mod.spectro.corr1m[,7],na.rm=T))
s<-c(sd(mod.spectro.corr1m[,1],na.rm=T),sd(mod.spectro.corr1m[,2],na.rm=T),
     sd(mod.spectro.corr1m[,3],na.rm=T),sd(mod.spectro.corr1m[,4],na.rm=T),
     sd(mod.spectro.corr1m[,5],na.rm=T),sd(mod.spectro.corr1m[,6],na.rm=T),
     sd(mod.spectro.corr1m[,7],na.rm=T))

modTRO<-as.data.frame(cbind(d,m,s))
rm(m,s)

###### tv ######
tv.spectro<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),rep(64,300),rep(128,300),rep(256,300))
tv.spectro<-cbind(tv.spectro,
                  c(tv.spectro.corr1m[,1],tv.spectro.corr1m[,2],tv.spectro.corr1m[,3],tv.spectro.corr1m[,4],
                        tv.spectro.corr1m[,5],tv.spectro.corr1m[,6],tv.spectro.corr1m[,7]))
colnames(tv.spectro)<-c("dist","corr")
tv.spectro<-as.data.frame(tv.spectro)
tv.spectro<-na.omit(tv.spectro)

m<-c(mean(tv.spectro.corr1m[,1],na.rm=T),mean(tv.spectro.corr1m[,2],na.rm=T),
     mean(tv.spectro.corr1m[,3],na.rm=T),mean(tv.spectro.corr1m[,4],na.rm=T),
     mean(tv.spectro.corr1m[,5],na.rm=T),mean(tv.spectro.corr1m[,6],na.rm=T),
     mean(tv.spectro.corr1m[,7],na.rm=T))
s<-c(sd(tv.spectro.corr1m[,1],na.rm=T),sd(tv.spectro.corr1m[,2],na.rm=T),
     sd(tv.spectro.corr1m[,3],na.rm=T),sd(tv.spectro.corr1m[,4],na.rm=T),
     sd(tv.spectro.corr1m[,5],na.rm=T),sd(tv.spectro.corr1m[,6],na.rm=T),
     sd(tv.spectro.corr1m[,7],na.rm=T))

tvTRO<-as.data.frame(cbind(d,m,s))
rm(m,s)

##### figure #####
tiff(filename="./signal_degradation_nolegend.tiff",width=3000,height=3000,units="px",res=300,compression="none",bg="transparent")

ggplot(data = flatTRO, aes(x=d, y=m)) + 
  geom_point(data = flatTRO, aes(x=d, y=m), shape = 19, size=8, color="#7CCD7C") +
  geom_errorbar(data = flatTRO, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#7CCD7C") +
  geom_smooth(data = flat.spectro, aes(x=dist, y=corr),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=1,color="#7CCD7C") +
  geom_point(data = modTRO, aes(x=d, y=m), size=8, shape = 15, color="#8B4726") +
  geom_errorbar(data = modTRO, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#8B4726") +
  geom_smooth(data = mod.spectro, aes(x=dist, y=corr),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=1, color="#8B4726") +
  geom_point(data = tvTRO, aes(x=d, y=m), size=8, shape = 17, color="#27408B") +
  geom_errorbar(data = tvTRO, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#27408B") +
  geom_smooth(data = tv.spectro, aes(x=dist, y=corr),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=1, color="#27408B") +
  labs(x="Distance (m)", y="Correlation with reference (1m)",cex.lab=2) + 
  scale_x_continuous(breaks = c(4,16,32,64,128,256))+
  ylim(0,1)+
  theme_bw()+
  theme(axis.text=element_text(size=24),axis.title=element_text(size=28,face="bold"),legend.position='none')

dev.off()

####### Figure in Supplementary 4 #######
rm(list = ls())

##### load data ####
flat.spectro.corr1m<-read.csv(file="Data_correlation1m_flat_call_spectrogram.csv")
mod.spectro.corr1m<-read.csv(file="Data_correlation1m_modulated_call_spectrogram.csv")
tv.spectro.corr1m<-read.csv(file="Data_correlation1m_twovoice_call_spectrogram.csv")

flat.env.corr1m<-read.csv(file="Data_correlation1m_flat_call_envelope.csv")
mod.env.corr1m<-read.csv(file="Data_correlation1m_modulated_call_envelope.csv")
tv.env.corr1m<-read.csv(file="Data_correlation1m_twovoice_call_envelope.csv")

flat.spect.corr1m<-read.csv(file="Data_correlation1m_flat_call_amplitude_spectrum.csv")
mod.spect.corr1m<-read.csv(file="Data_correlation1m_modulated_call_amplitude_spectrum.csv")
tv.spect.corr1m<-read.csv(file="Data_correlation1m_twovoice_call_amplitude_spectrum.csv")

#### flat calculate ####
flat.env.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
                rep(64,300),rep(128,300),rep(256,300))
flat.env.all<-cbind(flat.env.all,
                    c(flat.env.corr1m[,1],flat.env.corr1m[,2],
                      flat.env.corr1m[,3],flat.env.corr1m[,4],
                      flat.env.corr1m[,5],flat.env.corr1m[,6],
                      flat.env.corr1m[,7]))
colnames(flat.env.all)<-c("dist","corr")
flat.env.all<-as.data.frame(flat.env.all)
flat.env.all<-na.omit(flat.env.all)
d<-c(4,8,16,32,64,128,256)

m<-c(mean(flat.env.corr1m[,1],na.rm=T),
     mean(flat.env.corr1m[,2],na.rm=T),
     mean(flat.env.corr1m[,3],na.rm=T),
     mean(flat.env.corr1m[,4],na.rm=T),
     mean(flat.env.corr1m[,5],na.rm=T),
     mean(flat.env.corr1m[,6],na.rm=T),
     mean(flat.env.corr1m[,7],na.rm=T))
s<-c(sd(flat.env.corr1m[,1],na.rm=T),
     sd(flat.env.corr1m[,2],na.rm=T),
     sd(flat.env.corr1m[,3],na.rm=T),
     sd(flat.env.corr1m[,4],na.rm=T),
     sd(flat.env.corr1m[,5],na.rm=T),
     sd(flat.env.corr1m[,6],na.rm=T),
     sd(flat.env.corr1m[,7],na.rm=T))
flatENV<-as.data.frame(cbind(d,m,s))
rm(m, s)

flat.spect.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
                  rep(64,300),rep(128,300),rep(256,300))
flat.spect.all<-cbind(flat.spect.all,
                      c(flat.spect.corr1m[,1],flat.spect.corr1m[,2],
                        flat.spect.corr1m[,3],flat.spect.corr1m[,4],
                        flat.spect.corr1m[,5],flat.spect.corr1m[,6],
                        flat.spect.corr1m[,7]))
colnames(flat.spect.all)<-c("dist","corr")
flat.spect.all<-as.data.frame(flat.spect.all)
flat.spect.all<-na.omit(flat.spect.all)

m<-c(mean(flat.spect.corr1m[,1],na.rm=T),
     mean(flat.spect.corr1m[,2],na.rm=T),
     mean(flat.spect.corr1m[,3],na.rm=T),
     mean(flat.spect.corr1m[,4],na.rm=T),
     mean(flat.spect.corr1m[,5],na.rm=T),
     mean(flat.spect.corr1m[,6],na.rm=T),
     mean(flat.spect.corr1m[,7],na.rm=T))
s<-c(sd(flat.spect.corr1m[,1],na.rm=T),
     sd(flat.spect.corr1m[,2],na.rm=T),
     sd(flat.spect.corr1m[,3],na.rm=T),
     sd(flat.spect.corr1m[,4],na.rm=T),
     sd(flat.spect.corr1m[,5],na.rm=T),
     sd(flat.spect.corr1m[,6],na.rm=T),
     sd(flat.spect.corr1m[,7],na.rm=T))
flatSPEC<-as.data.frame(cbind(d,m,s))

flat.spectro.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
                    rep(64,300),rep(128,300),rep(256,300))
flat.spectro.all<-cbind(flat.spectro.all,
                        c(flat.spectro.corr1m[,1],flat.spectro.corr1m[,2],
                          flat.spectro.corr1m[,3],flat.spectro.corr1m[,4],
                          flat.spectro.corr1m[,5],flat.spectro.corr1m[,6],
                          flat.spectro.corr1m[,7]))
colnames(flat.spectro.all)<-c("dist","corr")
flat.spectro.all<-as.data.frame(flat.spectro.all)
flat.spectro.all<-na.omit(flat.spectro.all)

m<-c(mean(flat.spectro.corr1m[,1],na.rm=T),
     mean(flat.spectro.corr1m[,2],na.rm=T),
     mean(flat.spectro.corr1m[,3],na.rm=T),
     mean(flat.spectro.corr1m[,4],na.rm=T),
     mean(flat.spectro.corr1m[,5],na.rm=T),
     mean(flat.spectro.corr1m[,6],na.rm=T),
     mean(flat.spectro.corr1m[,7],na.rm=T))
s<-c(sd(flat.spectro.corr1m[,1],na.rm=T),
     sd(flat.spectro.corr1m[,2],na.rm=T),
     sd(flat.spectro.corr1m[,3],na.rm=T),
     sd(flat.spectro.corr1m[,4],na.rm=T),
     sd(flat.spectro.corr1m[,5],na.rm=T),
     sd(flat.spectro.corr1m[,6],na.rm=T),
     sd(flat.spectro.corr1m[,7],na.rm=T))
flatTRO<-as.data.frame(cbind(d,m,s))

#### flat figure ####
tiff(filename="./signal_degradation_flat.tiff",width=3000,height=3000,units="px",res=300,compression="none",bg="transparent")

ggplot(data = flatSPEC, aes(x=d, y=m)) + 
  geom_point(data = flatSPEC, aes(y = m),size=8,color="#7CCD7C") +
  geom_errorbar(data = flatSPEC, aes(ymin=m-s, ymax=m+s), linewidth=0.8,color="#7CCD7C") +
  geom_smooth(data = flat.spect.all, aes(x=dist, y=corr,color="Amplitude spectrum"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  geom_point(data = flatENV, aes(x=d, y=m), size=8, color="#8B4726") +
  geom_errorbar(data = flatENV, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#8B4726") +
  geom_smooth(data = flat.env.all, aes(x=dist, y=corr, color="Envelope"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  geom_point(data = flatTRO, aes(x=d, y=m), size=8, color="#27408B") +
  geom_errorbar(data = flatTRO, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#27408B") +
  geom_smooth(data = flat.spectro.all, aes(x=dist, y=corr, color="Spectrogram"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  labs(x="Distance (m)", y="Correlation at 1m",cex.lab=2) + 
  scale_x_continuous(breaks = d)+
  ylim(0,1)+
  ggtitle("Flat")+
  scale_color_manual(values=c('#7CCD7C','#8B4726','#27408B')) +
  theme(legend.position = c(0.8,0.9), legend.text=element_text(size=24), legend.title=element_blank(),
        legend.key=element_rect(fill='black'))+
  theme(axis.text=element_text(size=24),axis.title=element_text(size=28,face="bold"),
        plot.title=element_text(size=28,face="bold",hjust=0.5))
dev.off()

#### mod calculate ####
mod.env.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
               rep(64,300),rep(128,300),rep(256,300))
mod.env.all<-cbind(mod.env.all,
                   c(mod.env.corr1m[,1],mod.env.corr1m[,2],
                     mod.env.corr1m[,3],mod.env.corr1m[,4],
                     mod.env.corr1m[,5],mod.env.corr1m[,6],
                     mod.env.corr1m[,7]))
colnames(mod.env.all)<-c("dist","corr")
mod.env.all<-as.data.frame(mod.env.all)
mod.env.all<-na.omit(mod.env.all)
d<-c(4,8,16,32,64,128,256)

m<-c(mean(mod.env.corr1m[,1],na.rm=T),
     mean(mod.env.corr1m[,2],na.rm=T),
     mean(mod.env.corr1m[,3],na.rm=T),
     mean(mod.env.corr1m[,4],na.rm=T),
     mean(mod.env.corr1m[,5],na.rm=T),
     mean(mod.env.corr1m[,6],na.rm=T),
     mean(mod.env.corr1m[,7],na.rm=T))
s<-c(sd(mod.env.corr1m[,1],na.rm=T),
     sd(mod.env.corr1m[,2],na.rm=T),
     sd(mod.env.corr1m[,3],na.rm=T),
     sd(mod.env.corr1m[,4],na.rm=T),
     sd(mod.env.corr1m[,5],na.rm=T),
     sd(mod.env.corr1m[,6],na.rm=T),
     sd(mod.env.corr1m[,7],na.rm=T))
d<-c(4,8,16,32,64,128,256)
modENV<-as.data.frame(cbind(d,m,s))
rm(m, s)

mod.spect.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
                 rep(64,300),rep(128,300),rep(256,300))
mod.spect.all<-cbind(mod.spect.all,
                     c(mod.spect.corr1m[,1],mod.spect.corr1m[,2],
                       mod.spect.corr1m[,3],mod.spect.corr1m[,4],
                       mod.spect.corr1m[,5],mod.spect.corr1m[,6],
                       mod.spect.corr1m[,7]))
colnames(mod.spect.all)<-c("dist","corr")
mod.spect.all<-as.data.frame(mod.spect.all)
mod.spect.all<-na.omit(mod.spect.all)

m<-c(mean(mod.spect.corr1m[,1],na.rm=T),
     mean(mod.spect.corr1m[,2],na.rm=T),
     mean(mod.spect.corr1m[,3],na.rm=T),
     mean(mod.spect.corr1m[,4],na.rm=T),
     mean(mod.spect.corr1m[,5],na.rm=T),
     mean(mod.spect.corr1m[,6],na.rm=T),
     mean(mod.spect.corr1m[,7],na.rm=T))
s<-c(sd(mod.spect.corr1m[,1],na.rm=T),
     sd(mod.spect.corr1m[,2],na.rm=T),
     sd(mod.spect.corr1m[,3],na.rm=T),
     sd(mod.spect.corr1m[,4],na.rm=T),
     sd(mod.spect.corr1m[,5],na.rm=T),
     sd(mod.spect.corr1m[,6],na.rm=T),
     sd(mod.spect.corr1m[,7],na.rm=T))
modSPEC<-as.data.frame(cbind(d,m,s))
rm(m,s)

mod.spectro.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
                   rep(64,300),rep(128,300),rep(256,300))
mod.spectro.all<-cbind(mod.spectro.all,
                       c(mod.spectro.corr1m[,1],mod.spectro.corr1m[,2],
                         mod.spectro.corr1m[,3],mod.spectro.corr1m[,4],
                         mod.spectro.corr1m[,5],mod.spectro.corr1m[,6],
                         mod.spectro.corr1m[,7]))
colnames(mod.spectro.all)<-c("dist","corr")
mod.spectro.all<-as.data.frame(mod.spectro.all)
mod.spectro.all<-na.omit(mod.spectro.all)

m<-c(mean(mod.spectro.corr1m[,1],na.rm=T),
     mean(mod.spectro.corr1m[,2],na.rm=T),
     mean(mod.spectro.corr1m[,3],na.rm=T),
     mean(mod.spectro.corr1m[,4],na.rm=T),
     mean(mod.spectro.corr1m[,5],na.rm=T),
     mean(mod.spectro.corr1m[,6],na.rm=T),
     mean(mod.spectro.corr1m[,7],na.rm=T))
s<-c(sd(mod.spectro.corr1m[,1],na.rm=T),
     sd(mod.spectro.corr1m[,2],na.rm=T),
     sd(mod.spectro.corr1m[,3],na.rm=T),
     sd(mod.spectro.corr1m[,4],na.rm=T),
     sd(mod.spectro.corr1m[,5],na.rm=T),
     sd(mod.spectro.corr1m[,6],na.rm=T),
     sd(mod.spectro.corr1m[,7],na.rm=T))
modTRO<-as.data.frame(cbind(d,m,s))
rm(m,s)

#### mod figure ####
tiff(filename="./signal_degradation_modulated.tiff",width=3000,height=3000,units="px",res=300,compression="none",bg="transparent")

ggplot(data = modSPEC, aes(x=d, y=m)) + 
  geom_point(data = modSPEC, aes(y = m),size=8,color="#7CCD7C") +
  geom_errorbar(data = modSPEC, aes(ymin=m-s, ymax=m+s), linewidth=0.8,color="#7CCD7C") +
  geom_smooth(data = mod.spect.all, aes(x=dist, y=corr,color="Amplitude spectrum"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  geom_point(data = modENV, aes(x=d, y=m), size=8, color="#8B4726") +
  geom_errorbar(data = modENV, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#8B4726") +
  geom_smooth(data = mod.env.all, aes(x=dist, y=corr, color="Envelope"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  geom_point(data = modTRO, aes(x=d, y=m), size=8, color="#27408B") +
  geom_errorbar(data = modTRO, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#27408B") +
  geom_smooth(data = mod.spectro.all, aes(x=dist, y=corr, color="Spectrogram"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
   labs(x="Distance (m)", y="Correlation at 1m",cex.lab=2) + 
  scale_x_continuous(breaks = d)+
  ylim(0,1)+
  ggtitle("Modulated calls")+
  scale_color_manual(values=c('#7CCD7C','#8B4726','#27408B')) +
  theme(legend.position = c(0.8,0.9), legend.text=element_text(size=24), legend.title=element_blank(),
        legend.key=element_rect(fill='black'))+
  theme(axis.text=element_text(size=24),axis.title=element_text(size=28,face="bold"),
        plot.title=element_text(size=28,face="bold",hjust=0.5))
dev.off()

##### tv calculate ####
tv.env.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
              rep(64,300),rep(128,300),rep(256,300))
tv.env.all<-cbind(tv.env.all,
                  c(tv.env.corr1m[,1],tv.env.corr1m[,2],
                    tv.env.corr1m[,3],tv.env.corr1m[,4],
                    tv.env.corr1m[,5],tv.env.corr1m[,6],
                    tv.env.corr1m[,7]))
colnames(tv.env.all)<-c("dist","corr")
tv.env.all<-as.data.frame(tv.env.all)
tv.env.all<-na.omit(tv.env.all)
d<-c(4,8,16,32,64,128,256)

m<-c(mean(tv.env.corr1m[,1],na.rm=T),
     mean(tv.env.corr1m[,2],na.rm=T),
     mean(tv.env.corr1m[,3],na.rm=T),
     mean(tv.env.corr1m[,4],na.rm=T),
     mean(tv.env.corr1m[,5],na.rm=T),
     mean(tv.env.corr1m[,6],na.rm=T),
     mean(tv.env.corr1m[,7],na.rm=T))
s<-c(sd(tv.env.corr1m[,1],na.rm=T),
     sd(tv.env.corr1m[,2],na.rm=T),
     sd(tv.env.corr1m[,3],na.rm=T),
     sd(tv.env.corr1m[,4],na.rm=T),
     sd(tv.env.corr1m[,5],na.rm=T),
     sd(tv.env.corr1m[,6],na.rm=T),
     sd(tv.env.corr1m[,7],na.rm=T))
d<-c(4,8,16,32,64,128,256)
tvENV<-as.data.frame(cbind(d,m,s))
rm(m, s)

tv.spect.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
                rep(64,300),rep(128,300),rep(256,300))
tv.spect.all<-cbind(tv.spect.all,
                    c(tv.spect.corr1m[,1],tv.spect.corr1m[,2],
                      tv.spect.corr1m[,3],tv.spect.corr1m[,4],
                      tv.spect.corr1m[,5],tv.spect.corr1m[,6],
                      tv.spect.corr1m[,7]))
colnames(tv.spect.all)<-c("dist","corr")
tv.spect.all<-as.data.frame(tv.spect.all)
tv.spect.all<-na.omit(tv.spect.all)

m<-c(mean(tv.spect.corr1m[,1],na.rm=T),
     mean(tv.spect.corr1m[,2],na.rm=T),
     mean(tv.spect.corr1m[,3],na.rm=T),
     mean(tv.spect.corr1m[,4],na.rm=T),
     mean(tv.spect.corr1m[,5],na.rm=T),
     mean(tv.spect.corr1m[,6],na.rm=T),
     mean(tv.spect.corr1m[,7],na.rm=T))
s<-c(sd(tv.spect.corr1m[,1],na.rm=T),
     sd(tv.spect.corr1m[,2],na.rm=T),
     sd(tv.spect.corr1m[,3],na.rm=T),
     sd(tv.spect.corr1m[,4],na.rm=T),
     sd(tv.spect.corr1m[,5],na.rm=T),
     sd(tv.spect.corr1m[,6],na.rm=T),
     sd(tv.spect.corr1m[,7],na.rm=T))
tvSPEC<-as.data.frame(cbind(d,m,s))
rm(m,s)

tv.spectro.all<-c(rep(4,300),rep(8,300),rep(16,300),rep(32,300),
                  rep(64,300),rep(128,300),rep(256,300))
tv.spectro.all<-cbind(tv.spectro.all,
                      c(tv.spectro.corr1m[,1],tv.spectro.corr1m[,2],
                        tv.spectro.corr1m[,3],tv.spectro.corr1m[,4],
                        tv.spectro.corr1m[,5],tv.spectro.corr1m[,6],
                        tv.spectro.corr1m[,7]))
colnames(tv.spectro.all)<-c("dist","corr")
tv.spectro.all<-as.data.frame(tv.spectro.all)
tv.spectro.all<-na.omit(tv.spectro.all)

m<-c(mean(tv.spectro.corr1m[,1],na.rm=T),
     mean(tv.spectro.corr1m[,2],na.rm=T),
     mean(tv.spectro.corr1m[,3],na.rm=T),
     mean(tv.spectro.corr1m[,4],na.rm=T),
     mean(tv.spectro.corr1m[,5],na.rm=T),
     mean(tv.spectro.corr1m[,6],na.rm=T),
     mean(tv.spectro.corr1m[,7],na.rm=T))
s<-c(sd(tv.spectro.corr1m[,1],na.rm=T),
     sd(tv.spectro.corr1m[,2],na.rm=T),
     sd(tv.spectro.corr1m[,3],na.rm=T),
     sd(tv.spectro.corr1m[,4],na.rm=T),
     sd(tv.spectro.corr1m[,5],na.rm=T),
     sd(tv.spectro.corr1m[,6],na.rm=T),
     sd(tv.spectro.corr1m[,7],na.rm=T))
tvTRO<-as.data.frame(cbind(d,m,s))
rm(m,s)

#### tv figure ####
tiff(filename="./signal_degradation_twovoice.tiff",width=3000,height=3000,units="px",res=300,compression="none",bg="transparent")

ggplot(data = tvSPEC, aes(x=d, y=m)) + 
  geom_point(data = tvSPEC, aes(y = m),size=8,color="#7CCD7C") +
  geom_errorbar(data = tvSPEC, aes(ymin=m-s, ymax=m+s), linewidth=0.8,color="#7CCD7C") +
  geom_smooth(data = tv.spect.all, aes(x=dist, y=corr,color="Amplitude spectrum"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  geom_point(data = tvENV, aes(x=d, y=m), size=8, color="#8B4726") +
  geom_errorbar(data = tvENV, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#8B4726") +
  geom_smooth(data = tv.env.all, aes(x=dist, y=corr, color="Envelope"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  geom_point(data = tvTRO, aes(x=d, y=m), size=8, color="#27408B") +
  geom_errorbar(data = tvTRO, aes(ymin=m-s, ymax=m+s), linewidth=0.8, color="#27408B") +
  geom_smooth(data = tv.spectro.all, aes(x=dist, y=corr, color="Spectrogram"),
              method='gam',formula = y ~ s(x, k=3),level=0.9,linewidth=2) +
  labs(x="Distance (m)", y="Correlation at 1m",cex.lab=2) + 
  scale_x_continuous(breaks = d)+
  ylim(0,1)+
  ggtitle("Two-voices calls")+
  scale_color_manual(values=c('#7CCD7C','#8B4726','#27408B')) +
  theme(legend.position = c(0.8,0.9), legend.text=element_text(size=24), legend.title=element_blank(),
        legend.key=element_rect(fill='black'))+
  theme(axis.text=element_text(size=24),axis.title=element_text(size=28,face="bold"),
        plot.title=element_text(size=28,face="bold",hjust=0.5))
dev.off()

