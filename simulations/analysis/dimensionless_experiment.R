require(ggplot2)
library(readr)
require(dplyr)
require(gridExtra)
library(latex2exp)
library(plotrix)

library(grid)
library(RColorBrewer)
library(ggsci)

make_gradient <- function(data, cols = blues9) {
  n = length(data)
  cols <- colorRampPalette(cols)(n + 1)
  mat = matrix(data, nrow=length(data), ncol=length(data), byrow=FALSE)
  mat <- mat - min(mat)
  mat <- mat / max(mat)
  mat <- 1 + mat * n
  mat <- matrix(data = cols[round(mat)], ncol = n)
  grid::rasterGrob(
    image = mat,
    width = unit(1, "npc"),
    height = unit(1, "npc"), 
    interpolate = TRUE
  )
}

assign_region = function(vect){
  newvect = as.character(vect)
  for (i in 1:length(vect)){
    if (vect[i] >= 0.24){
      newvect[i] = "shallow"
    } else if (vect[i] < 0.24 & vect[i] >= 0.17){
      newvect[i] = "mid"
    } else if (vect[i] < 0.17 & vect[i] >= 0.10){
      newvect[i] = "deep"
    } else {
      newvect[i] = "none"
    }
  }
  return(newvect)
}

region_colours = function(vect){
  colourvect = as.character(vect)
  for (i in 1:length(vect)){
    if (vect[i] == "shallow"){
      colourvect[i] = "#f8766d"
    } else if (vect[i] == "mid"){
      colourvect[i] = "#00ba38"
    } else if (vect[i] == "deep"){
      colourvect[i] = "#5d9df7"
    } else {
      colourvect[i] = "none"
    }
  }
  return(colourvect)
}

setwd("/media/alexander/AKC Passport 2TB")

eps_min_ocean = 1e-8
eps_max_ocean = 1e-4
nu_min_ocean = 8.01e-07
nu_max_ocean = 1.31e-06

PHI_min_ocean_v10 = 1e-5/(eps_max_ocean * nu_max_ocean)^0.25
PHI_max_ocean_v10 = 1e-5/(eps_min_ocean * nu_min_ocean)^0.25
PHI_min_ocean_v100 = 1e-4/(eps_max_ocean * nu_max_ocean)^0.25
PHI_max_ocean_v100 = 1e-4/(eps_min_ocean * nu_min_ocean)^0.25
PHI_min_ocean_v500 = 5e-4/(eps_max_ocean * nu_max_ocean)^0.25
PHI_max_ocean_v500 = 5e-4/(eps_min_ocean * nu_min_ocean)^0.25

PSI_min_ocean_B1 = 1 * ((eps_min_ocean / nu_max_ocean)^0.5)
PSI_max_ocean_B1 = 1 * ((eps_max_ocean / nu_min_ocean)^0.5)
PSI_min_ocean_B3 = 3 * ((eps_min_ocean / nu_max_ocean)^0.5)
PSI_max_ocean_B3 = 3 * ((eps_max_ocean / nu_min_ocean)^0.5)
PSI_min_ocean_B5 = 5 * ((eps_min_ocean / nu_max_ocean)^0.5)
PSI_max_ocean_B5 = 5 * ((eps_max_ocean / nu_min_ocean)^0.5)

nu = 5e-6

time_offset = 30.0  # epsilon and t01 csv files include DNS spin-up timesteps, which the Parcels simulation does not.
sprintf("Time offset set to %.1f seconds. Ensure this is correct.", time_offset)

epsilon <- read_csv("epsilon.csv")
epsilon = epsilon %>% filter(time >= time_offset) %>% filter(zb >= 0.1)
epsilon$time = epsilon$time - time_offset  # align timestamps with the Parcels simulation.
epsilon$epsilon = -epsilon$epsilon  # make epsilon column positive
epsilon_timestamps = unique(epsilon$time)  # extract timestamps at which we have epsilon data
epsilon$V_K = (nu * epsilon$epsilon)^0.25
epsilon$omega_K = (epsilon$epsilon / nu)^0.5

eps_av = epsilon %>% group_by(zb) %>% summarise(mean(epsilon))
eps_sd = epsilon %>% group_by(zb) %>% summarise(sd(epsilon))
# eps_sd = epsilon %>% group_by(zb) %>% summarise(std.error(epsilon))
epsilon_stat = dplyr::full_join(eps_av, eps_sd, by = "zb") %>% rename(av="mean(epsilon)", sd="sd(epsilon)")

V_K_av = epsilon %>% group_by(zb) %>% summarise(mean(V_K))
V_K_sd = epsilon %>% group_by(zb) %>% summarise(sd(V_K))
# V_K_sd = epsilon %>% group_by(zb) %>% summarise(std.error(V_K))
V_K_stat = dplyr::full_join(V_K_av, V_K_sd, by = "zb") %>% rename(av="mean(V_K)", sd="sd(V_K)")

omega_K_av = epsilon %>% group_by(zb) %>% summarise(mean(omega_K))
omega_K_sd = epsilon %>% group_by(zb) %>% summarise(sd(omega_K))
# omega_K_sd = epsilon %>% group_by(zb) %>% summarise(std.error(omega_K))
omega_K_stat = dplyr::full_join(omega_K_av, omega_K_sd, by = "zb") %>% rename(av="mean(omega_K)", sd="sd(omega_K)")

phi = data.frame(z = epsilon_stat$zb,
                    region = assign_region(epsilon_stat$zb),
                    V_K_av = V_K_stat$av,
                    V_K_sd = V_K_stat$sd)

psi = data.frame(z = epsilon_stat$zb,
                 region = assign_region(epsilon_stat$zb),
                 omega_K_av = omega_K_stat$av,
                 omega_K_sd = omega_K_stat$sd)

# all B columns in phipsi are the psi-values at that B value. Likewise v columns are phi-values at that vswim value.
phipsi = data.frame(zb = epsilon_stat$zb,
                    region = assign_region(epsilon_stat$zb))
phipsi$region_colours = region_colours(phipsi$region)

phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v10_av=mean(1e-5 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v10_sd=sd(1e-5 / V_K)), by="zb")
# phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v10_sd=std.error(1e-5 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v100_av=mean(1e-4 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v100_sd=sd(1e-4 / V_K)), by="zb")
# phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v100_sd=std.error(1e-4 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v500_av=mean(5e-4 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v500_sd=sd(5e-4 / V_K)), by="zb")
# phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v500_sd=std.error(5e-4 / V_K)), by="zb")

phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B1_av=mean(1 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B1_sd=sd(1 * omega_K)), by="zb")
# phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B1_sd=std.error(1 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B3_av=mean(3 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B3_sd=sd(3 * omega_K)), by="zb")
# phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B3_sd=std.error(3 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B5_av=mean(5 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B5_sd=sd(5 * omega_K)), by="zb")
# phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B5_sd=std.error(5 * omega_K)), by="zb")


B1_v500 = ggplot(phipsi) +
  #geom_ribbon(aes(x = B1_av, ymin=v500_av-v500_sd, ymax=v500_av+v500_sd, fill=region, color=region), alpha=1., linetype=1) +
  #geom_ribbon(aes(y = v500_av, xmin=B1_av-B1_sd, xmax=B1_av+B1_sd, fill=region, color=region), alpha=1., linetype=1) +
  #geom_rect(mapping=aes(xmin=B1_av[1]-B1_sd[1], xmax = B1_av[1], ymin = v500_av[1], ymax = v500_av[1]+v500_sd[1], fill=region, color=region), alpha=1.) + 
  geom_line(aes(x=B1_av, y=v500_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B1_av+phipsi$B1_sd), y = min(phipsi$v500_av), label="(B1, v500)", hjust=0) +
  labs(y=expression(Phi) , x=expression(Psi), tag = "a") +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 
B1_v100 = B1_v500 +
  # geom_ribbon(aes(x = B1_av, ymin=v100_av-v100_sd, ymax=v100_av+v100_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v100_av, xmin=B1_av-B1_sd, xmax=B1_av+B1_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  geom_line(aes(x=B1_av, y=v100_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B1_av+phipsi$B1_sd), y = min(phipsi$v100_av), label="(B1, v100)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21))
B1_v10 = B1_v100 +
  # geom_ribbon(aes(x = B1_av, ymin=v10_av-v10_sd, ymax=v10_av+v10_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v10_av, xmin=B1_av-B1_sd, xmax=B1_av+B1_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  geom_line(aes(x=B1_av, y=v10_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B1_av+phipsi$B1_sd), y = min(phipsi$v10_av), label="(B1, v10)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 

B3_v500 = B1_v10 +
  # geom_ribbon(aes(x = B3_av, ymin=v500_av-v500_sd, ymax=v500_av+v500_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v500_av, xmin=B3_av-B3_sd, xmax=B3_av+B3_sd, fill=region, color=region), alpha=1., linetype=1) +
  geom_line(aes(x=B3_av, y=v500_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B3_av+phipsi$B3_sd), y = min(phipsi$v500_av), label="(B3, v500)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 
B3_v100 = B3_v500 +
  # geom_ribbon(aes(x = B3_av, ymin=v100_av-v100_sd, ymax=v100_av+v100_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v100_av, xmin=B3_av-B3_sd, xmax=B3_av+B3_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  geom_line(aes(x=B3_av, y=v100_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B3_av+phipsi$B3_sd), y = min(phipsi$v100_av), label="(B3, v100)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21))
B3_v10 = B3_v100 +
  # geom_ribbon(aes(x = B3_av, ymin=v10_av-v10_sd, ymax=v10_av+v10_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v10_av, xmin=B3_av-B3_sd, xmax=B3_av+B3_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  geom_line(aes(x=B3_av, y=v10_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B3_av+phipsi$B3_sd), y = min(phipsi$v10_av), label="(B3, v10)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 

B5_v500 = B3_v10 +
  # geom_ribbon(aes(x = B5_av, ymin=v500_av-v500_sd, ymax=v500_av+v500_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v500_av, xmin=B5_av-B5_sd, xmax=B5_av+B5_sd, fill=region, color=region), alpha=1., linetype=1) +
  geom_line(aes(x=B5_av, y=v500_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B5_av+phipsi$B5_sd), y = min(phipsi$v500_av), label="(B5, v500)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 
B5_v100 = B5_v500 +
  # geom_ribbon(aes(x = B5_av, ymin=v100_av-v100_sd, ymax=v100_av+v100_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v100_av, xmin=B5_av-B5_sd, xmax=B5_av+B5_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  geom_line(aes(x=B5_av, y=v100_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B5_av+phipsi$B5_sd), y = min(phipsi$v100_av), label="(B5, v100)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21))
B5_v10 = B5_v100 +
  # geom_ribbon(aes(x = B5_av, ymin=v10_av-v10_sd, ymax=v10_av+v10_sd, fill=region, color=region), alpha=1., linetype=1) +
  # geom_ribbon(aes(y = v10_av, xmin=B5_av-B5_sd, xmax=B5_av+B5_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  geom_line(aes(x=B5_av, y=v10_av, color=region), size=1.5) +
  annotate("text", x=max(phipsi$B5_av+phipsi$B5_sd), y = min(phipsi$v10_av), label="(B5, v10)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21))

BVplot = B5_v10 + geom_hline(aes(yintercept = PHI_max_ocean_v500), color="grey60", size=0.8, linetype="dashed")
print(BVplot + scale_x_log10(limits=c(0.1, 50)) + scale_y_log10(limits=c(0.001, 2)))

w = 15
B1 = ggplot(phipsi) +
  geom_segment(x=log10(PSI_min_ocean_B1), y=1, xend=log10(PSI_max_ocean_B1), yend=1, size=w, color="grey80", alpha=0.1) +
  geom_line(aes(x=B1_av, y=as.factor(1), color=region), size=10) +
  annotate("text", x=0.07, y = 1, label="B=1s", hjust="right", vjust="center", size=10) +
  #annotate("text", x=max(phipsi$B1_av+phipsi$B1_sd), y = 1, label="(B1, v500)", hjust=0) +
  labs(x=expression(Psi))#, tag = "a") +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=30)) 
B3 = B1 +
  geom_segment(x=log10(PSI_min_ocean_B3), y=2, xend =log10(PSI_max_ocean_B3), yend=2, size=w, color="grey80", alpha=0.1) +
  geom_line(aes(x=B3_av, y=as.factor(2), color=region), size=10) +
  annotate("text", x=0.2, y = 2, label="B=3s", hjust="right", vjust="center", size=10) +
  #annotate("text", x=max(phipsi$B3_av+phipsi$B3_sd), y = 2, label="(B3, v10)", hjust=0) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=30))
B5 = B3 +
  geom_segment(x=log10(PSI_min_ocean_B5), y=3, xend =log10(PSI_max_ocean_B5), yend=3, size=w, color="grey80", alpha=0.1) +
  geom_line(aes(x=B5_av, y=as.factor(3), color=region), size=10) +
  annotate("text", x=0.35, y = 3, label="B=5s", hjust="right", vjust="center", size=10) +
  theme(panel.background = element_blank(), axis.line.x = element_line(colour = "black"), text=element_text(size=30), legend.key=element_blank(), legend.title=element_blank(),
        axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), axis.line.y=element_blank())
Bplot = B5 +
  # geom_vline(aes(xintercept = 1.0), color="grey60", size=1, linetype="dashed") +
  geom_segment(x=0, y=0.4, xend=0, yend=3.75, color="#00924e", size=1, linetype="dashed") +
  guides(colour = guide_legend(override.aes = list(alpha = 0.3, fill=NA))) +
  geom_segment(x = -0.05, y = 3.75, xend = -1.5, yend = 3.75, lineend = "butt", linejoin = "mitre", size = 1, arrow = arrow(length = unit(0.2, "inches")), colour = "#00924e") +
  geom_segment(x = 0.05, y = 3.75, xend = 1.8, yend = 3.75, lineend = "butt", linejoin = "mitre", size = 1, arrow = arrow(length = unit(0.2, "inches")), colour = "#00924e") +
  annotate("text", x=0.04, y=c(3.9, 3.6), label=c("less", "patchy"), hjust="left", size=10, color="#00924e") +
  annotate("text", x=50, y=c(3.9, 3.6), label=c("less", "patchy"), hjust="right", size=10, color="#00924e") +
  #annotate("text", x=1, y=c(3.75, 3.6), label=c("max", "patchiness"), size=10, color="#00924e") +
  coord_cartesian( clip = "off") + 
  geom_blank(aes(x=1, y=4))
# print(Bplot + scale_x_log10(limits=c(0.1, 50)) + scale_y_discrete(labels=c("1"="1", "2"="3", "3"="5")) + scale_color_manual(values=alpha(c("#009ad8", "#e79035", "#be313f"), 0.7)))
print(Bplot + scale_x_log10(limits=c(0.04, 50)) + scale_color_manual(values=alpha(c("#009ad8", "#e79035", "#be313f"), 0.7)))


V10 = ggplot(phipsi) +
  geom_segment(x=0.5, y=log10(PHI_min_ocean_v10), xend=0.5, yend=log10(PHI_max_ocean_v10), size=w, color="grey80", alpha=0.1) +
  geom_line(aes(x=0.5, y=v10_av, color=region), size=10) +
  annotate("text", x=0.5, y = 0.001, label=expression(paste(v["swim"], "=", 10, mu, "m/s")), hjust="center", vjust="top", size=10) +
  labs(y=expression(Phi))#, tag = "b") +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=30)) 
V100 = V10 +
  geom_segment(x=1.5, y=log10(PHI_min_ocean_v100), xend=1.5, yend=log10(PHI_max_ocean_v100), size=w, color="grey80", alpha=0.1) +
  geom_line(aes(x=1.5, y=v100_av, color=region), size=10) +
  annotate("text", x=1.5, y = 0.01, label=expression(paste(v["swim"], "=", 100, mu, "m/s")), hjust="center", vjust="top", size=10) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=30)) 
V500 = V100 +
  geom_segment(x=2.5, y=log10(PHI_min_ocean_v500), xend=2.5, yend=log10(PHI_max_ocean_v500), size=w, color="grey80", alpha=0.1) +
  geom_line(aes(x=2.5, y=v500_av, color=region), size=10) +
  annotate("text", x=2.5, y = 0.055, label=expression(paste(v["swim"], "=", 500, mu, "m/s")), hjust="center", vjust="top", size=10) +
  theme(panel.background = element_blank(), axis.line.y = element_line(colour = "black"), text=element_text(size=30), legend.position="none",
        axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.line.x=element_blank())
Vplot = V500 +
  # geom_segment(x=0, y=log10(PHI_max_ocean_v500), xend=2.1, yend=log10(PHI_max_ocean_v500), color="red", size=1, linetype="dashed") +
  # guides(colour = guide_legend(override.aes = list(alpha = 0.3, fill=NA))) +
  geom_segment(x = 3.2, y = -3.3, xend = 3.2, yend = 0.35, lineend = "butt", linejoin = "mitre", size = 1, arrow = arrow(length = unit(0.2, "inches"), ends="both"), colour = "#00924e") +
  annotate("text", x=3.2, y=2.7, label="more patchy", hjust="center", vjust="bottom", size=10, color="#00924e") +
  annotate("text", x=3.2, y=0.0004, label="less patchy", hjust="center", size=10, color="#00924e") +
  coord_cartesian( clip = "off") + 
  geom_blank(aes(x=0, y=0.0004)) + 
  geom_blank(aes(x=3.5, y=2.6))
# print(Vplot + scale_x_continuous(limits=c(0, 2.4), breaks=c(0, 1, 2), labels=c(10, 100, 500)) + scale_y_log10(limits=c(0.001, 2.5), breaks=c(0.001, 0.01, 0.1, 1), labels=c("0.001", "0.01", "0.1", "1"))  + scale_color_manual(values=alpha(c("#009ad8", "#e79035", "#be313f"), 0.7)))
print(Vplot + scale_y_log10(limits=c(0.0004, 2.7), breaks=c(0.001, 0.01, 0.1, 1), labels=c("0.001", "0.01", "0.1", "1"))  + scale_color_manual(values=alpha(c("#009ad8", "#e79035", "#be313f"), 0.7)))

