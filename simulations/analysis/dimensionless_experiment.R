require(ggplot2)
library(readr)
require(dplyr)
require(gridExtra)
library(latex2exp)

library(grid)
library(RColorBrewer)

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


setwd("/media/alexander/AKC Passport 2TB")

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
epsilon_stat = dplyr::full_join(eps_av, eps_sd, by = "zb") %>% rename(av="mean(epsilon)", sd="sd(epsilon)")

V_K_av = epsilon %>% group_by(zb) %>% summarise(mean(V_K))
V_K_sd = epsilon %>% group_by(zb) %>% summarise(sd(V_K))
V_K_stat = dplyr::full_join(V_K_av, V_K_sd, by = "zb") %>% rename(av="mean(V_K)", sd="sd(V_K)")

omega_K_av = epsilon %>% group_by(zb) %>% summarise(mean(omega_K))
omega_K_sd = epsilon %>% group_by(zb) %>% summarise(sd(omega_K))
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

phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v10_av=mean(1e-5 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v10_sd=sd(1e-5 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v100_av=mean(1e-4 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v100_sd=sd(1e-4 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v500_av=mean(5e-4 / V_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(v500_sd=sd(5e-4 / V_K)), by="zb")

phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B1_av=mean(1 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B1_sd=sd(1 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B3_av=mean(3 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B3_sd=sd(3 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B5_av=mean(5 * omega_K)), by="zb")
phipsi = left_join(phipsi, epsilon %>% group_by(zb) %>% summarise(B5_sd=sd(5 * omega_K)), by="zb")

B1_v500 = ggplot(phipsi) +
  geom_ribbon(aes(x = B1_av, ymin=v500_av-v500_sd, ymax=v500_av+v500_sd, fill=region, color=region), alpha=1., linetype=1) +
  geom_ribbon(aes(y = v500_av, xmin=B1_av-B1_sd, xmax=B1_av+B1_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  #geom_rect(mapping=aes(xmin=B1_av-B1_sd, xmax = B1_av+B1_sd, ymin = v500_av-v500_sd, ymax = v500_av+v500_sd, fill=region), color=NA, alpha=0.5) + 
  geom_line(aes(x=B1_av, y=v500_av), colour="black", size=0.2) +
  #geom_vline(xintercept=0.1, colour="black", linetype=2, alpha=0.7, size=1) +
  labs(y=expression(Phi) , x=expression(Psi), tag = "a") +
  #coord_flip() +
  #scale_x_continuous(trans='log10', limits=c(-1, 10)) + 
  #scale_y_continuous(trans='log10', limits=c(-1, 10)) +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 

B5_v10 = B1_v500 +
  geom_ribbon(aes(x = B5_av, ymin=v10_av-v10_sd, ymax=v10_av+v10_sd, fill=region, color=region), alpha=1., linetype=1) +
  geom_ribbon(aes(y = v10_av, xmin=B5_av-B5_sd, xmax=B5_av+B5_sd, fill=region), alpha=1., colour=NA, linetype=1) +
  #geom_rect(mapping=aes(xmin=B1_av-B1_sd, xmax = B1_av+B1_sd, ymin = v500_av-v500_sd, ymax = v500_av+v500_sd, fill=region), color=NA, alpha=0.5) + 
  geom_line(aes(x=B5_av, y=v10_av), colour="black", size=0.2) +
  #geom_vline(xintercept=0.1, colour="black", linetype=2, alpha=0.7, size=1) +
  labs(y=expression(Phi) , x=expression(Psi), tag = "a") +
  #coord_flip() +
  scale_x_log10() + 
  scale_y_log10() +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 



