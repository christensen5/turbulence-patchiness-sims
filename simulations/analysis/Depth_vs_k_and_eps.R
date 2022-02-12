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


setwd("/media/alexander/AKC Passport 2TB")

time_offset = 30.0  # epsilon and t01 csv files include DNS spin-up timesteps, which the Parcels simulation does not.
sprintf("Time offset set to %.1f seconds. Ensure this is correct.", time_offset)

epsilon <- read_csv("epsilon.csv")
epsilon = epsilon %>% filter(time >= time_offset)
epsilon$time = epsilon$time - time_offset  # align timestamps with the Parcels simulation.
epsilon$epsilon = -epsilon$epsilon  # make epsilon column positive
epsilon_timestamps = unique(epsilon$time)  # extract timestamps at which we have epsilon data

eps_av = epsilon %>% group_by(zb) %>% summarise(mean(epsilon))
eps_sd = epsilon %>% group_by(zb) %>% summarise(sd(epsilon))
epsilon_stat = dplyr::full_join(eps_av, eps_sd, by = "zb")
epsilon_stat = dplyr::rename(epsilon_stat, av="mean(epsilon)", sd="sd(epsilon)")

tke <- read_csv("tke_plot.csv")
tke = tke %>% filter(time >= time_offset)
tke$time = tke$time - time_offset # align timesteps with the Parcels simulation
tke_timesteps = unique(tke$time) # extract timestamps at which we have tke data
tke_stat = tke %>% group_by(zb) %>% summarise(mean(tke))
tke_stat = dplyr::rename(tke_stat, val="mean(tke)")

t01 <- read_csv("t01_plot.csv")
colorgrad = make_gradient(t01$av)

p1 = ggplot(tke_stat, aes(x=zb,y=val, ymin=0, ymax=7.0e-4, xmin=0., xmax=0.3)) +
  annotation_custom(grob = colorgrad, xmin = 0.0, xmax = 0.3, ymin = -Inf, ymax = Inf) +
  #geom_ribbon(aes(ymin=av - sqrt(var), ymax=av + sqrt(var)), alpha=0.8, colour="goldenrod1", linetype=1, fill="goldenrod1") +
  geom_line(colour="red3", size=1) +
  geom_vline(xintercept=0.1, colour="black", linetype=2, alpha=0.7, size=1) +
  geom_vline(xintercept=0.17, colour="black", linetype=2, alpha=0.7, size=1) +
  geom_vline(xintercept=0.24, colour="black", linetype=2, alpha=0.7, size=1) +
  geom_vline(xintercept=0.3, colour="black", linetype=2, alpha=0.7, size=1) +
  labs(y=expression(paste(k, " [", m^{2}, s^{-2}, "]")) , x="z [m]", tag = "a") +
  coord_flip() +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 
  #theme(plot.margin=unit(c(0,0,0.31,0), "cm"))
p2 = ggplot(epsilon_stat, aes(x=zb, y=av, ymin=0, ymax=3.0e-4, xmin=0., xmax=0.3)) +
  annotation_custom(grob = colorgrad, xmin = 0.0, xmax = 0.3, ymin = -Inf, ymax = Inf) +
  geom_ribbon(aes(ymin=av - sd^2, ymax=av + sd^2), alpha=0.8, colour="goldenrod1", linetype=1, fill="goldenrod1") +
  geom_line(colour="red3", size=1) +
  geom_vline(xintercept=0.1, colour="black", linetype=2, alpha=0.7, size=1) +
  geom_vline(xintercept=0.17, colour="black", linetype=2, alpha=0.7, size=1) +
  geom_vline(xintercept=0.24, colour="black", linetype=2, alpha=0.7, size=1) +
  geom_vline(xintercept=0.3, colour="black", linetype=2, alpha=0.7, size=1) +
  scale_x_continuous(position = "top") +
  labs(y=expression(paste(epsilon, " [", m^{2}, s^{-3}, "]")), x="z [m]", tag = "b") +
  coord_flip() +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) 
  #theme(plot.margin=unit(c(0,0,0.31,0), "cm"))

grid.arrange(p1, p2, ncol=2)

#expression(paste(Delta, rho, "/", rho, ""[0]))
