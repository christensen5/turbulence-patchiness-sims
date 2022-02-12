require(ggplot2)
library(readr)
require(dplyr)
require(gridExtra)
library(latex2exp)


setwd("/media/alexander/AKC Passport 2TB")

epsilon <- read_csv("epsilon.csv")
time_offset = 30.0  # epsilon csv file includes DNS spin-up timesteps, which the Parcels simulation does not.
sprintf("Time offset set to %.1f seconds. Ensure this is correct.", time_offset)
epsilon = epsilon %>% filter(time >= time_offset)
epsilon$time = epsilon$time - time_offset  # align timestamps with the Parcels simulation.
epsilon$epsilon = -epsilon$epsilon  # make epsilon column positive
epsilon_timestamps = unique(epsilon$time)  # extract timestamps at which we have epsilon data
epsilon60 = epsilon %>% filter(time == 60)

eps_av = epsilon %>% group_by(zb) %>% summarise(mean(epsilon))
eps_sd = epsilon %>% group_by(zb) %>% summarise(sd(epsilon))
epsilon_stat = dplyr::full_join(eps_av, eps_sd, by = "zb")
epsilon_stat = dplyr::rename(epsilon_stat, av="mean(epsilon)", sd="sd(epsilon)")


t01 <- read_csv("t01_plot.csv")

p1 = ggplot(t01, aes(x=zb,y=av, ymin=0, ymax=0.9, xmin=0., xmax=0.3)) +
  geom_ribbon(aes(ymin=av - sqrt(var), ymax=av + sqrt(var)), alpha=0.5, colour="dodgerblue3", linetype=2, fill="dodgerblue3") +
  geom_line(colour="darkred", size=1) +
  geom_vline(xintercept=0.1, colour="black", linetype=3, alpha=0.7) +
  geom_vline(xintercept=0.17, colour="black", linetype=3, alpha=0.7) +
  geom_vline(xintercept=0.24, colour="black", linetype=3, alpha=0.7) +
  geom_vline(xintercept=0.3, colour="black", linetype=3, alpha=0.7) +
  labs(y=expression(paste(Delta, rho, "/", rho, ""[0])), x="z [m]", tag = "a") +
  coord_flip() +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) + 
  theme(plot.margin=unit(c(0,0,0.5,0), "cm"))
p2 = ggplot(epsilon_stat, aes(x=zb, y=av, ymin=0, ymax=3.0e-4, xmin=0., xmax=0.3)) +
  geom_ribbon(aes(ymin=av - sd, ymax=av + sd), alpha=0.5, colour="dodgerblue3", linetype=2, fill="dodgerblue3") +
  geom_line(colour="darkred", size=1) +
  geom_vline(xintercept=0.1, colour="black", linetype=3, alpha=0.7) +
  geom_vline(xintercept=0.17, colour="black", linetype=3, alpha=0.7) +
  geom_vline(xintercept=0.24, colour="black", linetype=3, alpha=0.7) +
  geom_vline(xintercept=0.3, colour="black", linetype=3, alpha=0.7) +
  scale_x_continuous(position = "top") +
  labs(y=expression(paste(epsilon, " [", m^{2}, s^{-3}, "]")), x="z [m]", tag = "b") +
  coord_flip() +
  theme(panel.background = element_blank(), axis.line = element_line(colour = "black"), text=element_text(size=21)) + 
  theme(plot.margin=unit(c(0,0,0.31,0), "cm"))

  grid.arrange(p1, p2, ncol=2)
