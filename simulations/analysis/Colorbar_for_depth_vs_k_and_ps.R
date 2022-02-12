require(ggplot2)
library(readr)
require(dplyr)
require(gridExtra)
library(latex2exp)

library(grid)
library(RColorBrewer)
library(ggtext)

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

colorgrad_bar = make_gradient(1:100)
t01 <- read_csv("t01_plot.csv")
setwd("/media/alexander/AKC Passport 2TB")

p3 = ggplot(data=t01, aes(x="",y="", ymin=0., ymax=1, xmin=0., xmax=0.1)) +
  geom_richtext(x=0.007, y=0.2, label="1", size=8, label.color=NA) +
  geom_richtext(x=0.009, y=0.8, label="0.75", size=8, label.color=NA) +
  geom_text(aes(0.009, 0.5), label=expression(paste(Delta, rho, "/", rho, ""[0])), size=8, check_overlap=TRUE) +
  annotation_custom(grob=colorgrad_bar, xmin = 0.0025, xmax = 0.005, ymin = 0.2, ymax = 0.8) +
  theme(axis.line=element_blank(),axis.text.x=element_blank(),
        axis.text.y=element_blank(),axis.ticks=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),legend.position="none",
        panel.background=element_blank(),panel.border=element_blank(),panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),plot.background=element_blank())

print(p3)