import os, sys
import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade
hv.extension('matplotlib')

# load data
os.chdir("/media/alexander/AKC Passport 2TB/0-30")
tempsWithDeps = np.load("twd.npy")
# sample = tempsWithDeps[np.random.choice(tempsWithDeps.shape[0], 10000000), :]
# generate scatter plot with holoviews
points = hv.Points(tempsWithDeps)
points.opts(alpha=0.1, s=2,
            xlabel=r"Temperature (relative to reference temp $\theta_{z_0}$) [$^{\circ}C$]",
            ylabel=r"Depth [m]",
            fontscale=2.5,
            aspect=1,
            fig_inches=11,
            edgecolors='none')
hv.save(points, "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/DepthvsTemp/DepthvsTemp.png")


