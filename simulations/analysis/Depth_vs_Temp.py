import os, sys
import numpy as np
import holoviews as hv
from holoviews.operation.datashader import datashade
hv.extension('matplotlib')

# ensure python3.6 or higher (for compatibility with holoviews)
if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception("Python 3.6 or a more recent version is required.")

# load data
os.chdir("/media/alexander/AKC Passport 2TB/0-30")
tempsWithDeps = np.load("twd.npy")

# generate scatter plot with holoviews
points = hv.Points(tempsWithDeps)
scatter = datashade(points)
scatter.opts(title=r"Fluid DNS temperature vs depth",
             xlabel=r"Temperature (relative to reference temp $\theta_{z_0}$) [$^{\circ}C$]",
             ylabel=r"Depth [m]",
             fontscale=2,
             aspect=1,
             fig_inches=11)
hv.save(scatter, "scatter.png")
