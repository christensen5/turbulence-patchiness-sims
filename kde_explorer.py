import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, sigma, mu):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

data_basic = np.array((0, 1, 2, 5, 6, 10))
data_sat = np.array((490,499,459,575,575,513,382,525,510,542,368,564,509,530,485,521,495,526,474,500,441,750,495,476,456,440,547,479,501,476,457,444,444,467,482,449,464,501,670,740,590,700,590,450,452,468,472,447,520,506,570,474,532,472,585,466,549,736,654,585,574,621,542,616,547,554,514,592,531,550,507,441,551,450,548,589,549,485,480,545,451,448,487,480,540,470,529,445,460,457,560,495,480,430,644,489,506,660,444,551,583,457,440,470,486,413,470,408,440,596,442,544,528,559,505,450,477,557,446,553,370,533,496,513,403,496,543,533,471,404,439,459,393,470,650,512,445,446,474,449,529,538,450,570,499,375,515,445,571,442,492,456,428,536,515,450,537,490,446,410,526,560,560,540,502,642,590,480,557,468,524,445,479))

## GENERATE BASIC DATA PLOTS
def generate_plots(data):
    n = 10000
    width = data.max() - data.min()
    l, r = data.min() - width, data.min() + width + width
    dx = (r - l) / (n - 1)
    x = np.linspace(l, r, n)
    # hist
    fig = plt.figure(figsize=(12, 9))
    ax_hist = fig.add_subplot(1, 3, 1)
    ax_hist.hist(data, 5, facecolor='blue')
    ax_hist.set_title("Histogram", fontsize=20)
    ax_hist.set_xlim(l, r)
    # rect
    y = np.zeros(n)
    kwidth = 3#20
    kpts = int(kwidth/dx)
    kernel = np.ones(kpts)
    for d in data:
        centres = int((d - l)/dx)
        bottom = centres - int(kpts / 2)
        top = centres + int(kpts / 2)
        if top - bottom < kpts: top = top + 1
        if top - bottom > kpts: top = top - 1
        y[bottom:top] += kernel
    ax_rect = fig.add_subplot(1, 3, 2)
    ax_rect.plot(x, y, lw=2)
    ax_rect.set_xlim(l, r)
    ax_rect.set_title("Step Kernel", fontsize=20)
    ax_rect.set_ylim(min(0,y.min()),1.1*y.max())


    # gauss
    y = np.zeros(n)
    # kwidth = 5
    kpts = int(kwidth / dx)
    kernel = gaussian(np.linspace(-3, 3, kpts), 1, 0)
    for d in data:
        centres = int((d - l)/dx)
        bottom = centres - int(kpts / 2)
        top = centres + int(kpts / 2)
        if top - bottom < kpts: top = top + 1
        if top - bottom > kpts: top = top - 1
        y[bottom:top] += kernel
    ax_gauss = fig.add_subplot(1, 3, 3)
    ax_gauss.plot(x, y, lw=2)
    ax_gauss.set_xlim(l, r)
    ax_gauss.set_title("Gaussian Kernel", fontsize=20)
    ax_gauss.set_ylim(min(0, y.min()), 1.1 * y.max())

    ax_hist.set_xlim(x[np.where(y > 0)[0][0]], x[np.where(y > 0)[0][-1]])
    ax_rect.set_xlim(x[np.where(y > 0)[0][0]], x[np.where(y > 0)[0][-1]])
    ax_gauss.set_xlim(x[np.where(y > 0)[0][0]], x[np.where(y > 0)[0][-1]])

    fig.savefig("/home/alexander/Documents/QMEE/LSR/report/fig/hist_vs_kde_basic")
    # plt.show()

if __name__ == "__main__":
    generate_plots(data_basic)




