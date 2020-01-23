import psutil
from time import sleep
import numpy as np
import sys

pid = int(sys.argv[1])
savepath = sys.argv[2]

memuseage = []
max = 0
while 0 < 1:
    if psutil.pid_exists(pid):
        m = psutil.Process(pid).memory_full_info().uss / (1024 * 1024)
        if m > max:
            # print("%s MB" % m)
            max = m
        memuseage.append(m)
        sleep(1)
    else:
        np.save(savepath, np.array(memuseage))
        exit()