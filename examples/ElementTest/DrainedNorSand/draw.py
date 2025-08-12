#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rcParams

base_path = "/home/yj/works/GeoTaichi_Yihao_v1/examples/ElementTest/DrainedNorSand"


params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 4.5,
             'lines.markersize': 10,
             'xtick.labelsize': 26,
             'ytick.labelsize': 26,
             'xtick.major.pad': 12,
             'ytick.major.pad': 12,
             "axes.labelpad":   8,
             'legend.fontsize': 26,
             'figure.figsize': [12, 9],
             'font.family': 'serif',
             'text.usetex': False,
             'font.serif': 'Arial',
             'savefig.dpi': 300
         }
rcParams.update(params)

         
color = [(0/255, 0/255, 0/255), 
         (255/255, 0/255, 0/255), 
         (94/255, 114/255, 255/255), 
         (0/255, 128/255, 0/255)]

pid = 0

start_num=0
end_num=38

def meanstress(stress):
    return (stress[2]+stress[1]+stress[0])/3.
    
    
def equistress(stress):
    return math.sqrt(3.*(((stress[0] - stress[1]) * (stress[0] - stress[1]) \
                          + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
                          + (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6. \
                          + stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]))


def deviatoric_stress(stress):
    sigma = meanstress(stress)

    deviatoric_stress = np.copy(stress)
    for i in range(3):
        deviatoric_stress[i] -= sigma
    return deviatoric_stress


def get_result(path):
    p = []
    q = []
    ratio = []
    time = []
    for printNum in range(start_num, end_num):
        data = np.load(f'{path}/particles/MPMParticle{printNum:06d}.npz', allow_pickle=True)
    
        p.append(-meanstress(data['stress'][pid]))
        q.append(equistress(data['stress'][pid]))
        ratio.append(equistress(data['stress'][pid])/-meanstress(data['stress'][pid]))
        time.append(data['t_current']*0.01)
    return p, q, ratio, time
    
            
# p1, q1, ratio1, time1 = get_result("./DrainedNorSand/33kpa")
# p2, q2, ratio2, time2 = get_result("./DrainedNorSand/98kpa")
p3, q3, ratio3, time3 = get_result(f"{base_path}/303kpa")

fig, ax=plt.subplots()
# ax.scatter(p1, q1, s=125, label='$p$=33kPa', color=(0/255, 128/255, 0/255))
# ax.scatter(p2, q2, s=125, label='$p$=98kPa', color=(255/255, 0/255, 0/255))
ax.scatter(p3, q3, label='$p$=303kPa', color=(94/255, 114/255, 255/255))
ax.set_xlabel("Mean stress, $p$ (Pa)")
ax.set_ylabel("Equivalent stress, $q$ (Pa)")
# ax.set_xlim([0,800000])
# ax.set_ylim([0,800000])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig (f"{base_path}/pqcurve.svg")
plt.close()

fig, ax=plt.subplots()
# ax.scatter(time1, q1, s=125, color=(0/255, 128/255, 0/255), label='$p$=33kPa')
# ax.scatter(time2, q2, s=125, color=(255/255, 0/255, 0/255), label='$p$=98kPa')
ax.scatter(time3, q3, color=(94/255, 114/255, 255/255), label='$p$=303kPa')
# ax.set_xlim([0,0.5])
# ax.set_ylim([0,500000])
ax.set_xlabel("Axial strain, $\\epsilon_a$ (%)")
ax.set_ylabel("Equivalent stress, $q$ (Pa)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig (f"{base_path}/qtcurve.svg")
plt.close()

# print((q1[-1]-q1[0])/(p1[-1]-p1[0]))
# print((q2[-1]-q2[0])/(p2[-1]-p2[0]))
# print((q3[-1]-q3[0])/(p3[-1]-p3[0]))


