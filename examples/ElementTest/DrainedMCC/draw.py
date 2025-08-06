#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from tablelegend import tablelegend

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
             'text.usetex': True,
             'font.serif': 'Arial',
             'savefig.dpi': 300
         }
rcParams.update(params)

         
color = [(0/255, 0/255, 0/255), 
         (255/255, 0/255, 0/255), 
         (94/255, 114/255, 255/255), 
         (0/255, 128/255, 0/255)]

pid = 0 
m_theta = 1.02
possion = 0.25
pc0 = 392000
lambda_ = 0.12
kappa = 0.023
e_ref = 1.7
p_ref = 1000

start_num=0
end_num=266

def get_strain(printNum):
    data = np.load('33kpa/particles/MPMParticle{0:06d}.npz'.format(printNum), allow_pickle=True)
    print(data['strain'][pid])


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
    pc = []
    ratio = []
    time = []
    for printNum in range(start_num, end_num):
        data = np.load(path+'/particles/MPMParticle{0:06d}.npz'.format(printNum), allow_pickle=True)
    
        p.append(-meanstress(data['stress'][pid]))
        q.append(equistress(data['stress'][pid]))
        pc.append(data['state_vars'].item()['pc'][pid])
        ratio.append(equistress(data['stress'][pid])/-meanstress(data['stress'][pid]))
        time.append(data['t_current']*0.01)
    return p, q, pc, ratio, time
    
    
def compute_dpdsigma():
    return np.array([1./3., 1./3., 1./3., 0., 0., 0.])
    

def compute_dfdp(p, pc01):
    return m_theta * m_theta * (2 * p - pc01)
    
    
def compute_plastic_modulus(p, pc01, void_ratio):
    upslion = (1 + void_ratio) / (lambda_ - kappa)
    dfdpc = -m_theta * m_theta * p
    dpdmul = pc01 * upslion
    dfdp = compute_dfdp(p, pc01)
    return -dfdpc * dpdmul * dfdp
        

def compute_dfdsigma(p, devsig, pc01):
    return compute_dfdp(p, pc01) * compute_dpdsigma() + 3. * devsig
        
        
def compute_voidratio(p, pc01):
    return e_ref - lambda_ * np.log(pc01 / p_ref) + kappa * np.log(pc01 / p) 
        

def compute_elastic_modulus(p, void_ratio):
    bulk_modulus = (1. + void_ratio) / kappa * p
    shear_modulus = 3. * bulk_modulus * (1 - 2 * possion) / (2 * (1 + possion))
    return bulk_modulus, shear_modulus
    
    
def compute_internal_variable_increment(void_ratio, p, pc01, dlambda):
    dfdp = compute_dfdp(p, pc01)
    upslion = (1 + void_ratio) / (lambda_ - kappa)
    return upslion * pc01 * dlambda * dfdp
   
   
def compute_yield_function(stress, pc01):
    p = meanstress(stress)
    q = equistress(stress)

    return (q / m_theta) * (q / m_theta) + p * (p - pc01)
   
        
def get_analytical(strain, stress, pc01, depsilon_a):
    p = meanstress(stress)
    devsig = deviatoric_stress(stress)
    q = equistress(stress)
    void_ratio = compute_voidratio(p, pc01)
    bulk_modulus, shear_modulus = compute_elastic_modulus(p, void_ratio)
    a1 = bulk_modulus + 4./3. * shear_modulus
    a2 = bulk_modulus - 2./3. * shear_modulus
    
    De = np.array([[a1, a2, a2, 0, 0, 0],
                   [a2, a1, a2, 0, 0, 0],
                   [a2, a2, a1, 0, 0, 0],
                   [0, 0, 0, shear_modulus, 0, 0],
                   [0, 0, 0, 0, shear_modulus, 0],
                   [0, 0, 0, 0, 0, shear_modulus]])
                   
    dstrain = np.array([-a2 / (a1 + a2) * depsilon_a, -a2 / (a1 + a2) * depsilon_a, depsilon_a, 0, 0, 0])
    destress = np.dot(De, dstrain)
    trial_stress = stress + destress
    trial_p = meanstress(trial_stress)
    trial_q = equistress(trial_stress)
    trial_f_func = compute_yield_function(trial_stress, pc01)
    
    Dp = np.zeros((6, 6))
    if trial_f_func > -1e-8:
        dfdsigma = compute_dfdsigma(p, devsig, pc01)
        Kp = compute_plastic_modulus(p, pc01, void_ratio)
        dfdsigmaDedgdsigma = dfdsigma @ De @ dfdsigma
        Dp = np.outer((De @ dfdsigma), (dfdsigma @ De)) / (Kp + dfdsigmaDedgdsigma)
        pc01 = (q * q / m_theta / m_theta + p * p) / p
        
    Dep = De - Dp
    depsilon_r = -Dep[0,2] * depsilon_a / (Dep[0,0] + Dep[0,1])
    dstrain = np.array([depsilon_r, depsilon_r, depsilon_a, 0, 0, 0])
    dstress = np.dot(Dep, dstrain)
    stress = stress + dstress
    return stress, pc01, strain + dstrain
    

def drained(pressure):
    p = []
    q = []
    ratio = []
    time = []
    
    depsilon_a = 1e-4
    pc01 = pc0
    stress = np.array([pressure, pressure, pressure, 0, 0, 0])
    strain = np.array([0, 0, 0, 0, 0, 0])
    count = 0
    while count * depsilon_a < 0.25:
        p.append(meanstress(stress))
        q.append(equistress(stress))
        ratio.append(equistress(stress)/meanstress(stress))
        time.append(strain[2])
        stress, pc01, strain = get_analytical(strain, stress, pc01, depsilon_a)
        count += 1
        
    p.append(meanstress(stress))
    q.append(equistress(stress))  
    ratio.append(equistress(stress)/meanstress(stress))
    time.append(strain[2])  
    return p, q, ratio, time
        
pa1, qa1, ratioa1, timea1 = drained(33000)
pa2, qa2, ratioa2, timea2 = drained(98000)
pa3, qa3, ratioa3, timea3 = drained(303000)

p1, q1, pc1, ratio1, time1 = get_result("33kpa")
p2, q2, pc2, ratio2, time2 = get_result("98kpa")
p3, q3, pc3, ratio3, time3 = get_result("303kpa")

yield_p = np.linspace(0, 2*pc1[0], 200)    
yield_p0 = np.linspace(0, pc1[0], 200)
yield_q0 = np.sqrt(m_theta**2*yield_p0*(pc1[0]-yield_p0))

yield_p1 = np.linspace(0, pc1[end_num-start_num-1], 200)
yield_q1 = np.sqrt(m_theta**2*yield_p1*(pc1[end_num-start_num-1]-yield_p1))

yield_p2 = np.linspace(0, pc2[end_num-start_num-1], 200)
yield_q2 = np.sqrt(m_theta**2*yield_p2*(pc2[end_num-start_num-1]-yield_p2))

yield_p3 = np.linspace(0, pc3[end_num-start_num-1], 200)
yield_q3 = np.sqrt(m_theta**2*yield_p3*(pc3[end_num-start_num-1]-yield_p3))

CSL = m_theta*yield_p

fig, ax=plt.subplots()
ax.plot(yield_p0, yield_q0, color='black')
ax.plot(pa1, qa1, label='Analytical', color=(0/255, 128/255, 0/255))
ax.plot(pa2, qa2, label='Analytical', color=(255/255, 0/255, 0/255))
ax.plot(pa3, qa3, label='Analytical', color=(94/255, 114/255, 255/255))
ax.scatter(p1, q1, s=125, label='GeoTaichi', color=(0/255, 128/255, 0/255))
ax.scatter(p2, q2, s=125, label='GeoTaichi', color=(255/255, 0/255, 0/255))
ax.scatter(p3, q3, s=125, label='GeoTaichi', color=(94/255, 114/255, 255/255))
ax.plot(yield_p, CSL, color='black')
ax.set_xlabel("Mean stress, $p$ (Pa)")
ax.set_ylabel("Equivalent stress, $q$ (Pa)")
ax.set_xlim([0,800000])
ax.set_ylim([0,800000])
ax.legend(frameon=False)
tablelegend(ax, ncol=2, frameon=False, row_labels=['$p$=33kPa', '$p$=98kPa', '$p$=303kPa'], col_labels=['Analytical', 'GeoTaichi'], columnspacing=1, title_label='Confining pressure')
fig.tight_layout()
fig.savefig ("pqcurve.svg")
plt.close()

fig, ax=plt.subplots()
x=np.linspace(0, 0.3, 10)
y=np.zeros(10)+m_theta   
ax.plot(timea1, qa1, color=(0/255, 128/255, 0/255), label='Analytical')
ax.plot(timea2, qa2, color=(255/255, 0/255, 0/255), label='Analytical')
ax.plot(timea3, qa3, color=(94/255, 114/255, 255/255), label='Analytical')
ax.scatter(time1, q1, s=125, color=(0/255, 128/255, 0/255), label='GeoTaichi')
ax.scatter(time2, q2, s=125, color=(255/255, 0/255, 0/255), label='GeoTaichi')
ax.scatter(time3, q3, s=125, color=(94/255, 114/255, 255/255), label='GeoTaichi')
ax.set_xlim([0,0.5])
ax.set_ylim([0,500000])
ax.set_xlabel("Axial strain, $\epsilon_a$ (\%)")
ax.set_ylabel("Equivalent stress, $q$ (Pa)")
ax.legend(frameon=False)
tablelegend(ax, ncol=2, frameon=False, row_labels=['$p$=33kPa', '$p$=98kPa', '$p$=303kPa'], col_labels=['Analytical', 'GeoTaichi'], columnspacing=1, title_label='Confining pressure')
fig.tight_layout()
fig.savefig ("qtcurve.svg")
plt.close()

print((q1[-1]-q1[0])/(p1[-1]-p1[0]))
print((q2[-1]-q2[0])/(p2[-1]-p2[0]))
print((q3[-1]-q3[0])/(p3[-1]-p3[0]))


