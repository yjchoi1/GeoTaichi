#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import sys
import os
import glob

base_path = "/home/yj/works/GeoTaichi_Yihao_v1/examples/ElementTest/UndrainedMCC"

pid = 0 
m_theta = 1.02
poission = 0.25
pc0 = 392000
lambda_ = 0.12
kappa = 0.023
e_ref = 1.7
p_ref = 1000

# start_num and end_num are deprecated (auto-detect files)


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
    p, q, pc, ratio, time = [], [], [], [], []
    # Support absolute or relative paths
    path_abs = path if os.path.isabs(path) else os.path.join(base_path, path)
    files = sorted(glob.glob(os.path.join(path_abs, 'particles', 'MPMParticle*.npz')))
    for f in files:
        data = np.load(f, allow_pickle=True)
        sig = -data['stress'][pid]
        mean_stress = meanstress(sig)
        equivalent_stress = equistress(sig)
        pc01 = data['state_vars'].item()['pc'][pid]

        p.append(mean_stress)
        q.append(equivalent_stress)
        pc.append(pc01)
        ratio.append(equivalent_stress / mean_stress)
        time.append(data['t_current']*0.01)
    return p, q, pc, ratio, time
    
    
def compute_dpdsigma():
    return np.array([1./3., 1./3., 1./3., 0., 0., 0.])
    

def compute_dfdp(p, pc01):
    return m_theta * m_theta * (2 * p - pc01)
    
    
def compute_elastic_modulus(p, void_ratio):
    bulk_modulus = (1. + void_ratio) / kappa * p
    shear_modulus = 3. * bulk_modulus * (1 - 2 * poission) / (2 * (1 + poission))
    return bulk_modulus, shear_modulus
    
    
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
    
    
def compute_internal_variable_increment(void_ratio, p, pc01, dlambda):
    dfdp = compute_dfdp(p, pc01)
    upslion = (1 + void_ratio) / (lambda_ - kappa)
    return upslion * pc01 * dlambda * dfdp
   
   
def compute_yield_function(stress, pc01):
    p = meanstress(stress)
    q = equistress(stress)

    return (q / m_theta) * (q / m_theta) + p * (p - pc01)


def compute_elastic_stiffness(p, void_ratio):
    bulk_modulus, shear_modulus = compute_elastic_modulus(p, void_ratio)
    a1 = bulk_modulus + 4./3. * shear_modulus
    a2 = bulk_modulus - 2./3. * shear_modulus
    
    return np.array([[a1, a2, a2, 0, 0, 0],
                     [a2, a1, a2, 0, 0, 0],
                     [a2, a2, a1, 0, 0, 0],
                     [0, 0, 0, shear_modulus, 0, 0],
                     [0, 0, 0, 0, shear_modulus, 0],
                     [0, 0, 0, 0, 0, shear_modulus]])

   
def sub_stepping(dsig_e, stress, pc01, void_ratio):
    p = meanstress(stress)
    devsig = deviatoric_stress(stress)
    De = compute_elastic_stiffness(p, void_ratio)

    dfdsigma = compute_dfdsigma(p, devsig, pc01)
    dfdp = compute_dfdp(p, pc01)
    Kp = compute_plastic_modulus(p, pc01, void_ratio)
    dfdsigmaDedgdsigma = dfdsigma @ De @ dfdsigma
    dlambda = max(np.dot(dfdsigma, dsig_e) / (dfdsigmaDedgdsigma + Kp), 0)
    dsig = dsig_e - dlambda * De @ dfdsigma
    dpc01 = compute_internal_variable_increment(void_ratio, p, pc01, dlambda)
    return dsig, dpc01
   
   
def consistent_correction(f_function, stress, pc01):
    p = meanstress(stress)
    devsig = deviatoric_stress(stress)
    void_ratio = compute_voidratio(p, pc01)
    De = compute_elastic_stiffness(p, void_ratio)

    dfdsigma = compute_dfdsigma(p, devsig, pc01)
    Kp = compute_plastic_modulus(p, pc01, void_ratio)
    dfdsigmaDedgdsigma = dfdsigma @ De @ dfdsigma
    dlambda = f_function / (dfdsigmaDedgdsigma + Kp)
    stress_new = stress - dlambda * De @ dfdsigma
    depstrain = dlambda * dfdsigma
    dpc01 = compute_internal_variable_increment(void_ratio, p, pc01, dlambda)
    pc01_new = pc01 + dpc01
    return stress_new, pc01_new
    

def normal_correction(f_function, stress, pc01):
    p = meanstress(stress)
    devsig = deviatoric_stress(stress)
    dfdsigma = compute_dfdsigma(p, devsig, pc01)

    dlambda = f_function / np.dot(dfdsigma, dfdsigma)
    stress_new = stress - dlambda * dfdsigma
    return stress_new 
   
   
def drift_correct(f_function, stress, pc01):
    for _ in range(30):
        stress_new, pc01_new = consistent_correction(f_function, stress, pc01)
        f_function_new = compute_yield_function(stress_new, pc01)

        if abs(f_function_new) > abs(f_function):
            stress_new = normal_correction(f_function, stress, pc01)
            f_function_new = compute_yield_function(stress_new, pc01)
            pc01_new = pc01

        if abs(f_function_new) <= 1e-8:
            stress = stress_new
            pc01 = pc01_new
            break

        stress = stress_new
        pc01 = pc01_new
        f_function = f_function_new
    return stress, pc01
   
        
def get_analytical(strain, stress, pc01, depsilon_a):
    p = meanstress(stress)
    devsig = deviatoric_stress(stress)
    q = equistress(stress)
    void_ratio = compute_voidratio(p, pc01)
    De = compute_elastic_stiffness(p, void_ratio)
                   
    dstrain = np.array([-0.5 * depsilon_a, -0.5 * depsilon_a, depsilon_a, 0, 0, 0])
    destress = np.dot(De, dstrain)
    trial_stress = stress + destress
    trial_p = meanstress(trial_stress)
    trial_q = equistress(trial_stress)
    trial_f_func = compute_yield_function(trial_stress, pc01)
    
    stress = trial_stress
    Dp = np.zeros((6, 6))
    if trial_f_func > -1e-8:
        dfdsigma = compute_dfdsigma(p, devsig, pc01)
        Kp = compute_plastic_modulus(p, pc01, void_ratio)
        dfdsigmaDedgdsigma = dfdsigma @ De @ dfdsigma
        Dp = np.outer((De @ dfdsigma), (dfdsigma @ De)) / (Kp + dfdsigmaDedgdsigma)
        dlambda = trial_f_func / (Kp + dfdsigmaDedgdsigma)
        stress -= dlambda * De @ dfdsigma
        pc01 += compute_internal_variable_increment(void_ratio, p, pc01, dlambda)
        
        f_func = compute_yield_function(stress, pc01)
        if f_func > -1e-8:
            stress, pc01 = drift_correct(f_func, stress, pc01)
    return stress, pc01, strain + dstrain
    

def drained(pressure):
    p = []
    q = []
    ratio = []
    time = []
    
    depsilon_a = 0.0001
    pc01 = pc0
    stress = np.array([pressure, pressure, pressure, 0, 0, 0])
    strain = np.array([0, 0, 0, 0, 0, 0])
    count = 0
    while count * depsilon_a < 0.1:
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

p1, q1, pc1, ratio1, time1 = get_result(f"{base_path}/33kpa")
p2, q2, pc2, ratio2, time2 = get_result(f"{base_path}/98kpa")
p3, q3, pc3, ratio3, time3 = get_result(f"{base_path}/303kpa")

dat1 = np.array([[0,306000,0],[0.001,305000,80000],[0.002,295000,124000],[0.009,261000,176000],[0.024,228000,200000],[.06,212000,210000],[.08,206000,208000]])
dat2 = np.array([[0,98000,0],[0.002,103000,37000],[0.003,110000,75000],[0.01,130000,100000],[0.016,142000,150000],[.02,149000,165000],[.03,151000,174000],[.08,176000,180000]])
dat3 = np.array([[0,33000,0],[0.004,36000,23000],[0.01,48000,54000],[0.015,55000,76000],[0.02,71000,100000],[.03,92000,129000],[.04,118000,148000],[.06,130000,152000],[.08,145000,151000]])

yield_p = np.linspace(0, 1.2*pc1[0], 200)    
yield_p0 = np.linspace(0, pc1[0], 200)
yield_q0 = np.sqrt(m_theta**2*yield_p0*(pc1[0]-yield_p0))

yield_p1 = np.linspace(0, pc1[-1], 200)
yield_q1 = np.sqrt(m_theta**2*yield_p1*(pc1[-1]-yield_p1))

yield_p2 = np.linspace(0, pc2[-1], 200)
yield_q2 = np.sqrt(m_theta**2*yield_p2*(pc2[-1]-yield_p2))

yield_p3 = np.linspace(0, pc3[-1], 200)
yield_q3 = np.sqrt(m_theta**2*yield_p3*(pc3[-1]-yield_p3))

CSL = m_theta*yield_p

fig = plt.figure(figsize=(15,6))
fig.suptitle('Undrained Test (Modified Cam Clay)', size=18)

ax1=plt.subplot(1,2,1)    
ax1.plot(yield_p0, yield_q0, color='black')
ax1.plot(yield_p, CSL, color='black')
ax1.plot(pa1, qa1, label='p=33kPa Analytical', color=(0/255, 128/255, 0/255), alpha=0.5)
ax1.plot(pa2, qa2, label='p=98kPa Analytical', color=(255/255, 0/255, 0/255), alpha=0.5)
ax1.plot(pa3, qa3, label='p=303kPa Analytical', color=(94/255, 114/255, 255/255), alpha=0.5)
# ax1.plot(dat3[:,1],dat3[:,2], label='p=33kPa Experiment', color=(0/255, 128/255, 0/255), linestyle='--',marker='x')
# ax1.plot(dat2[:,1],dat2[:,2], label='p=98kPa Experiment', color=(255/255, 0/255, 0/255), linestyle='--',marker='x')
# ax1.plot(dat1[:,1],dat1[:,2], label='p=303kPa Experiment', color=(94/255, 114/255, 255/255), linestyle='--',marker='x')
ax1.scatter(p1, q1, color=(0/255, 128/255, 0/255), label='p=33kPa Experiment')
ax1.scatter(p2, q2, color=(255/255, 0/255, 0/255), label='p=98kPa Experiment')
ax1.scatter(p3, q3, color=(94/255, 114/255, 255/255), label='p=303kPa Experiment')
ax1.set_xlabel("mean stress")
ax1.set_ylabel("equivalent stress")
ax1.legend(loc='best')

x=np.linspace(0, 0.3, 10)
y=np.zeros(10)+m_theta
ax2=plt.subplot(1,2,2)    
ax2.plot(timea1, qa1, color=(0/255, 128/255, 0/255), alpha=0.5)
ax2.plot(timea2, qa2, color=(255/255, 0/255, 0/255), alpha=0.5)
ax2.plot(timea3, qa3, color=(94/255, 114/255, 255/255), alpha=0.5)
# ax2.plot(dat3[:,0],dat3[:,2], color=(0/255, 128/255, 0/255), linestyle='--',marker='x')
# ax2.plot(dat2[:,0],dat2[:,2], color=(255/255, 0/255, 0/255), linestyle='--',marker='x')
# ax2.plot(dat1[:,0],dat1[:,2], color=(94/255, 114/255, 255/255), linestyle='--',marker='x')
ax2.scatter(time1, q1, color=(0/255, 128/255, 0/255), label='p=33kPa Experiment')
ax2.scatter(time2, q2, color=(255/255, 0/255, 0/255), label='p=98kPa Experiment')
ax2.scatter(time3, q3, color=(94/255, 114/255, 255/255), label='p=303kPa Experiment')
ax2.set_xlim([0,0.08])
ax2.set_xlabel("axial strain")
ax2.set_ylabel("equivalent stress")
plt.savefig(f"{base_path}/result.png")
plt.show()

