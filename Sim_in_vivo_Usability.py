# -*- coding: utf-8 -*-
import numpy as np
from functions import G_SNR

#####
# %% Set Up Scan Parameters
Setup = 'ExVivo'    

if Setup == 'InVivo':
    PE_size = 128   # Matrix Dimensions 
    PAT = 2         # Parallel Acceleartion
    PF = 0.75       # Partial Fourier Factor
    Segm = 3        # Segmentation Factor
    ES = 0.6        # Echo Spacing in ms
    TE = 80         # Echo Time in ms
    Echos = 3       # Number of Echos
    
if Setup == 'ExVivo':
    PE_size = 120   # Matrix Dimensions 
    PAT = 1         # Parallel Acceleartion
    PF = 1          # Partial Fourier Factor
    Segm = 40       # Segmentation Factor
    ES = 1.98335    # Echo Spacing in ms
    TE = 45         # Echo Time in ms
    Echos = 10      # Number of Echos
# Calculate Duration of EPI Readout Train
Readout_Duration = ES * (PE_size * PF) / PAT /Segm
print('Readout Duration = {:.2f}ms, Number of Lines = {}'.format(Readout_Duration, int(Readout_Duration/ES)))

# %%Calculate Additional Time Expense from Multi Echo
Growth = 100*(TE+Readout_Duration/2 + (Echos-1)*Readout_Duration)/(TE+Readout_Duration/2)
TEs = np.linspace(0, (Echos-1)*Readout_Duration, Echos)
print('TEs = {}'.format(TEs + TE))

# %% Set up Relaxation Times 
T2_CC = 58.5    #ms
T2_Cing = 50.3  #ms
T2_CST = 53.3   #ms
T2_exvivo_WM = 30  #ms
T2_exvivo_GM = 58  #ms

# %% Calculate SNR Gain
if Setup == 'InVivo':
    G_SNR_CC = G_SNR(T2_CC, TEs)
    G_SNR_Cing = G_SNR(T2_Cing, TEs)
    G_SNR_CST = G_SNR(T2_CST, TEs)
    
    print('---')
    print('SNR Gain CC = {:.2f} - comparable to {:.2f} averages'.format(G_SNR_CC, G_SNR_CC**2))
    print('SNR Gain Cingulum = {:.2f} - comparable to {:.2f} averages'.format(G_SNR_Cing, G_SNR_Cing**2))
    print('SNR Gain CST = {:.2f} - comparable to {:.2f} averages'.format(G_SNR_CST, G_SNR_CST**2))

if Setup == 'ExVivo':
    G_SNR_exvivo_WM = G_SNR(T2_exvivo_WM, TEs)
    G_SNR_exvivo_GM = G_SNR(T2_exvivo_GM, TEs)

    print('---')
    print('SNR Gain Ex Vivo WM = {:.2f} - comparable to {:.2f} averages'.format(G_SNR_exvivo_WM, G_SNR_exvivo_WM**2))
    print('SNR Gain Ex Vivo GM = {:.2f} - comparable to {:.2f} averages'.format(G_SNR_exvivo_GM, G_SNR_exvivo_GM**2))

print('Duration of ME dMRI Acquisition {:.1f}% of original acquisition'.format(Growth))

# %% Loop through number of echoes to show dependency of SNR Gain in White and Gray Matter Ex Vivo (Reviewer Question)
import pylab as plt

max_n_Echoes = 100
n_Echoes= np.linspace(1 ,max_n_Echoes, max_n_Echoes, dtype = int)
Readout_Duration = ES * (PE_size * PF) / PAT /Segm
TE_full = np.linspace(0, (max(n_Echoes)-1)*Readout_Duration, max(n_Echoes))

iG_SNR_WM = np.ones(n_Echoes.shape)
iG_SNR_GM = np.ones(n_Echoes.shape)

for ii in range(len(n_Echoes)):
    if ii > 0:
        iG_SNR_WM[ii] = G_SNR(T2_exvivo_WM, TE_full[:ii])
        iG_SNR_GM[ii] = G_SNR(T2_exvivo_GM, TE_full[:ii])
plt.figure()
plt.subplot(121)
plt.title('SNR Gain for Different Numbers of Echoes')
plt.plot(n_Echoes[1::], (iG_SNR_WM)[1::], label = 'WM T2*={}ms'.format(T2_exvivo_WM))
plt.plot(n_Echoes[1::], (iG_SNR_GM)[1::], label = 'GM T2*={}ms'.format(T2_exvivo_GM))
plt.grid('on')
plt.ylabel('G SNR')
plt.xlabel('Number of Echoes')
plt.legend()
plt.subplot(122)
plt.title('SNR Gain for Different Numbers of Echoes (Zoom)')
plt.plot(n_Echoes[1:10], (iG_SNR_WM)[1:10], label = 'WM T2*={}ms'.format(T2_exvivo_WM))
plt.plot(n_Echoes[1:10], (iG_SNR_GM)[1:10], label = 'GM T2*={}ms'.format(T2_exvivo_GM))
plt.grid('on')
plt.ylabel('G SNR')
plt.xlabel('Number of Echoes')
plt.legend()
plt.show()