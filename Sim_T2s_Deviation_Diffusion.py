# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns

# %%Estimate T2* Deviations based on Kleban at et Literature Values 
from functions import MacroR2s

R2s_intra = 14.3 #s−1
R2s_extra = 17.5 #s−1

D_slow_in_vivo = 0.07*10**(-3)  # mm**2/s
D_fast_in_vivo = 1.03*10**(-3)  # mm**2/s

# For ex vivo estimations, we will assume a reduction of diffusivity by a factor 3
D_slow_ex_vivo = 0.07*10**(-3)/3  # mm**2/s
D_fast_ex_vivo = 1.03*10**(-3)/3  # mm**2/s

# Datathief generated f_in histogram from Kleban et al.
f_in_hist = np.genfromtxt('f_in_histogram_kleban.csv', delimiter = ',')
f_in = f_in_hist.prod(axis = 1).sum()

# Select b value range for simulations
b_range_ex_vivo = np.linspace(0, 20000, 41)
b_range_in_vivo = np.linspace(0, 10000, 21)

R2s_macro_range_in_vivo = np.zeros_like(b_range_in_vivo)
R2s_macro_range_ex_vivo = np.zeros_like(b_range_ex_vivo)

for i_b in np.arange(b_range_in_vivo.shape[0]):
    R2s_macro_range_in_vivo[i_b] = MacroR2s(f_in, R2s_extra, R2s_intra, D_fast_in_vivo, D_slow_in_vivo, b_range_in_vivo[i_b])

for i_b in np.arange(b_range_ex_vivo.shape[0]):
    R2s_macro_range_ex_vivo[i_b] = MacroR2s(f_in, R2s_extra, R2s_intra, D_fast_ex_vivo, D_slow_ex_vivo, b_range_ex_vivo[i_b])
   
T2s_macro_range_in_vivo = 1/R2s_macro_range_in_vivo
T2s_macro_range_in_vivo_norm = T2s_macro_range_in_vivo/T2s_macro_range_in_vivo[0]

T2s_macro_range_ex_vivo = 1/R2s_macro_range_ex_vivo
T2s_macro_range_ex_vivo_norm = T2s_macro_range_ex_vivo/T2s_macro_range_ex_vivo[0]


####################333
# Simulation of MLE Reconstruction with T2* Deviation
from functions import T2star_decay, T2star_decay_noise, LLS, LL_exp_gauss_fixT2

# Set the simulation parameters
S0 = 1
SNR = 50
TEs = np.linspace(0,23.6,5)
reps = 1000
lower_bound = 5e-4

sigma = S0/SNR

T2s_assumed_in_vivo = T2s_macro_range_in_vivo[0]*1000 # Base T2*, chose in vivo setting [ms]
T2s_actual_in_vivo = T2s_macro_range_in_vivo_norm * T2s_assumed_in_vivo

T2s_assumed_ex_vivo = 30 # Base T2*, chose in vivo setting [ms]
T2s_actual_ex_vivo = T2s_macro_range_ex_vivo_norm * T2s_assumed_ex_vivo


# Initialize Matrix for fit values
fit_gauss_lls_in_vivo = np.zeros((reps, len(T2s_actual_in_vivo)))
fit_gauss_mle_in_vivo = np.zeros((reps, len(T2s_actual_in_vivo)))
fit_gauss_lls_ex_vivo = np.zeros((reps, len(T2s_actual_ex_vivo)))
fit_gauss_mle_ex_vivo = np.zeros((reps, len(T2s_actual_ex_vivo)))

for i_T2 in range(len(T2s_actual_in_vivo)): # Loop over T2* decays
    for i_rep in range(reps): # Loop over Monte Carlo Repetitions

    	############
    	# IN VIVO

        # Generate Synthetic data with Gaussian distribution
        tmp_data_in_vivo = T2star_decay_noise(TEs, S0, T2s_actual_in_vivo[i_T2], sigma)
        tmp_data_in_vivo_gauss = np.real(tmp_data_in_vivo)

        ####################
        ### Real Valued Data with Gaussian Distribution

        # Fit LLS on Data
        tmp_fit_gauss_lls_in_vivo = LLS(tmp_data_in_vivo_gauss, TEs, T2s_assumed_in_vivo)
        fit_gauss_lls_in_vivo[i_rep, i_T2] = tmp_fit_gauss_lls_in_vivo

        # MLE Fit with Fix T2 on Gaussian Data
        bnds = [(lower_bound, 2*np.abs(tmp_data_in_vivo_gauss).sum())] # Force results to be positive
        args = [TEs, tmp_data_in_vivo_gauss, sigma, T2s_assumed_in_vivo]
        if tmp_fit_gauss_lls_in_vivo < tmp_data_in_vivo_gauss.sum():
            guess = [tmp_fit_gauss_lls_in_vivo]
        else:
            guess = [tmp_data_in_vivo_gauss.mean()]

        # If minimize does not, skip errorous runs. 
        try:
            param_MLE_gauss_T2fix = minimize(LL_exp_gauss_fixT2, x0 = guess, args = args, bounds = bnds)
        except RuntimeError:
            print("Minimize did not converge - setting initial guess as fit for MLE with fix T2*")
            param_MLE_gauss_T2fix.x[0] = tmp_fit_gauss_lls_in_vivo
        tmp_fit_gauss_mle_in_vivo = param_MLE_gauss_T2fix.x[0]

        fit_gauss_mle_in_vivo[i_rep, i_T2] = tmp_fit_gauss_mle_in_vivo

    print('In Vivo Diffusion Weighting b = {:.2f} / {} [s/mm2]'.format(b_range_in_vivo[i_T2], max(b_range_in_vivo)))

for i_T2 in range(len(T2s_actual_ex_vivo)): # Loop over T2* decays
    for i_rep in range(reps): # Loop over Monte Carlo Repetitions
    	############
    	# EX VIVO

        # Generate Synthetic data with Gaussian distribution
        tmp_data_ex_vivo = T2star_decay_noise(TEs, S0, T2s_actual_ex_vivo[i_T2], sigma)
        tmp_data_ex_vivo_gauss = np.real(tmp_data_ex_vivo)

        ####################
        ### Real Valued Data with Gaussian Distribution

        # Fit LLS on Data
        tmp_fit_gauss_lls_ex_vivo = LLS(tmp_data_ex_vivo_gauss, TEs, T2s_assumed_ex_vivo)
        fit_gauss_lls_ex_vivo[i_rep, i_T2] = tmp_fit_gauss_lls_ex_vivo

        # MLE Fit with Fix T2 on Gaussian Data
        bnds = [(lower_bound, 2*np.abs(tmp_data_ex_vivo_gauss).sum())] # Force results to be positive
        args = [TEs, tmp_data_ex_vivo_gauss, sigma, T2s_assumed_ex_vivo]
        if tmp_fit_gauss_lls_ex_vivo < tmp_data_ex_vivo_gauss.sum():
            guess = [tmp_fit_gauss_lls_ex_vivo]
        else:
            guess = [tmp_data_ex_vivo_gauss.mean()]

        # If minimize does not, skip errorous runs. 
        try:
            param_MLE_gauss_T2fix = minimize(LL_exp_gauss_fixT2, x0 = guess, args = args, bounds = bnds)
        except RuntimeError:
            print("Minimize did not converge - setting initial guess as fit for MLE with fix T2*")
            param_MLE_gauss_T2fix.x[0] = tmp_fit_gauss_lls_ex_vivo
        tmp_fit_gauss_mle_ex_vivo = param_MLE_gauss_T2fix.x[0]

        fit_gauss_mle_ex_vivo[i_rep, i_T2] = tmp_fit_gauss_mle_ex_vivo

    print('Ex Vivo Diffusion Weighting b = {:.2f} / {} [s/mm2]'.format(b_range_ex_vivo[i_T2], max(b_range_ex_vivo)))




# Get rid of all NaN values
fit_gauss_lls_in_vivo[np.isnan(fit_gauss_lls_in_vivo)] = 0
fit_gauss_mle_in_vivo[np.isnan(fit_gauss_mle_in_vivo)] = 0
fit_gauss_lls_ex_vivo[np.isnan(fit_gauss_lls_ex_vivo)] = 0
fit_gauss_mle_ex_vivo[np.isnan(fit_gauss_mle_ex_vivo)] = 0

# Generate "Actual T2*" array for Pandas dataframe
T2s_actual_in_vivo_array = np.zeros(fit_gauss_mle_in_vivo.shape)
T2s_actual_in_vivo_array[:, ...] = T2s_actual_in_vivo[None, :]
T2s_actual_ex_vivo_array = np.zeros(fit_gauss_mle_ex_vivo.shape)
T2s_actual_ex_vivo_array[:, ...] = T2s_actual_ex_vivo[None, :]

# Generate "B-Value" array for Pandas dataframe
B_value_array_in_vivo = np.zeros(fit_gauss_mle_in_vivo.shape)
B_value_array_in_vivo[:, ...] = b_range_in_vivo[None, :]
B_value_array_ex_vivo = np.zeros(fit_gauss_mle_ex_vivo.shape)
B_value_array_ex_vivo[:, ...] = b_range_ex_vivo[None, :]


dict_result_in_vivo = {'Reconstructed Signal' : fit_gauss_mle_in_vivo.ravel(), 'T2*': T2s_actual_in_vivo_array.ravel(), 'Diffusion Weighting': B_value_array_in_vivo.ravel()}
df_results_in_vivo = pd.DataFrame(data = dict_result_in_vivo)
dict_result_ex_vivo = {'Reconstructed Signal' : fit_gauss_mle_ex_vivo.ravel(), 'T2*': T2s_actual_ex_vivo_array.ravel(), 'Diffusion Weighting': B_value_array_ex_vivo.ravel()}
df_results_ex_vivo = pd.DataFrame(data = dict_result_ex_vivo)


# Plot the results 
thickness = 2.

plt.subplot(221)
plt.title('T2* under Diffusion Weighting IN VIVO')
plt.plot(b_range_in_vivo, T2s_actual_in_vivo, label = 'in vivo')
plt.grid('on')
plt.xlabel('b value [s/mm2]')
plt.ylabel('T2* [ms]')

plt.subplot(223)
sns.lineplot(x='Diffusion Weighting', y="Reconstructed Signal",  linewidth = thickness, data= df_results_in_vivo, ci = 'sd', zorder = 1, label = 'Reconstructed Signal')
plt.hlines(S0, min(b_range_in_vivo), max(b_range_in_vivo), linestyle = 'dotted',  linewidth = thickness, label = 'Ground Truth S0', zorder = 2) 
plt.legend()
plt.grid('on')

plt.subplot(222)
plt.title('T2* under Diffusion Weighting EX VIVO')
plt.plot(b_range_ex_vivo, T2s_actual_ex_vivo, label = 'ex vivo')
plt.grid('on')
plt.xlabel('b value [s/mm2]')
plt.ylabel('T2* [ms]')

plt.subplot(224)
sns.lineplot(x='Diffusion Weighting', y="Reconstructed Signal",  linewidth = thickness, data= df_results_ex_vivo, ci = 'sd', zorder = 1, label = 'Reconstructed Signal')
plt.hlines(S0, min(b_range_ex_vivo), max(b_range_ex_vivo), linestyle = 'dotted',  linewidth = thickness, label = 'Ground Truth S0', zorder = 2) 
plt.legend()
plt.grid('on')
plt.show()