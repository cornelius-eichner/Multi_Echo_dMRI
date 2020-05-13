import numpy as np
from scipy.optimize import minimize
import pylab as plt
#from scipy.signal import savgol_filter
import pandas as pd
import seaborn as sns

sns.set()
sns.color_palette("deep")
sns.set_style("whitegrid")

from functions import T2star_decay, T2star_decay_noise, wLLS, LLS, LL_exp_gauss_fixT2, LL_exp_rice_fixT2, G_SNR

##############################
# %%Simulate SNR gains for LLS and wLLS using Monte Carlo for different T2 values

S0 = 1
TEs = np.linspace(0,23.6,5)
T2min = 1
T2max = 100
T2_steps = 100
reps = 1000
sigma = 0.2
lower_bound = 5e-4

print('Simulate SNR gains for LLS and MLE using Monte Carlo for different T2 values for SNR = {:.2f}'.format(S0/sigma))

T2_levels = np.linspace(T2min, T2max, T2_steps)

# Initialize Matrix for fit values
fit_gauss_lls = np.zeros((reps, T2_steps))
fit_gauss_wlls = np.zeros((reps, T2_steps))
fit_gauss_mle = np.zeros((reps, T2_steps))

fit_rice_lls = np.zeros((reps, T2_steps))
fit_rice_wlls = np.zeros((reps, T2_steps))
fit_rice_mle = np.zeros((reps, T2_steps))

Gain_LLS = []

for iT2 in range(len(T2_levels)): # Loop over T2* decays
    for i_rep in range(reps): # Loop over Monte Carlo Repetitions

        # Generate Synthetic data with Gaussian distribution
        tmp_data = T2star_decay_noise(TEs, S0, T2_levels[iT2], sigma)
        tmp_data_gauss = np.real(tmp_data)
        tmp_data_rice = np.abs(tmp_data)

        ####################
        ### Real Valued Data with Gaussian Distribution

        # Fit LLS on Data
        tmp_fit_gauss_lls = LLS(tmp_data_gauss, TEs, T2_levels[iT2])
        fit_gauss_lls[i_rep, iT2] = tmp_fit_gauss_lls

        # Fit wLLS on Data
        tmp_fit_gauss_wlls = wLLS(tmp_data_gauss, TEs, T2_levels[iT2])
        fit_gauss_wlls[i_rep, iT2] = tmp_fit_gauss_wlls

        # MLE Fit with Fix T2 on Gaussian Data
        bnds = [(lower_bound, 2*np.abs(tmp_data_gauss).sum())] # Force results to be positive
        args = [TEs, tmp_data_gauss, sigma, T2_levels[iT2]]
        if tmp_fit_gauss_lls < tmp_data_gauss.sum():
            guess = [tmp_fit_gauss_lls]
        else:
            guess = [tmp_data_gauss.mean()]

        # If minimize does not, skip errorous runs. 
        try:
            param_MLE_gauss_T2fix = minimize(LL_exp_gauss_fixT2, x0 = guess, args = args, bounds = bnds)
        except RuntimeError:
            print("Minimize did not converge - setting initial guess as fit for MLE with fix T2*")
            param_MLE_gauss_T2fix.x[0] = tmp_fit_gauss_lls
        tmp_fit_gauss_mle = param_MLE_gauss_T2fix.x[0]

        fit_gauss_mle[i_rep, iT2] = tmp_fit_gauss_mle


        ####################
        ### Mag Valued Data

        # Fit LLS on Data
        tmp_fit_rice_lls = LLS(tmp_data_rice, TEs, T2_levels[iT2])
        fit_rice_lls[i_rep, iT2] = tmp_fit_rice_lls

        # Fit wLLS on Data
        tmp_fit_rice_wlls = wLLS(tmp_data_rice, TEs, T2_levels[iT2])
        fit_rice_wlls[i_rep, iT2] = tmp_fit_rice_wlls

        # MLE Fit with Fix T2 on Gaussian Data
        bnds = [(lower_bound, 2*tmp_data_rice.sum())] # Force results to be positive
        args = [TEs, tmp_data_rice, sigma, T2_levels[iT2]]
        if tmp_fit_rice_lls < tmp_data_rice.sum():
            guess = [tmp_fit_rice_lls]
        else:
            guess = [tmp_data_rice.mean()]

        # Sometimes, minimize wil not converge. To prevent crash of the script, skip errorous runs. 
        try:
            param_MLE_rice_T2fix = minimize(LL_exp_rice_fixT2, x0 = guess, args = args, bounds = bnds)
        except RuntimeError:
            print("Minimize did not converge - setting initial guess as fit for MLE with fix T2*")
            param_MLE_rice_T2fix.x[0] = tmp_fit_gauss_mle
        tmp_fit_rice_mle = param_MLE_rice_T2fix.x[0]

        fit_rice_mle[i_rep, iT2] = tmp_fit_rice_mle


    Gain_LLS.append(G_SNR(T2_levels[iT2], TEs))

    print('T2* = {:.2f} / {}'.format(T2_levels[iT2], T2max))


# Get rid of all NaN values
fit_gauss_lls[np.isnan(fit_gauss_lls)] = 0
fit_gauss_wlls[np.isnan(fit_gauss_wlls)] = 0
fit_gauss_mle[np.isnan(fit_gauss_mle)] = 0
fit_rice_lls[np.isnan(fit_rice_lls)] = 0
fit_rice_wlls[np.isnan(fit_rice_wlls)] = 0
fit_rice_mle[np.isnan(fit_rice_mle)] = 0

# Calculate Mean across fitted values
sigma_gauss_lls = fit_gauss_lls.std(axis = 0)
sigma_gauss_wlls = fit_gauss_wlls.std(axis = 0)
sigma_gauss_mle = fit_gauss_mle.std(axis = 0)

sigma_rice_lls = fit_rice_lls.std(axis = 0)
sigma_rice_wlls = fit_rice_wlls.std(axis = 0)
sigma_rice_mle = fit_rice_mle.std(axis = 0)


###############################
# %%Simulate Reconstruction Bias for LLS, wLLS and MLE using Monte Carlo for different SNR values

S0 = 1
n_acquisitions = 3 # In our experiment, we acquired 3 repetitions which were concatenated for the fit. This parameter replicates this experimental setting
TEs_rep = np.repeat(TEs, n_acquisitions)
T2 = 30
SNR_min = 1
SNR_max = 100

# Monte Carlo and Fitting Settings
reps = 1000 # Number of repetitions
lower_bound = 5e-4

sigma_min = S0 / SNR_max
sigma_max = S0 / SNR_min
sigma_steps = 100
sigma_levels = np.linspace(sigma_min, sigma_max, sigma_steps)

# Initialize Matrix for fit values
fit_gauss_lls_sigmavar = np.zeros((reps, sigma_steps))
fit_gauss_wlls_sigmavar = np.zeros((reps, sigma_steps))
fit_gauss_mle_sigmavar = np.zeros((reps, sigma_steps))

fit_rice_lls_sigmavar = np.zeros((reps, sigma_steps))
fit_rice_wlls_sigmavar = np.zeros((reps, sigma_steps))
fit_rice_mle_sigmavar = np.zeros((reps, sigma_steps))

print('Simulate Reconstruction Bias for LLS and MLE for T2*={:.1f} at different SNR values'.format(T2))

for i_sigma in range(len(sigma_levels)):
    for i_rep in range(reps):
        # Generate Synthetic data with Gaussian distribution
        tmp_data = T2star_decay_noise(TEs_rep, S0, T2, sigma_levels[i_sigma])
        tmp_data_gauss = np.real(tmp_data)
        tmp_data_rice = np.abs(tmp_data)

        ####################
        ### Real Valued Data

        # Fit LLS on Data
        tmp_fit_gauss_lls = LLS(tmp_data_gauss, TEs_rep, T2)
        fit_gauss_lls_sigmavar[i_rep, i_sigma] = tmp_fit_gauss_lls

        # Fit wLLS on Data
        tmp_fit_gauss_wlls = wLLS(tmp_data_gauss, TEs_rep, T2)
        fit_gauss_wlls_sigmavar[i_rep, i_sigma] = tmp_fit_gauss_wlls


        # MLE Fit with Fix T2 on Gaussian Data
        bnds = [(lower_bound, 2*np.abs(tmp_data_gauss).sum())] # Force results to be positive
        args = [TEs_rep, tmp_data_gauss, sigma_levels[i_sigma], T2]
        if tmp_fit_gauss_lls < tmp_data_gauss.sum():
            guess = [tmp_fit_gauss_lls]
        else:
            guess = [tmp_data_gauss.mean()]

        # If minimize does converge, errorous runs will be skipped. 
        try:
            param_MLE_gauss_T2fix = minimize(LL_exp_gauss_fixT2, x0 = guess, args = args, bounds = bnds)
        except RuntimeError:
            print("Minimize did not converge - setting initial guess as fit for MLE with fix T2*")
            param_MLE_gauss_T2fix.x[0] = tmp_fit_gauss_lls
        tmp_fit_gauss_mle = param_MLE_gauss_T2fix.x[0]

        fit_gauss_mle_sigmavar[i_rep, i_sigma] = tmp_fit_gauss_mle



        ####################
        ### Mag Valued Data

        # Fit LLS on Data
        tmp_fit_rice_lls = LLS(tmp_data_rice, TEs_rep, T2)
        fit_rice_lls_sigmavar[i_rep, i_sigma] = tmp_fit_rice_lls

        # Fit wLLS on Data
        tmp_fit_rice_wlls = wLLS(tmp_data_rice, TEs_rep, T2)
        fit_rice_wlls_sigmavar[i_rep, i_sigma] = tmp_fit_rice_wlls

        # MLE Fit with Fix T2 on Gaussian Data
        bnds = [(lower_bound, 2*tmp_data_rice.sum())] # Force results to be positive
        args = [TEs_rep, tmp_data_rice, sigma_levels[i_sigma], T2]
        if tmp_fit_rice_lls < tmp_data_rice.sum():
            guess = [tmp_fit_rice_lls]
        else:
            guess = [tmp_data_rice.mean()]

        # If minimize does converge, errorous runs will be skipped. 
        try:
            param_MLE_rice_T2fix = minimize(LL_exp_rice_fixT2, x0 = guess, args = args, bounds = bnds)
        except RuntimeError:
            print("Minimize did not converge - setting initial guess as fit for MLE with fix T2*")
            param_MLE_rice_T2fix.x[0] = tmp_fit_gauss_mle
        tmp_fit_rice_mle = param_MLE_rice_T2fix.x[0]

        fit_rice_mle_sigmavar[i_rep, i_sigma] = tmp_fit_rice_mle

    print('SNR = {:.2f} / {}'.format(S0/sigma_levels[i_sigma], S0/sigma_max))


# Get rid of all NaN values
fit_gauss_lls_sigmavar[np.isnan(fit_gauss_lls_sigmavar)]   = 0
fit_gauss_wlls_sigmavar[np.isnan(fit_gauss_wlls_sigmavar)] = 0
fit_gauss_mle_sigmavar[np.isnan(fit_gauss_mle_sigmavar)]   = 0
fit_rice_lls_sigmavar[np.isnan(fit_rice_lls_sigmavar)]     = 0
fit_rice_wlls_sigmavar[np.isnan(fit_rice_wlls_sigmavar)]   = 0
fit_rice_mle_sigmavar[np.isnan(fit_rice_mle_sigmavar)]     = 0

# Calculate average values across fits
avg_gauss_lls_sigmavar  = fit_gauss_lls_sigmavar.mean(axis = 0)
avg_gauss_wlls_sigmavar = fit_gauss_wlls_sigmavar.mean(axis = 0)
avg_gauss_mle_sigmavar  = fit_gauss_mle_sigmavar.mean(axis = 0)

avg_rice_lls_sigmavar = fit_rice_lls_sigmavar.mean(axis = 0)
avg_rice_wlls_sigmavar = fit_rice_wlls_sigmavar.mean(axis = 0)
avg_rice_mle_sigmavar = fit_rice_mle_sigmavar.mean(axis = 0)

# Put Fit Results in Pandas Dataframe for visualization with Seaborn
fit_results_gauss = np.concatenate((fit_gauss_lls_sigmavar.T[..., None], fit_gauss_wlls_sigmavar.T[..., None], fit_gauss_mle_sigmavar.T[..., None]), axis = 2)
fit_results_rice = np.concatenate((fit_rice_lls_sigmavar.T[..., None], fit_rice_wlls_sigmavar.T[..., None], fit_rice_mle_sigmavar.T[..., None]), axis = 2)

# SNR level array for dataframe
SNR_level = np.zeros(fit_results_gauss.shape)
SNR_level[:, ...] = S0/sigma_levels[:, None, None]

# Reconstruction Type array for dataframe
Reco_type = np.zeros(fit_results_gauss[:,...].shape, dtype = object)
Reco_type[...,:] = np.array(["LLS", 'wLLS', 'MLE'], dtype = object)

dict_result_gauss = {'Reconstructed Signal' : fit_results_gauss.ravel(), 'SNR': SNR_level.ravel(), 'Reconstruction Type': Reco_type.ravel()}
df_results_gauss = pd.DataFrame(data = dict_result_gauss)
dict_result_rice = {'Reconstructed Signal' : fit_results_rice.ravel(), 'SNR': SNR_level.ravel(), 'Reconstruction Type': Reco_type.ravel()}
df_results_rice = pd.DataFrame(data = dict_result_rice)


# Remove the weighted LLS results

df_results_gauss = df_results_gauss[df_results_gauss['Reconstruction Type'] != 'wLLS']
df_results_rice = df_results_rice[df_results_rice['Reconstruction Type'] != 'wLLS']
df_results_rice = df_results_rice[df_results_rice['Reconstructed Signal'] > lower_bound]



#############################
# %% Make Joint Plot of results

thickness = 2.
plt.clf()
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)


# Pick Colors according to Seaborn palette
MLE_color = sns.color_palette("deep")[0]
LLS_color = sns.color_palette("deep")[1]
Analytical_color = sns.color_palette("deep")[2]


# T2 Dependent SNR Gain for Gaussian Data
plt.subplot(221)
plt.title('SNR Gain Gaussian Data')
plt.plot(T2_levels, sigma/sigma_gauss_mle, label = 'MLE', color = MLE_color, linewidth = thickness)
plt.plot(T2_levels, sigma/sigma_gauss_lls, label = 'LLS', color = LLS_color, linewidth = thickness)
plt.plot(T2_levels, np.array(Gain_LLS), label = 'Analytical Solution LLS', linestyle = 'dotted', color = Analytical_color, linewidth = thickness)
plt.hlines(np.sqrt(len(TEs)), T2min, T2max, linewidth=thickness, linestyle = 'dashed')
plt.hlines(1, T2min, T2max, linewidth=thickness, linestyle = 'dotted')
plt.grid(which='minor', linestyle='dotted', linewidth=1.5)
plt.legend()
plt.xlabel('$T_{2}*$[ms]')
plt.ylabel('$G_{SNR}$')

# T2 Dependent SNR Gain for Rician Data
plt.subplot(222)
plt.title('SNR Gain Rician Data')
plt.plot(T2_levels, sigma/sigma_rice_mle, label = 'MLE', color = MLE_color, linewidth = thickness)
plt.plot(T2_levels, np.array(Gain_LLS), label = 'Analytical Solution LLS', linestyle = 'dotted', color = 'green', linewidth = thickness)
plt.hlines(np.sqrt(len(TEs)), T2min, T2max, linewidth=thickness, linestyle = 'dashed')
plt.hlines(1, T2min, T2max, linestyle = 'dotted', linewidth=thickness)
plt.grid(which='minor', linestyle='dotted', linewidth=1.5)
plt.xlabel('$T_{2}*$[ms]')
plt.ylabel('$G_{SNR}$')

# Gaussian Data SNR Dependent Bias
plt.subplot(223)
plt.title('Reconstruction Bias Gaussian Data')
sns.lineplot(x='SNR', y="Reconstructed Signal", hue="Reconstruction Type", style = "Reconstruction Type",  linewidth = thickness, data= df_results_gauss, ci = 'sd', zorder = 1)
plt.hlines(S0, S0/sigma_max, S0/sigma_min, linestyle = 'dotted',  linewidth = thickness, label = 'Ground Truth S0', zorder = 2) 
plt.ylim(0, 2.5)
plt.xscale('log')
plt.grid(which='minor', linestyle='dotted', linewidth=1.5)

# Rician Data SNR Dependent Bias
plt.subplot(224)
plt.title('Reconstruction Bias Rician Data')
sns.lineplot(x="SNR", y="Reconstructed Signal", hue="Reconstruction Type", style = "Reconstruction Type",  linewidth = 3.0, data= df_results_rice, ci = 'sd', zorder = 1)
plt.hlines(S0, S0/sigma_max, S0/sigma_min, linestyle = 'dotted',  linewidth = thickness, label = 'Ground Truth S0', zorder = 2) 
plt.ylim(0, 2.5)
plt.xscale('log')
plt.grid(which='minor', linestyle='dotted', linewidth=1.5)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=.5)
plt.show()




