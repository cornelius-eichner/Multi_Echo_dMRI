# Simulations for Multi Echo dMRI Acquisitions

The code from this repository was employed for the development of the Multi Echo diffusion MRI acquisition scheme presented in Eichner et al 2020 
(https://doi.org/10.1016/j.neuroimage.2020.117172)

## Increased Sensitivity and Signal-to-Noise Ratio in Diffusion-Weighted MRI using Multi-Echo Acquisitions
*Cornelius Eichner, Michael Paquette, Toralf Mildner, Torsten Schlumm, Kamilla Pleh, Liran Samuni, Catherine Crockford, Roman M. Wittig, Carsten Jäger, Harald E. Möller, Angela D. Friederici, Alfred Anwander*

* *Sim_GSNR_and_Bias.py* performs numerical simulations to generate Figure 2 of the paper. 
* *Sim_T2s_Deviation_Diffusion.py* employs literature values from Kleban et al 2020 to provide an error estimation of diffusion weighting induced changes of T2*
* *Sim_in_vivo_Usability.py* is a tool to provide an estimation how many echo acquisitions help gain the SNR of the final reconstruction. Please note that this script calculates the SNR gain based on the LLS solution presented in the paper (Equation 8 of the paper). The actual SNR gains will be higher if nonlinear minimization is employed.  


## Code for Data Reconstruction will be added soon 
