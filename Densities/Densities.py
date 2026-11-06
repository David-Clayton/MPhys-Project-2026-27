import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pyneb as pn

#Windows filepath
#filepath = r"C:\Users\drcla\OneDrive\MPhys Project\Densities\sii_fluxes_mc.csv"

"""Linux filepaths"""
densities_dir = Path(__file__).resolve().parent
filepath = densities_dir / "sii_fluxes_mc.csv"

targetid = pd.read_csv(filepath, delimiter = ",", usecols = [0], header = 0).to_numpy().flatten()   

flux6716 = pd.read_csv(filepath, delimiter = ",", usecols = [2], header = 0).to_numpy().flatten()
flux6716_err = pd.read_csv(filepath, delimiter = ",", usecols = [3], header = 0).to_numpy().flatten()
flux6731 = pd.read_csv(filepath, delimiter = ",", usecols = [4], header = 0).to_numpy().flatten()
flux6731_err = pd.read_csv(filepath, delimiter = ",", usecols = [5], header = 0).to_numpy().flatten()

redshift = pd.read_csv(filepath, delimiter = ",", usecols = [6], header = 0).to_numpy().flatten()
redshift_err = pd.read_csv(filepath, delimiter = ",", usecols = [7], header = 0).to_numpy().flatten()

#Check length 
print(f"{len(redshift)} initial spectra passing chisq")

#Only include data points where the SNR for at least one of the lines is greater than 3
#Copilot copypaste start
snr6716 = flux6716 / flux6716_err
snr6731 = flux6731 / flux6731_err

valid = (flux6716 > 1e-20) & (flux6731 > 1e-20) & (flux6716_err > 1e-20) & (flux6731_err > 1e-20) & ((snr6716 > 3) | (snr6731 > 3)) & (redshift_err < 700)

targetid = targetid[valid]

flux6716 = flux6716[valid]
flux6716_err = flux6716_err[valid]
flux6731 = flux6731[valid]
flux6731_err = flux6731_err[valid]

redshift = redshift[valid]
redshift_err = redshift_err[valid]

#Check length after SNR cut
print(f"{len(redshift)} spectra passing SNR cut")

emission_line_ratio = flux6716 / flux6731
#Propagate errors
emission_line_ratio_error = emission_line_ratio * np.sqrt((flux6716_err/flux6716)**2 + (flux6731_err/flux6731)**2)

#arrays to store density information
n_e = np.zeros(len(emission_line_ratio))
n_e_err_upper = np.zeros(len(emission_line_ratio))
n_e_err_lower = np.zeros(len(emission_line_ratio))

#Atomic class
S2 = pn.Atom('S', 2)

#Temperature assumptions for galaxies
T_e = 15000 #K

#For each target, loop through and Monte Carlo the density calculation

for i in range(len(emission_line_ratio)):

    #Array of realisations distributed as a Gaussian centred on emission_line_ratio[i]
    line_ratio_realisation  = np.random.normal(emission_line_ratio[i], emission_line_ratio_error[i], size = 1000)

    #Turn temperature into an array of the same size as line_ratio_realisation
    T_e_array = np.full(shape = 1000, fill_value = T_e)

    #Calculate the density for each realisation
    n_e_realisation = S2.getTemDen(line_ratio_realisation, tem = T_e_array, wave1 = 6716, wave2 = 6731)

    median_n_e = np.nanmedian(n_e_realisation)
    upper_err_n_e = np.nanpercentile(n_e_realisation, 84) - median_n_e
    lower_err_n_e = median_n_e - np.nanpercentile(n_e_realisation, 16)

    #Store the results in the arrays
    n_e[i] = median_n_e
    n_e_err_upper[i] = upper_err_n_e
    n_e_err_lower[i] = lower_err_n_e

    print(f"File no. {i+1} of {len(emission_line_ratio)} complete")

plt.scatter(redshift, n_e, s = 10, c = 'blue', alpha = 0.5)
plt.errorbar(redshift, n_e, xerr = redshift_err, yerr = [n_e_err_lower, n_e_err_upper], fmt = 'none', ecolor = 'blue', alpha = 0.5)
plt.xlabel(r"$z$")
plt.ylabel(r"$n_e$ (cm$^{-3}$)")
plt.yscale("log")
plt.title(rf"$T_e$ = {T_e} K")
plt.savefig(densities_dir / f"n_e_vs_z_Te_{T_e}.png", dpi = 300)
plt.show()

#Save density data to CSV
data = np.column_stack((targetid, redshift, redshift_err, n_e, n_e_err_lower, n_e_err_upper))
df = pd.DataFrame(data, columns = ["targetid", "redshift", "redshift_err", "n_e", "n_e_err_lower", "n_e_err_upper"])
df.to_csv(densities_dir / f"n_e_data_Te_{T_e}.csv", index = False)

