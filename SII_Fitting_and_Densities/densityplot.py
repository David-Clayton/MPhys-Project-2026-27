import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pyneb as pn

#Windows filepath
#filepath = r"C:\Users\drcla\OneDrive\MPhys Project\Densities\sii_fluxes_mc.csv"

"""Linux filepaths"""
densities_dir = Path(__file__).resolve().parent
filepath = densities_dir / "sii_fluxes.csv"

targetid = pd.read_csv(filepath, delimiter = ",", usecols = [0], header = 0).to_numpy().flatten()   

flux6716 = pd.read_csv(filepath, delimiter = ",", usecols = [2], header = 0).to_numpy().flatten()
flux6716_err = pd.read_csv(filepath, delimiter = ",", usecols = [3], header = 0).to_numpy().flatten()
flux6731 = pd.read_csv(filepath, delimiter = ",", usecols = [4], header = 0).to_numpy().flatten()
flux6731_err = pd.read_csv(filepath, delimiter = ",", usecols = [5], header = 0).to_numpy().flatten()

redshift = pd.read_csv(filepath, delimiter = ",", usecols = [6], header = 0).to_numpy().flatten()
redshift_err = pd.read_csv(filepath, delimiter = ",", usecols = [7], header = 0).to_numpy().flatten()

n_e = pd.read_csv(filepath, delimiter = ",", usecols = [8], header = 0).to_numpy().flatten()
n_e_err_upper = pd.read_csv(filepath, delimiter = ",", usecols = [9], header = 0).to_numpy().flatten()
n_e_err_lower = pd.read_csv(filepath, delimiter = ",", usecols = [10], header = 0).to_numpy().flatten()

#Check length 
print(f"{len(redshift)} initial spectra passing chisq")

#Only include data points where the SNR for at least one of the lines is greater than 3
#Copilot copypaste start
snr6716 = flux6716 / flux6716_err
snr6731 = flux6731 / flux6731_err

valid = (flux6716 > 1e-20) & (flux6731 > 1e-20) & (flux6716_err > 1e-20) & (flux6731_err > 1e-20) & ((snr6716 > 3) | (snr6731 > 3))

targetid = targetid[valid] 

flux6716 = flux6716[valid]
flux6716_err = flux6716_err[valid]
flux6731 = flux6731[valid]
flux6731_err = flux6731_err[valid]

redshift = redshift[valid]
redshift_err = redshift_err[valid]

n_e = n_e[valid]
n_e_err_upper = n_e_err_upper[valid]
n_e_err_lower = n_e_err_lower[valid]

#Check length after SNR cut
print(f"{len(redshift)} spectra passing SNR cut")

plt.scatter(redshift, n_e, s = 10, c = 'red', alpha = 0.5)
plt.errorbar(redshift, n_e, xerr = redshift_err, yerr = [n_e_err_lower, n_e_err_upper], fmt = 'none', ecolor = 'red', alpha = 0.5)
plt.xlabel(r"$z$")
plt.ylabel(r"$n_e$ (cm$^{-3}$)")
plt.yscale("log")
plt.title(rf"$T_e$ = 15000 K (PyNeb limits)")
plt.savefig(densities_dir / f"n_e_vs_z_Te_15000.png", dpi = 300)
plt.show()

#Save density data to CSV
data = np.column_stack((targetid, redshift, redshift_err, n_e, n_e_err_lower, n_e_err_upper))
df = pd.DataFrame(data, columns = ["targetid", "redshift", "redshift_err", "n_e", "n_e_err_lower", "n_e_err_upper"])
df.to_csv(densities_dir / f"n_e_data_Te_15000.csv", index = False)

