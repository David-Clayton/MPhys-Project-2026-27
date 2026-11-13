import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

#Windows filepath
#filepath = r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\sii_fluxes_mc.csv"

"""Linux filepaths"""
sii_fitting_dir = Path(__file__).resolve().parent
filepath = sii_fitting_dir / "sii_fluxes_mc.csv"

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

valid = (flux6716 > 1e-20) & (flux6731 > 1e-20) & (flux6716_err > 1e-20) & (flux6731_err > 1e-20) & ((snr6716 > 5) | (snr6731 > 5))

targetid = targetid[valid]

flux6716 = flux6716[valid]
flux6716_err = flux6716_err[valid]
flux6731 = flux6731[valid]
flux6731_err = flux6731_err[valid]

redshift = redshift[valid]
redshift_err = redshift_err[valid]
#Copilot copypaste end

#Check length after mask
print(f"SNR criterion & friends have left {len(redshift)}")
print(np.sort(redshift_err)[-10:])
print(np.argmax(redshift_err))

print(np.min(flux6716))

emission_line_ratio = flux6716 / flux6731
#Propagate errors
emission_line_ratio_error = emission_line_ratio * np.sqrt((flux6716_err/flux6716)**2 + (flux6731_err/flux6731)**2)

plt.scatter(redshift, emission_line_ratio, s = 20, c = "purple", alpha = 0.5)
plt.errorbar(redshift, emission_line_ratio, yerr = emission_line_ratio_error,
             xerr = redshift_err, ecolor = "purple", color = "purple", fmt = "none")
plt.xlabel(r"$z$")
plt.ylabel(r"[SII]$\lambda$6716/$\lambda$6731")
#plt.title(f"Restricted SNR test")
plt.axhline(1.45, color = "orange", linestyle = "--", label = "theor. upper limit")
plt.axhline(0.44, color = "red", linestyle = "--", label = "theor. lower limit")
plt.legend()
plot_dir = sii_fitting_dir / "doublet_ratio_v_redshift.png"
plt.savefig(plot_dir)
plt.show()


print(np.sort(emission_line_ratio_error)[::-1])
print(targetid[np.argsort(emission_line_ratio_error)[::-1]])
print(targetid[np.argmax(redshift_err)])