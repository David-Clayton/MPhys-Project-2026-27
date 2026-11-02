import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

#Windows filepath
#filepath = r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\sii_fluxes_mc.csv"

"""Linux filepaths"""
sii_fitting_dir = Path(__file__).resolve().parent
filepath = sii_fitting_dir / "sii_fluxes_mc.csv"

flux6716 = pd.read_csv(filepath, delimiter = ",", usecols = [2], header = 0).to_numpy().flatten()
flux6716_err = pd.read_csv(filepath, delimiter = ",", usecols = [3], header = 0).to_numpy().flatten()
flux6731 = pd.read_csv(filepath, delimiter = ",", usecols = [4], header = 0).to_numpy().flatten()
flux6731_err = pd.read_csv(filepath, delimiter = ",", usecols = [5], header = 0).to_numpy().flatten()

redshift6716 = pd.read_csv(filepath, delimiter = ",", usecols = [6], header = 0).to_numpy().flatten()
redshift6716_err = pd.read_csv(filepath, delimiter = ",", usecols = [7], header = 0).to_numpy().flatten()
redshift6731 = pd.read_csv(filepath, delimiter = ",", usecols = [8], header = 0).to_numpy().flatten()
redshift6731_err = pd.read_csv(filepath, delimiter = ",", usecols = [9], header = 0).to_numpy().flatten()

#Only include data points where the SNR for at least one of the lines is greater than 3
#Copilot copypaste start
snr6716 = flux6716 / flux6716_err
snr6731 = flux6731 / flux6731_err

valid = (snr6716 > 3) | (snr6731 > 3)

flux6716 = flux6716[valid]
flux6716_err = flux6716_err[valid]
flux6731 = flux6731[valid]
flux6731_err = flux6731_err[valid]

redshift6716 = redshift6716[valid]
redshift6716_err = redshift6716_err[valid]
redshift6731 = redshift6731[valid]
redshift6731_err = redshift6731_err[valid]
#Copilot copypaste end

emission_line_ratio = flux6716 / flux6731
#Propagate errors
emission_line_ratio_error = emission_line_ratio * np.sqrt((flux6716_err/flux6716)**2 + (flux6731_err/flux6731)**2)

plt.scatter(redshift6716, emission_line_ratio, s = 20, c = "purple")
plt.errorbar(redshift6716, emission_line_ratio, yerr = emission_line_ratio_error,
             xerr = redshift6716_err, ecolor = "purple", color = "purple", fmt = "none")
plt.xlabel(r"$z_{6716}$")
plt.ylabel(r"[SII]$\lambda$6716/$\lambda$6731")
plot_dir = sii_fitting_dir / "doublet_ratio_v_redshift.png"
plt.savefig(plot_dir)
plt.show()


