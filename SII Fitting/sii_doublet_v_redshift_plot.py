import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filepath = r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\sii_fluxes_mc.csv"

flux6716 = pd.read_csv(filepath, delimiter = ",", usecols = [2], header = 0).to_numpy().flatten()
flux6716_err = pd.read_csv(filepath, delimiter = ",", usecols = [3], header = 0).to_numpy().flatten()
flux6731 = pd.read_csv(filepath, delimiter = ",", usecols = [4], header = 0).to_numpy().flatten()
flux6731_err = pd.read_csv(filepath, delimiter = ",", usecols = [5], header = 0).to_numpy().flatten()

redshift6716 = pd.read_csv(filepath, delimiter = ",", usecols = [6], header = 0).to_numpy().flatten()
redshift6716_err = pd.read_csv(filepath, delimiter = ",", usecols = [7], header = 0).to_numpy().flatten()
redshift6731 = pd.read_csv(filepath, delimiter = ",", usecols = [8], header = 0).to_numpy().flatten()
redshift6731_err = pd.read_csv(filepath, delimiter = ",", usecols = [9], header = 0).to_numpy().flatten()

emission_line_ratio = flux6716 / flux6731
#Propagate errors
emission_line_ratio_error = emission_line_ratio * np.sqrt((flux6716_err/flux6716)**2 + (flux6731_err/flux6731)**2)

plt.scatter(redshift6716, emission_line_ratio, s = 20, c = "purple", ls = "")
plt.errorbar(redshift6716, emission_line_ratio, yerr = emission_line_ratio_error,
             xerr = redshift6716_err, ecolor = "purple", color = "purple", fmt = "none")
plt.xlabel(r"$z_{6716}$")
plt.ylabel(r"[SII]$\lambda$6716/$\lambda$6731")
plt.savefig(f"doublet_ratio_v_redshift")
plt.show()

plt.scatter(redshift6731, emission_line_ratio, s = 20, c = "darkblue", ls = "")
plt.errorbar(redshift6731, emission_line_ratio, yerr = emission_line_ratio_error,
             xerr = redshift6731_err, ecolor = "darkblue", color = "darkblue", fmt = "none")
plt.xlabel(r"$z_{6731}$")
plt.ylabel(r"[SII]$\lambda$6716/$\lambda$6731")
plt.savefig(f"otherone")
plt.show()


