import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

"""Plots a histogram of the difference between the EXCELS calculated redshift
and my own redshifts calculated from the 6716 and 6731A lines"""

catalogue = Table.read(r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\excels_cat_internal_v2_0p5asec_120424_miri.fits")
redshift_excels = np.asarray(catalogue["z_excels_v1"])

#Homegrown redshifts
sii_fluxes_filepath = r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\sii_fluxes.csv"

# Convert the FITS table to a pandas DataFrame
catalogue_df = catalogue.to_pandas()

# Read the complete CSV
sii_df = pd.read_csv(sii_fluxes_filepath)

# Match rows using the target ID
comparison = sii_df.merge(
    catalogue_df[["targetid", "z_excels_v1"]],
    left_on="target_id",
    right_on="targetid",
    how="inner",
)
# Calculate matched redshift differences
difference_6716 = (comparison["z_excels_v1"] - comparison["redshift6716"])
difference_6731 = (comparison["z_excels_v1"] - comparison["redshift6731"])

# Remove invalid values
difference_6716 = difference_6716[np.isfinite(difference_6716)]
difference_6731 = difference_6731[np.isfinite(difference_6731)]

#Plot

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist((difference_6716), bins = 10)
ax1.set_xlabel(r"$z_{EXCELS}$ - $z_{6716}$") 

ax2.hist(difference_6731, bins = 10)
ax2.set_xlabel(r"$z_{EXCELS}$ - $z_{6731}$") 

plt.savefig("redshift_histogram.png")
plt.show()
