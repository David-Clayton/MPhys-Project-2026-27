import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from pathlib import Path

"""Plots a histogram of the difference between the EXCELS calculated redshift
and my own redshift"""

dir = Path(__file__).resolve().parent
catalogue_path = dir / "excels_cat_internal_v2_0p5asec_120424_miri.fits"
catalogue = Table.read(catalogue_path)

#catalogue = Table.read(r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\excels_cat_internal_v2_0p5asec_120424_miri.fits")
redshift_excels = np.asarray(catalogue["z_excels_v1"])

#Homegrown redshifts
sii_fluxes_filepath = dir / "sii_fluxes.csv"

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
difference_z = (comparison["z_excels_v1"] - comparison["redshift"])

# Remove invalid values
difference_z = difference_z[np.isfinite(difference_z)]

mask = (abs(difference_z) < 0.02)  # Adjust the threshold as needed
difference_z = difference_z[mask]

#Plot

fig, ax = plt.subplots()
ax.hist((difference_z), bins = 100)
ax.set_xlabel(r"$z_{EXCELS}$ - $z_{Mine}$") 

plt.savefig("redshift_histogram.png")
plt.show()

