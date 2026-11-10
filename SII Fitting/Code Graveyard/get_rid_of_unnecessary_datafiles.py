from pathlib import Path
import pandas as pd
import re

"""This script takes all the .txt files from the folder spec1d_fluxcal and recovers only
the files where the [SII] doublet was fitted properly and returned to sii_fluxes.csv. In
order to calculate uncertainties on the flux, the a Monte Carlo will be performed with each
raw spectrum"""

output_folder = Path(r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\spec1d_fluxcal_doublets_found")

#File with valid target IDs
fluxes_file = Path(r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\sii_fluxes.csv")

results = pd.read_csv(fluxes_file)

valid_files = set(
    zip(
        results["target_id"].astype(int),
        results["disperser"].str.lower(),
    )
)

pattern = re.compile(r"spec1d_(\d+)_(g\d+m)_final\.txt$")

for file_path in output_folder.glob("spec1d_*.txt"):
    match = pattern.match(file_path.name)

    if match is None:
        print(f"Skipped: {file_path.name}")
        continue

    target_id = int(match.group(1))
    disperser = match.group(2).lower()

    if (target_id, disperser) not in valid_files:
        print(f"Deleting: {file_path.name}")
        file_path.unlink()
        
print("Finished.")