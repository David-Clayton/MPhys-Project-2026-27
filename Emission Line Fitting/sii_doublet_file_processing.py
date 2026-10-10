import numpy as np
import pandas as pd
import csv
from pathlib import Path
from sii_doublet_fluxes import SIILineFitting

# Create directories
input_dir = Path(
    r"C:\Users\drcla\OneDrive\MPhys Project\Emission Line Fitting\spec1d_fluxcal"
)
plot_dir = Path(
    r"C:\Users\drcla\OneDrive\MPhys Project\Emission Line Fitting\Plots"
)

#This makes the directory if it didn't already exist. If it did exist and exist_ok = True, it would raise
#an error, if it did exist and exist_ok = False, and it was a file, it would raise an error.
#Since it exists and exist_ok = False and it is a directory, it won't raise an error or do anything.
plot_dir.mkdir(exist_ok=True)

flux_rows = []
parameter_rows = []

flux_header = [
    "target_id",
    "disperser",
    "flux6716",
    "flux6716_err",
    "flux6731",
    "flux6731_err",
    "redshift6716",
    "redshift6716_err"
    "redshift6731",
    "redshift6731_err",
]

parameter_header = [
    "target_id",
    "disperser",
    "c0", "c0_err",
    "c1", "c1_err",
    "amp6716", "amp6716_err",
    "mean6716", "mean6716_err",
    "stddev6716", "stddev6716_err",
    "amp6731", "amp6731_err",
    "mean6731", "mean6731_err",
    "stddev6731", "stddev6731_err",
]

for file_path in sorted(input_dir.glob("spec1d_*.txt")):
    try:
        line_fitter = SIILineFitting(str(file_path))

        if line_fitter.valid:
            plot_path = plot_dir / f"{file_path.stem}.png"
            line_fitter.plot_fit(str(plot_path))

        fluxes = line_fitter.calculate_fluxes()
        parameters = line_fitter.return_fit_params()

        flux_rows.append([
            line_fitter.target_id,
            line_fitter.disperser,
            *fluxes,
        ])

        parameter_rows.append([
            line_fitter.target_id,
            line_fitter.disperser,
            *parameters,
        ])

    except Exception as error:
        print(f"Failed to process {file_path.name}: {error}")

with open("sii_fluxes.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(flux_header)
    writer.writerows(flux_rows)

with open("sii_fit_parameters.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(parameter_header)
    writer.writerows(parameter_rows)
