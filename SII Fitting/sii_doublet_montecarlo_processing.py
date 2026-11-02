import csv
from pathlib import Path
from sii_doublet_montecarlo_class import SIILineFitting

"""Windows filepaths"""
# Create directories
#input_dir = Path(r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\spec1d_fluxcal_doublets")

"""Linux filepaths"""
sii_fitting_dir = Path(__file__).resolve().parent
#__file__ is a special Python variable containing the path of the currently running Python script. 
#Parent means the directory above
input_dir = sii_fitting_dir / "spec1d_fluxcal"
plot_dir = sii_fitting_dir / "Plots"
res_plot_dir = sii_fitting_dir / "Residuals"
csv_dir = sii_fitting_dir

#Empty lists
flux_rows = []

#Headers for csv
flux_header = [
    "target_id",
    "disperser",
    "flux6716",
    "flux6716_err",
    "flux6731",
    "flux6731_err",
    "redshift6716",
    "redshift6716_err",
    "redshift6731",
    "redshift6731_err",
    "(z31-z16)/z16_err",
    "(z31-z16)/z31_err",
]
files = sorted(input_dir.glob("spec1d_*.txt"))

for index, file_path in enumerate(sorted(input_dir.glob("spec1d_*.txt"))):
    line_fitter = SIILineFitting(str(file_path))

    if line_fitter.valid:
        plot_path = plot_dir / f"{file_path.stem}.png"
        residual_plot_path = res_plot_dir / f"{file_path.stem}.png"
        line_fitter.plot_fit(str(plot_path), str(residual_plot_path))
    
    results = line_fitter.monte_carlo()

    if results is None:
        print(f"Fit for {line_fitter.target_id}/{line_fitter.disperser} is too poor.")
        print(f"File no. {index + 1} out of {len(files)} processed.")
        continue

    flux_rows.append([
        line_fitter.target_id,
        line_fitter.disperser,
        *results,
    ])

    print(f"File no. {index + 1} out of {len(files)} processed.")

with open("sii_fluxes_mc.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(flux_header)
    writer.writerows(flux_rows)