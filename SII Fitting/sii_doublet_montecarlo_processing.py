import csv
from pathlib import Path
from sii_doublet_montecarlo_class import SIIMonteCarlo

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
    line_fitter = SIIMonteCarlo(str(file_path))
    
    results = line_fitter.monte_carlo()

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