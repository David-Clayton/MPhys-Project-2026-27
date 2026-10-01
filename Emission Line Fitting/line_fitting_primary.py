import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from scipy.optimize import least_squares
import re
from pathlib import Path

class LineFitting:

    """This class will read in a redshift of a target and the target flux from a certain disperser
    (either G140M, G235M, G395M) and will output the fitted spectrum for the [SII]6716,31 doublet
    along with a continuum fitted to a linear function of the form a*wavelength + b."""

    def __init__(self, file):

        filename = Path(file).name
        match = re.search(r"spec1d_(\d+)_", filename)

        if match is None:
            raise ValueError(f"Can't extract target ID from {filename}")
        
        self.target_id = int(match.group(1))
        catalogue = Table.read(r"C:\Users\drcla\OneDrive\MPhys Project\Emission Line Fitting\excels_cat_internal_v2_0p5asec_120424_miri.fits")
        self.redshift = catalogue[catalogue["targetid"] == self.target_id]["z_excels_v1"][0]

        flux_data = np.loadtxt(file)
        self.wavelengths = flux_data[:,0]
        self.flux = flux_data[:,1]
        self.flux_err = flux_data[:,2]

    def gaussian(self, amplitude, mean, sigma):
        #Gaussian emission profile for the sulphur lines
        return amplitude * np.exp(-(self.wavelengths - mean)**2/(2*sigma**2))

    def continuum(self, cont_lin, cont_const):
        #Linear continuum fit for a narrow-ish band (~100A) around the [SII] doublet
        return cont_lin*self.wavelengths + cont_const 

    def doublet(self, amp6716, amp6731, mean6716, mean6731, sigma, cont_lin, cont_const):
        #Combine the Gaussians for the doublet with the continuum 
        return self.continuum(cont_lin, cont_const) + self.gaussian(amp6716, mean6716, sigma) + self.gaussian(amp6731, mean6731, sigma)

    def residuals(self, theta):
        # Residual function, here "theta" is an array of all the free parameters
        amp6716, amp6731, sigma, cont_lin, cont_const, self.redshift = theta
        model = self.doublet(
            amp6716,
            amp6731,
            6716*(1+self.redshift),
            6731*(1+self.redshift),
            sigma,
            cont_lin, cont_const
            )
        return (self.flux - model)

    def fit_sii_doublet(self):
        # Find the best fit parameters by minimising the residuals^2, so minimising (model-data)^2
        initial = np.array([1, 1, 30, 0.2, 0.2, self.redshift]) # your initial guess
        sii_mask = (self.wavelengths>6600.*(1+self.redshift)) & (self.wavelengths<6845.*(1+self.redshift))
        result = least_squares(self.residuals, initial, args=(self.wavelengths[sii_mask], 1e19*self.flux[sii_mask])) # Finding the best parameters
        best_params = result.x
        return best_params

    