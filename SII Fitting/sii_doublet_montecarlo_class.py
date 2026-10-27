import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.modeling import models,fitting
from scipy.optimize import least_squares
import re
from pathlib import Path

class SIIMonteCarlo:

    """
    This class will calculate the flux of both lines of the [SII]6716,31 doublet along with errors. 
    
    It will read in the redshift and raw spectroscopic data from a target combined with a NIRSpec 
    disperser (G140M, G235M, or G395M), and fit the data to a model with a Levenberg-Marquardt least-
    squared fitting algorithm. The model in question is a pair of Gaussians to model the doublet and
    a linear polynomial to model the continuum around the doublet.

    This class will use a Monte-Carlo to derive errors on the best estimates on the 
    fluxes and redshifts, with ~100-1000 realisations of each spectrum based on the EXCELS data.

    NB: This class should only be used on datafiles where the [SII] doublet has already been found
    and adequately fitted.
    """

    @staticmethod
    def sii_mask(redshift, wavelengths):
            #Cut out the area of the spectrum surrounding the [SII] doublet
            upper_lim = 6850*(1+redshift)
            lower_lim = 6600*(1+redshift)
            mask = (wavelengths < upper_lim) & (wavelengths > lower_lim)
            return mask

    def __init__(self, file):

        filename = Path(file).name
        match = re.search(r"spec1d_(\d+)_(g\d+m)_", filename)

        if match is None:
            raise ValueError(f"Can't extract target ID and disperser from {filename}")
        
        self.target_id = int(match.group(1))
        self.disperser = match.group(2)
        catalogue = Table.read(r"C:\Users\drcla\OneDrive\MPhys Project\SII Fitting\excels_cat_internal_v2_0p5asec_120424_miri.fits")
        self.redshift = catalogue[catalogue["targetid"] == self.target_id]["z_excels_v1"][0]

        flux_data = np.loadtxt(file)
        self.wavelengths = flux_data[:,0]
        self.flux = 1e20 * flux_data[:,1] #Scale it up to improve fitter's numerical precision
        self.flux_err = 1e20 * flux_data[:,2]

        #Check for presence of doublet in masked area of spectrum
        #A chip gap will return a flux of 0 in the gap
        mask = self.sii_mask(self.redshift, self.wavelengths)

        self.valid = True
        #Conditional for if spectrum is sufficiently within masked doublet range
        #Limits of my doublet mask
        upper_mask_limit = 6850 * (1 + self.redshift)
        lower_mask_limit = 6600 * (1 + self.redshift)
        mask_width = upper_mask_limit - lower_mask_limit

        #Limits of data spectrum
        upper_data_limit = np.max(self.wavelengths)
        lower_data_limit = np.min(self.wavelengths)

        overlap = max(0, min(upper_mask_limit, upper_data_limit) - max(lower_mask_limit, lower_data_limit))
        overlap_ratio = overlap / mask_width
        #Fraction of doublet range in overlap - less than 90% of doublet in spectrum
        if overlap_ratio == 0 :
            print(f"SII doublet is not in range")
            self.valid = False
            return 

        #Reject if more than 10% of the doublet window is out of range
        elif overlap_ratio < 0.9:
            print(f"{overlap_ratio * 100}% of SII doublet window is out of range")
            self.valid = False
            return        

        #Conditional for if doublet overlaps with chip gap
        masked_flux = self.flux[mask]
        #Bad pixels are either in a chip gap, or otherwise have NaN fluxes for whatever reason
        bad_pixels = (masked_flux == 0) | np.isnan(masked_flux)

        #Reject doublet measurement if the chip gap or other issues affects more than 10% of pixels in the mask
        if np.count_nonzero(bad_pixels) > 0.1 * len(masked_flux):
            print(f"Doublet interrupted by chip gap")
            self.valid = False
            return

    def spectrum_model(self, amp6716, amp6731, sigma, redshift, cont_const, cont_lin):
        """Create a theoretical model of the spectrum around the [SII] doublet as
        a pair of Gaussians for the doublet and a linear function to model the 
        continuum.""" 
        continuum = models.Polynomial1D(1, c0 = cont_const, c1 = cont_lin)
        sii6716 = models.Gaussian1D(amplitude = amp6716, mean = 6716 * (1 + redshift), stddev = sigma)
        sii6731 = models.Gaussian1D(amplitude = amp6731, mean = 6731 * (1 + redshift), stddev = sigma) 
        #Fix peaks in place in order to fix redshifts
        #sii6716.mean.fixed = True
        #sii6731.mean.fixed = True
        model = continuum + sii6716 + sii6731
        return model

    def model_fit(self):
        """Use Astropy Lev-Mar Least Squares fitter to fit the model to the data"""
        if not self.valid:
            return
    
        mask = self.sii_mask(self.redshift, self.wavelengths)
        fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
        #Initial guess of fit parameters
        initial_guess = self.spectrum_model(np.max(self.flux[mask]), np.max(self.flux[mask]), 5 * (1 + self.redshift), self.redshift, np.median(self.flux[mask]) , 0)
        #Fit model to masked data
        fit_to_model = fitter(initial_guess, self.wavelengths[mask], self.flux[mask], maxiter=1000)
        #print(fit_to_model.parameters)
        """
        fit_to_model.parameters is as follows:
        index 0: continuum.c0  c0 = continuum intercept
        index 1: continuum.c1  c1 = continuum slope
        index 2: sii6716.amplitude
        index 3: sii6716.mean
        index 4: sii6716.stddev
        index 5: sii6731.amplitude
        index 6: sii6731.mean
        index 7: sii6731.stddev
        """
        #print((fit_to_model.parameters[3] / 6716) - 1)
        #print((fit_to_model.parameters[6] / 6731) - 1) #Fit's best estimate of redshifts
        #print(fitter.fit_info["param_cov"])
        return fit_to_model, fitter

    def monte_carlo(self, no_realisations = 1000):
        """Run a Monte Carlo on however many realisations of the masked spectrum"""

        mask = self.sii_mask(self.redshift, self.wavelengths)
        wavelengths = self.wavelengths[mask]
        flux = self.flux[mask]
        flux_err = self.flux_err[mask]

        flux6716_gauss_dist = np.zeros(no_realisations)
        flux6731_gauss_dist = np.zeros(no_realisations)
        redshift6716_gauss_dist = np.zeros(no_realisations)
        redshift6731_gauss_dist = np.zeros(no_realisations)

        #Initial guess of model for all N realisations
        model_guess, fitter = self.model_fit()

        #Run MC
        for i in range(no_realisations):
            realisation = np.random.normal(loc = flux, scale = flux_err)

            initial_guess = model_guess.copy()
            fit_to_model = fitter(initial_guess, wavelengths, realisation, weights = 1 / flux_err, maxiter=1000)#

            amp6716, mean6716, sigma6716, amp6731, mean6731, sigma6731 = fit_to_model.parameters[2:]

            #Add to data arrays

            flux6716_gauss_dist[i] = amp6716 * sigma6716 * np.sqrt(2 * np.pi)
            flux6731_gauss_dist[i] = amp6731 * sigma6731 * np.sqrt(2 * np.pi)

            redshift6716_gauss_dist[i] = mean6716 / 6716 - 1
            redshift6731_gauss_dist[i] = mean6731 / 6731 - 1


        #Take medians and errors from distributions
        flux6716_median = np.nanmedian(flux6716_gauss_dist)
        flux6716_error_upper = np.nanpercentile(flux6716_gauss_dist, 84) - flux6716_median
        flux6716_error_lower = flux6716_median - np.nanpercentile(flux6716_gauss_dist, 16)

        flux6716 = np.nanmedian(flux6716_gauss_dist)
        flux6716_error = (flux6716_error_upper + flux6716_error_lower) / 2

        flux6731_median = np.nanmedian(flux6731_gauss_dist)
        flux6731_error_upper = np.nanpercentile(flux6731_gauss_dist, 84) - flux6731_median
        flux6731_error_lower = flux6731_median - np.nanpercentile(flux6731_gauss_dist, 16)
        
        flux6731 = np.nanmedian(flux6731_gauss_dist)
        flux6731_error = (flux6731_error_upper + flux6731_error_lower) / 2

        redshift6716_median = np.nanmedian(redshift6716_gauss_dist)
        redshift6716_error_upper = np.nanpercentile(redshift6716_gauss_dist, 84) - redshift6716_median
        redshift6716_error_lower = redshift6716_median - np.nanpercentile(redshift6716_gauss_dist, 16)
        
        redshift6716 = np.nanmedian(redshift6716_gauss_dist)
        redshift6716_error = (redshift6716_error_upper + redshift6716_error_lower) / 2
        
        redshift6731_median = np.nanmedian(redshift6731_gauss_dist)
        redshift6731_error_upper = np.nanpercentile(redshift6731_gauss_dist, 84) - redshift6731_median
        redshift6731_error_lower = redshift6731_median - np.nanpercentile(redshift6731_gauss_dist, 16)
                
        redshift6731 = np.nanmedian(redshift6731_gauss_dist)
        redshift6731_error = (redshift6731_error_upper + redshift6731_error_lower) / 2

        return (flux6716, flux6716_error, flux6731, flux6731_error,
                redshift6716, redshift6716_error, redshift6731, redshift6731_error)





        

        


