from   astropy.io import fits
import numpy      as     np
import warnings
warnings.filterwarnings('ignore')

def extract_ccf(file, instrument):

    # ESPRESSO
    if instrument == 'espresso':

        # Load FITS file
        with fits.open(file) as hdul:

            # Header
            header = hdul[0].header

            # Extract CCF data
            ccf_val = hdul[1].data
            ccf_err = hdul[2].data
            vstart  = header['HIERARCH ESO RV START']
            vstep   = header['HIERARCH ESO RV STEP' ]
            vgrid   = vstart + vstep * np.arange(ccf_val.shape[1])

    return vgrid, ccf_val, ccf_err