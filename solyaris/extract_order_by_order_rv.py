import iCCF
import numpy    as     np
from   tqdm     import tqdm
import warnings
warnings.filterwarnings('ignore')

from .extract_ccf import extract_ccf

def extract_order_by_order_rv(files, instrument, Norder):

    # Nr. of files and orders
    Nfile  = len(files)
    Norder = Norder

    # NaN arrays
    vrad_val_ord = np.empty((Nfile, Norder+1), dtype=float)*np.nan
    vrad_err_ord = np.empty((Nfile, Norder+1), dtype=float)*np.nan

    # Loop files
    for i in tqdm(range(Nfile)):

        # Extract order-by-order CCF
        vgrid, ccf_val, ccf_err = extract_ccf(files[i], instrument)

        # Loop orders
        for j in range(Norder+1):

            # Check that CCF is finite
            if np.all(np.isfinite(ccf_val[j]) & np.isfinite(ccf_err[j])):

                # Extract order-by-order RV
                iccf = iCCF.Indicators(vgrid, ccf_val[j], ccf_err[j])
                vrad_val_ord[i,j] = iccf.RV
                vrad_err_ord[i,j] = iccf.RVerror

    return vrad_val_ord, vrad_err_ord