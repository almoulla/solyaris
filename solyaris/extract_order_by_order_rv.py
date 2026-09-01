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

    # Empty arrays
    vrad_val_ord = np.empty((Nfile, Norder+1), dtype=float)*np.nan
    vrad_err_ord = np.empty((Nfile, Norder+1), dtype=float)*np.nan

    # Loop files
    print('Extracting order-by-order RV ...')
    for i in tqdm(range(Nfile)):

        # Extract order-by-order CCF
        vgrid, ccf_val, ccf_err = extract_ccf(files[i], instrument)

        # Extract order-by-order RV
        for j in range(Norder+1):
            iccf = iCCF.Indicators(vgrid, ccf_val[j], ccf_err[j])
            vrad_val_ord[i,j] = iccf.RV
            vrad_err_ord[i,j] = iccf.RVerror

    return vrad_val_ord, vrad_err_ord