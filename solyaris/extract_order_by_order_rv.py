import iCCF
import numpy as np

from .extract_ccf import extract_ccf

def extract_order_by_order_rv(file, instrument, Norder):

    # NaN arrays
    vrad_val = np.empty(Norder+1, dtype=float)*np.nan
    vrad_err = np.empty(Norder+1, dtype=float)*np.nan

    # Extract order-by-order CCF
    vgrid, ccf_val, ccf_err = extract_ccf(file, instrument)

    # Loop orders
    for i in range(Norder+1):

        # Check that CCF is finite
        if not np.all(np.isfinite(ccf_val[i]) & np.isfinite(ccf_err[i])):
            continue

        # Extract order-by-order RV
        iccf = iCCF.Indicators(vgrid, ccf_val[i], ccf_err[i])
        vrad_val[i] = iccf.RV
        vrad_err[i] = iccf.RVerror

    return vrad_val, vrad_err