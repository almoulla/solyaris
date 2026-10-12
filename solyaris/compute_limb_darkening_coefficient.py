from __future__ import annotations
import numpy as np
from exotic_ld import StellarLimbDarkening
import subprocess

from typing import Literal

from .get_wavelength_bounds import get_wavelength_bounds

def compute_limb_darkening_coefficient(instrument:Literal['espresso', 'expres', 'harps', 'harps-n', 'kpf', 'neid', 'nirps'], mode:Literal['all', 'one']='all', ref_order:int=None) -> np.array[float]:
    """Compute limb darkening coefficient.

    Parameters
    ----------
    instrument : Literal['espresso', 'expres', 'harps', 'harps-n', 'kpf', 'neid', 'nirps']
        Instrument name.
    mode : Literal['all', 'one'], optional
        Whether to process all orders ('all') or only one reference order ('one'), specified by the ref_order keyword. By default 'all'.
    ref_order : int, optional
        Reference order index (starting from 1) used if mode is 'one'. By default None.

    Returns
    -------
    np.array[float]
        Limb darkening coefficient(s) for either all orders (if mode is 'all') or only one reference order (if mode is 'one').
    """

    # Solar properties
    M_H  = 0    # metallicity           [dex]
    Teff = 5777 # effective temperature [K]
    logg = 4.4  # surface gravity       [cgs]

    # Stellar limb darkening
    sld = StellarLimbDarkening(
    M_H = M_H,
    Teff = Teff,
    logg = logg,
    ld_model = 'mps1',
    ld_data_path = 'temp_data',
    verbose = 0,
    )

    # Order-wise wavelength bounds
    wave_bounds = get_wavelength_bounds(instrument)

    # Nr. of orders in total
    Norder = wave_bounds.shape[0]

    # Orders for mode with all orders
    if mode == 'all': orders = np.arange(1,Norder+1)

    # Orders for mode with one order
    if mode == 'one': orders = np.array([ref_order])

    # Nr. of orders to iterate
    Norder = len(orders)

    # NaN array
    ldc = np.empty(Norder, dtype=float)*np.nan

    # Loop orders
    for i, order in enumerate(orders):

        # Lower and upper wavelengths
        wave_lower = wave_bounds[order-1,0]
        wave_upper = wave_bounds[order-1,1]

        # Check that wavelengths are valid
        if (wave_lower > 0) & (wave_upper > 0):

            # Custom wavelengths and throughput
            custom_wavelengths = np.linspace(wave_lower, wave_upper, 1000)
            custom_throughput  = np.ones_like(custom_wavelengths)

            # Compute linear limb darkening coefficient
            ldc[i] = sld.compute_linear_ld_coeffs(
                wavelength_range = [wave_lower, wave_upper],
                mode = 'custom',
                custom_wavelengths = custom_wavelengths,
                custom_throughput = custom_throughput,
            )[0]

    # Delete downloaded data
    subprocess.check_call(['rm', '-r', 'temp_data'])

    # For ESPRESSO, enforce that limb darkening coefficients are the same for duplicate orders
    if (instrument == 'espresso') & (mode == 'all'):
        ldc = np.repeat(ldc[1::2], 2)

    return ldc