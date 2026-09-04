import astropy.constants as     ac
from   astropy.time      import Time
import barycorrpy
import numpy             as     np
from   tqdm              import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_berv_and_herv(obsname, date_obs, exp_time, photocen):

    # Nr. of observations
    Nobs = len(date_obs)

    # Empty arrays
    berv_val = np.empty(Nobs, dtype=float)*np.nan
    herv_val = np.empty(Nobs, dtype=float)*np.nan

    # Loop observations
    for i in tqdm(range(Nobs)):

        # Compute gravitational redshift of the Sun
        gr_sun = (ac.G.value * ac.M_sun.value) / (ac.R_sun.value * ac.c.value)

        # Compute BERV and HERV
        jd_utc_cen  = Time(date_obs[i], format='isot').jd + exp_time[i]*photocen[i]/(60*60*24)
        berv_val[i] = barycorrpy.get_BC_vel(JDUTC=jd_utc_cen, obsname=obsname, SolSystemTarget='Sun'                 )[0][0]
        herv_val[i] = barycorrpy.get_BC_vel(JDUTC=jd_utc_cen, obsname=obsname, SolSystemTarget='Sun', predictive=True)[0][0]*(-1) - berv_val[i] + gr_sun

    # Convert from m/s to km/s
    berv_val *= 1e-3
    herv_val *= 1e-3

    return berv_val, herv_val