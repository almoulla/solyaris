import astropy.constants as     ac
from   astropy.time      import Time
import barycorrpy

def compute_berv_and_herv(obsname, date_obs, exp_time, photocen):

    # Unit conversions
    sec_to_day = 1/(60*60*24) # seconds to days
    mps_to_kmps = 1e-3        # m/s to km/s

    # Compute gravitational redshift of the Sun
    gr_sun = (ac.G.value * ac.M_sun.value) / (ac.R_sun.value * ac.c.value)

    # Compute exposure photometric center in JD UTC
    jd_utc_cen = Time(date_obs, format='isot').jd + exp_time * photocen * sec_to_day

    # Compute BERV and HERV
    berv_val = barycorrpy.get_BC_vel(JDUTC=jd_utc_cen, obsname=obsname, SolSystemTarget='Sun'                 )[0][0]
    herv_val = barycorrpy.get_BC_vel(JDUTC=jd_utc_cen, obsname=obsname, SolSystemTarget='Sun', predictive=True)[0][0]*(-1) - berv_val + gr_sun

    # Convert from m/s to km/s
    berv_val *= mps_to_kmps
    herv_val *= mps_to_kmps

    return berv_val, herv_val