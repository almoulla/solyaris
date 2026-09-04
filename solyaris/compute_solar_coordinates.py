from   astropy             import units as u
from   astropy.coordinates import AltAz, Angle, EarthLocation, get_body, solar_system_ephemeris
from   astropy.time        import Time
from   datetime            import datetime, timedelta
import numpy               as     np
from   tqdm                import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_solar_coordinates(obsname, date_obs, exp_time, photocen):

    # Nr. of observations
    Nobs = len(date_obs)

    # Empty arrays
    alph_val = np.empty(Nobs, dtype=float)*np.nan
    delt_val = np.empty(Nobs, dtype=float)*np.nan
    airm_val = np.empty(Nobs, dtype=float)*np.nan

    # Loop observations
    for i in tqdm(range(Nobs)):

        # Get date at start, end and photocenter of observation
        date_start = datetime.strptime(date_obs[i], '%Y-%m-%dT%H:%M:%S.%f')
        date_end = date_start + timedelta(days=0, seconds=exp_time[i])
        date_photocenter = date_start + timedelta(days=0, seconds=exp_time[i]*photocen[i])
        
        # Get Sun coordinates
        solar_system_ephemeris.set('de432s') # use JPL ephemeredis
        loc = EarthLocation.of_site(obsname)
        obstime = Time(date_photocenter)
        sun_coord = get_body('sun', obstime, loc)
        sun_coord_string = sun_coord.to_string('hmsdms')
        alpha = sun_coord_string.split('s')[0]
        delta = sun_coord_string.split('s')[1].strip().replace('d',':').replace('m',':').replace('+', '')

        # Convert alpha and delta to decimal degrees
        alph_val[i] = Angle(alpha, unit=u.hour  ).to(u.degree).value
        delt_val[i] = Angle(delta, unit=u.degree)             .value
        
        # Compute airmass
        time_start = Time(date_start)
        time_end = Time(date_end)
        sun_altaz_start = sun_coord.transform_to(AltAz(obstime=time_start, location=loc))
        sun_altaz_end = sun_coord.transform_to(AltAz(obstime=time_end, location=loc))
        airm_start = sun_altaz_start.secz.value
        airm_end = sun_altaz_end.secz.value
        airm_val[i] = airm_start*(1-photocen[i]) + airm_end*photocen[i]

    return alph_val, delt_val, airm_val