from   astropy.time         import Time
import emcee                # version 2.2.1
from   matplotlib.colorbar  import Colorbar
import matplotlib.gridspec  as     gridspec
import matplotlib.pylab     as     pylab
import matplotlib.pyplot    as     plt
import numpy                as     np
import pandas               as     pd
from   tqdm                 import tqdm
import warnings
warnings.simplefilter(action='ignore')

def compute_quality_flag(time_jdb, airm_val, snr_val, mode):

    ### CONFIG

    # Column names of expected variables
    col_jdb      = 'time_jdb' # Julian Date Barycentric (JDB)
    col_airm     = 'airm_val' # airmass
    col_snrx     = 'snr_ref'  # signal-to-noise ratio (SNR)

    # Column names of computed variables
    col_jdn      = 'time_jdn' # Julian Day Number (JDN)    
    col_hang     = 'hang_val' # hour angle
    col_magn     = 'magn_val' # magnitude
    col_magn_err = 'magn_err' # magnitude error

    # Time zone w.r.t. UT (only for plotting the approximate hour angle)
    TZ = 0

    # Toggle True/False
    clip_outlier = True # outlier clipping
    plot_figures = True # plot and save figures

    ### PLOT PARAMS

    # Figure name
    def fig_name(jdn):

        date = Time(jdn, format='jd').isot[:10]

        return f'Fig_{mode}_{date}.pdf'

    # Figure size and fontsizes
    plot_params = {'figure.figsize'        : (20, 10),
                'figure.titlesize'      :  25,
                'axes.titlesize'        :  25,
                'axes.labelsize'        :  20,
                'xtick.labelsize'       :  20,
                'ytick.labelsize'       :  20,
                'legend.title_fontsize' :  15,
                'legend.fontsize'       :  15
                }
    pylab.rcParams.update(plot_params)

    # Colormap
    cmap = 'RdYlGn'

    ### CONSTANTS

    # Discription and bounds of MCMC parameters
    pname   = ['$\hat{c}_{0}$', '$c_{1}$', '$\sigma_{\mathrm{jit}}$', '$\hat{\mu}$', '$\sigma$', '$Q$']
    bounds  = [(-0.4  , 0.4 ),    # c_0 : intercept of foreground line w.r.t. weighted mean airmass
            ( 0.075, 0.4 ),    # c_1 : slope of foreground line
            ( 0.001, 0.08),    # jit : white noise of foreground
            ( 0.0  , 0.8 ),    # mu  : mean of background w.r.t. weighted mean magnitude
            ( 0.1  , 0.7 ),    # sig : std of background
            ( 0.0  , 1.0 )]    # Q   : fraction of foreground population
    ndim    = np.shape(bounds)[0] # nr. of parameters in MCMC
    max_c0  = 0.1                 # allowed max. for c0 w.r.t. weighted mean airmass
                                # (larger values indicate a swap between fore- & background)

    # MCMC walkers and chains
    min_pnt = 10                  # allowed min. nr. of points per day
    nwalker = 32                  # nr. of walkers in MCMC
    nthin_b = 50                  # thinning step for sampling burn-in chains
    nthin_p = 100                 # thinning step for sampling production chain
    npt_end = int(2e3)            # nr. of end points from which to sample burn-ins
    nstep_b = int(1e4)            # nr. of steps in burn-in
    nstep_p = nstep_b             # nr. of steps in production

    ### FUNCTIONS

    # Interquartile clipping.
    # data    : data to be filtered
    # returns : boolean for whether points in <data> are w/in the accepted range
    def iqr_clip(data, low=False):
        Q1, Q3 = np.nanpercentile(data, [25,75])
        IQR    = Q3 - Q1
        if low:
            return (data > Q1-IQR)
        else:
            return (data > Q1-IQR) & (data < Q3+IQR)

    # Sample Gaussian w/ median and MAD from an emcee chain, limited by bounds.
    # chain   : emcee chain
    # returns : list w/ IC for Walkers
    def gauss_bound(chain):
        sample = chain[:,-npt_end::nthin_b,:].reshape(-1,ndim)
        pmed   = np.median(sample, axis=0)
        pmad   = np.median(np.abs(sample-pmed), axis=0)
        p0     = [np.random.normal(pmed,pmad) for _ in range(nwalker)]
        for i in range(nwalker):
            for j in range(ndim):
                if   p0[i][j] < bounds[j][0]:
                        p0[i][j] = bounds[j][0]
                elif p0[i][j] > bounds[j][1]:
                        p0[i][j] = bounds[j][1]
        return p0

    # Likelihood of foreground population.
    # p       : sample parameters
    # x       : data x-coordinates
    # y       : data y-coordinates
    # yerr    : data y-errors
    # returns : Gaussian (ln) probability of belonging to foreground
    def lnlike_fg(p, x, y, yerr):
        *c, jit, _, _, _ = p
        ym  = c[0] + c[1]*x
        var = yerr**2 + jit**2
        return -0.5*((y-ym)**2/var + np.log(var))

    # Likelihood of background population.
    # p       : sample parameters
    # x       : data x-coordinates
    # y       : data y-coordinates
    # yerr    : data y-errors
    # returns : Gaussian (ln) probability of belonging to background
    def lnlike_bg(p, x, y, yerr):
        *_, mu, sig, _ = p
        var = yerr**2 + sig**2
        return -0.5*((y-mu)**2/var + np.log(var))

    # Prior probability.
    # p       : sample parameters
    # bounds  : uniform bounds on <p>
    # returns : probability 1 if w/in bounds else 0 (0 & -INF in natural logarithms)
    def lnprior(p, bounds):
        if not all(b[0] <= v <= b[1] for v, b in zip(p, bounds)):
            return -np.inf
        return 0

    # Posterior probability.
    # p       : sample parameters
    # bounds  : uniform bounds on <p>
    # x       : data x-coordinates
    # y       : data y-coordinates
    # yerr    : data y-errors
    # returns : combined probaility (prior+likelihoods in log scale);
    #           2D-blob for fore- & and background
    def lnprob(p, bounds, x, y, yerr):
        
        # Quality
        *_, Q = p
        
        # Check the prior
        lp = lnprior(p, bounds)
        if not np.isfinite(lp):
            return -np.inf, None
        
        # Compute the vector of foreground likelihoods & include the q prior
        ll_fg = lnlike_fg(p, x, y, yerr)
        if Q == 0.0:
            arg1 = -np.inf
        else:
            arg1 = ll_fg + np.log(Q)
        
        # Compute the vector of background likelihoods & include the q prior
        ll_bg = lnlike_bg(p, x, y, yerr)
        if Q == 1.0:
            arg2 = -np.inf
        else:
            arg2 = ll_bg + np.log(1.0 - Q)
        
        # Combine these using log-add-exp for numerical stability
        ll = np.sum(np.logaddexp(arg1, arg2))
        
        # Use the emcee 'blobs' feature to track fore- & background
        return lp + ll, (arg1, arg2)

    ### DATA

    # Read DataFrame
    df_all = pd.DataFrame()
    df_all[col_jdb ] = time_jdb
    df_all[col_airm] = airm_val
    df_all[col_snrx] = snr_val

    # Variables
    jdb      = df_all[col_jdb ].values
    airm     = df_all[col_airm].values
    snr      = df_all[col_snrx].values

    # Compute new variables
    jdn      = np.floor(jdb+0.5).astype('int')
    hang     = (jdb - jdn + TZ/24)*360
    magn     = -5*np.log10(snr)
    magn_err = 2.5/(np.log(10)*snr)

    # Add needed variables to DataFrame
    df_all[col_jdn     ] = jdn
    df_all[col_hang    ] = hang
    df_all[col_magn    ] = magn
    df_all[col_magn_err] = magn_err

    # Add boolean columns for interquartile clipping on RV and SNR
    df_all.insert(df_all.columns.get_loc(col_snrx)+1, 'in_snr' , np.ones(df_all.shape[0], dtype=bool))

    # Add NaN column for Quality Flag
    df_all['qualflag'] = np.empty(df_all.shape[0], dtype=float)*np.nan

    # Array w/ each unique day
    jdn_arr = np.unique(df_all[col_jdn])
    
    # New DataFrame w/ parameters for each day
    df_day = pd.DataFrame()
    df_day[col_jdn ] = jdn_arr
    df_day['in_day'] = np.zeros(df_day.shape[0], dtype=bool)
    df_day['c0'    ] = np.empty(df_day.shape[0])*np.nan
    df_day['c0_hat'] = np.empty(df_day.shape[0])*np.nan
    df_day['c1'    ] = np.empty(df_day.shape[0])*np.nan
    df_day['jit'   ] = np.empty(df_day.shape[0])*np.nan
    df_day['mu'    ] = np.empty(df_day.shape[0])*np.nan
    df_day['mu_hat'] = np.empty(df_day.shape[0])*np.nan
    df_day['sig'   ] = np.empty(df_day.shape[0])*np.nan
    df_day['Q'     ] = np.empty(df_day.shape[0])*np.nan

    ### QUALFLAG

    # Loop through days
    for jdni in tqdm(jdn_arr):

        # All index
        i_all = df_all[col_jdb][df_all[col_jdn] == jdni].index

        # Day index (skip iteration if day not in data)
        i_day = np.where(jdn_arr == jdni)[0]
        if len(i_day) > 0:
            i_day = i_day[0]
        else:
            continue

        # Outlier clipping
        if clip_outlier:
            df_all.loc[i_all, 'in_snr']  = iqr_clip(df_all[col_snrx][i_all], low=True)
            df_all.loc[i_all, 'in_snr'] *=         (df_all[col_snrx][i_all] > 1      )
            df_all.loc[i_all, 'in_snr'] *=         (df_all[col_airm][i_all] >=1      )

        # Skip MCMC if day has too few points
        xser = df_all[col_airm][(df_all[col_jdn] == jdni) & df_all.in_snr]
        if xser.size < min_pnt:
            continue

        # Data points
        x    = np.array(df_all[col_airm    ][xser.index])
        y    = np.array(df_all[col_magn    ][xser.index])
        yerr = np.array(df_all[col_magn_err][xser.index])
        xhat = np.average(x, weights=1/yerr**2)
        yhat = np.average(y, weights=1/yerr**2)
        
        # Set up the sampler
        sampler = emcee.EnsembleSampler(nwalker, ndim, lnprob, args=(bounds,x-xhat,y-yhat,yerr))
        
        # 1st burn-in chain
        p0 = [[(b[1]+b[0])/2 for b in bounds] + 1e-5*np.random.randn(ndim) for _ in range(nwalker)]
        sampler.run_mcmc(np.array(p0), nstep_b)
        if plot_figures:
            chain_b1 = sampler.chain
            cname_b1 = '1$^{\mathrm{st}}$ burn-in chain'
        
        # 2nd burn-in chain
        p0 = gauss_bound(sampler.chain)
        sampler.reset()
        sampler.run_mcmc(p0, nstep_b)
        if plot_figures:
            chain_b2 = sampler.chain
            cname_b2 = '2$^{\mathrm{nd}}$ burn-in chain'
        
        # Production chain
        p0 = gauss_bound(sampler.chain)
        sampler.reset()
        sampler.run_mcmc(p0, nstep_p)
        if plot_figures:
            chain_p = sampler.chain
            cname_p = 'Production chain'
        
        # Extract MCMC parameters
        para = np.array(sampler.flatchain[::nthin_p,:])
        c0, c1, jit, mu, sig, Q = np.median(para, axis=0).T
        
        # Save results if fore- & background correctly identified
        if c0 < max_c0:
            
            # Flag day as valid
            df_day.loc[i_day, 'in_day'] = True
            
            # Save parameters
            c0_hat = c0
            mu_hat = mu
            c0    += yhat - xhat*c1
            mu    += yhat
            df_day.loc[i_day, 'c0'    ] = c0
            df_day.loc[i_day, 'c0_hat'] = c0_hat
            df_day.loc[i_day, 'c1'    ] = c1
            df_day.loc[i_day, 'jit'   ] = jit
            df_day.loc[i_day, 'mu'    ] = mu
            df_day.loc[i_day, 'mu_hat'] = mu_hat
            df_day.loc[i_day, 'sig'   ] = sig
            df_day.loc[i_day, 'Q'     ] = Q
            
            # Quality (i.e. probability of belonging to foreground)
            norm = 0.0
            qual = np.zeros(x.size)
            for ii in range(sampler.chain.shape[1]):
                for jj in range(sampler.chain.shape[0]):
                    ll_fg, ll_bg = sampler.blobs[ii][jj]
                    qual += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                    norm += 1
            qual /= norm
            
            # Save quality
            df_all.loc[xser.index, 'qualflag'] = qual

    ### PLOT
        
        if plot_figures & df_day.in_day[i_day]:
            
            # FIGURE 2: 2x1 mosaic of Quality distribution
            fig     = plt.figure(figsize=(21,7))
            gsg_ext = gridspec.GridSpec(1, 2, hspace=0, wspace=0, width_ratios=[1,0.025])
            gsg_int = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsg_ext[0], hspace=0, wspace=0)
            
            # Data
            qual     = df_all.qualflag[(df_all[col_jdn] == jdni) & df_all.in_snr]
            hang     = df_all[col_hang    ][qual.index]
            airm     = df_all[col_airm    ][qual.index]
            magn     = df_all[col_magn    ][qual.index]
            magn_err = df_all[col_magn_err][qual.index]
            
            # Accepted & Rejected sub-sets
            cut = 0.95
            qual_in, qual_out = qual[qual>cut]     , qual[qual<cut]
            hang_in, hang_out = hang[qual_in.index], hang[qual_out.index]
            airm_in, airm_out = airm[qual_in.index], airm[qual_out.index]
            magn_in, magn_out = magn[qual_in.index], magn[qual_out.index]
            
            # SUBPLOT 1: Magnitude vs. Airmass
            
            ax00 = plt.subplot(gsg_int[0])
            
            # Elements
            axc0 = ax00.scatter(airm_in , magn_in , c=qual_in , marker='o', s=25, edgecolors='k', cmap=cmap,
                                vmin=0, vmax=1, zorder=100)
            axc0 = ax00.scatter(airm_out, magn_out, c=qual_out, marker='s', s=25, edgecolors='k', cmap=cmap,
                                vmin=0, vmax=1, zorder=100)
            ax00.errorbar(airm, magn, magn_err, fmt=',k', capsize=2.5)
            xlim     = ax00.get_xlim()
            airm_lin = np.array(xlim)
            ax00.plot(airm_lin, c0+c1*airm_lin, '-k')
            ax00.fill_between(airm_lin, c0+c1*airm_lin-jit, c0+c1*airm_lin+jit, color='gray', alpha=0.5)
            ax00.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            
            # x-axis
            ax00.set_xlabel('Airmass')
            ax00.set_xlim(xlim)
            
            # y-axis
            ax00.set_ylabel('$m$ [mag]')
            ax00.invert_yaxis()
            
            # Legend
            col1, = ax00.plot([], 'ko'  , mfc='w', label='Accepted')
            col2, = ax00.plot([], 'ks'  , mfc='w', label='Rejected')
            col3, = ax00.plot([], 'k'   ,          label='Best fit')
            col4, = ax00.plot([], 'gray', lw=4   , label=r'$\pm\sigma_{\mathrm{jit}}$')
            leg   = ax00.legend(handles=[col1,col2,col3,col4], loc='upper right', edgecolor='k', framealpha=1)
            leg.set_zorder(101)
            
            # SUBPLOT 2: Magnitude vs. Hour Angle
            
            ax01 = plt.subplot(gsg_int[1], sharey=ax00)
            
            # Elements
            ax01.scatter(hang_in , magn_in , c=qual_in , marker='o', s=25, edgecolors='k', cmap=cmap,
                        vmin=0, vmax=1, zorder=100)
            ax01.scatter(hang_out, magn_out, c=qual_out, marker='s', s=25, edgecolors='k', cmap=cmap,
                        vmin=0, vmax=1, zorder=100)
            ax01.errorbar(hang, magn, magn_err, fmt=',k', capsize=2.5)
            ax01.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelleft=False)
            
            # x-axis
            ax01.set_xlabel('Hour angle [°]')
            
            # COLORBAR
            cbax = plt.subplot(gsg_ext[1])
            cb   = Colorbar(ax=cbax, mappable=axc0)
            cb.set_label('P(good point)')
            cb.ax.axhline(cut, linestyle='--', color='k')
            
            # SAVE FIGURE
            fig.tight_layout()
            plt.savefig(fig_name(jdni))
            plt.close(fig)
    
    ### SAVE

    sun_data = df_all.merge(df_day, on=col_jdn, how='left')
    sun_data = sun_data.drop(columns=['in_snr', 'in_day'])

    return sun_data