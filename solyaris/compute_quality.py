from   astropy.time         import Time
import emcee                # version 2.2.1
from   matplotlib.colorbar  import Colorbar
import matplotlib.gridspec  as     gridspec
import matplotlib.pylab     as     pylab
import matplotlib.pyplot    as     plt
import numpy                as     np

def compute_quality(time_jdb, time_jdn, airm_val, snrx_val, mode=None, plot_results=False):

    ### MCMC PARAMETERS

    # Discription and bounds of MCMC parameters
    pname      = ['$\hat{c}_{0}$', '$c_{1}$', '$\sigma_{\mathrm{jit}}$', '$\hat{\mu}$', '$\sigma$', '$Q$']
    bounds     = [(-0.4  , 0.4 ),    # c_0_hat : intercept of foreground line w.r.t. weighted mean airmass
                  ( 0.075, 0.4 ),    # c_1     : slope of foreground line
                  ( 0.001, 0.08),    # jit     : white noise of foreground
                  ( 0.0  , 0.8 ),    # mu_hat  : mean of background w.r.t. weighted mean magnitude
                  ( 0.1  , 0.7 ),    # sig     : std of background
                  ( 0.0  , 1.0 )]    # Q       : fraction of foreground population
    ndim       = np.shape(bounds)[0] # nr. of parameters in MCMC
    c0_hat_max = 0.1                 # allowed max. for c0 w.r.t. weighted mean airmass
                                     # (larger values indicate a swap between fore- & background)

    # MCMC walkers and chains
    min_pnt = 10                  # allowed min. nr. of points per day
    nwalker = 32                  # nr. of walkers in MCMC
    nthin_b = 50                  # thinning step for sampling burn-in chains
    nthin_p = 100                 # thinning step for sampling production chain
    npt_end = int(2e3)            # nr. of end points from which to sample burn-ins
    nstep_b = int(1e4)            # nr. of steps in burn-in
    nstep_p = nstep_b             # nr. of steps in production

    ### PLOT PARAMETERS

    # Figure name
    def fig_name():

        date = Time(time_jdn[0], format='jd').isot[:10]
        name = f'Fig_{date}.pdf' if mode is None else f'Fig_{mode}_{date}.pdf'

        return name

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

    ### FUNCTIONS

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

    ### QUALITY

    # Nr. of points
    Npnt = len(time_jdb)

    # NaN placeholders
    c0, c0_hat, c1, jit, mu, mu_hat, sig, Q = np.empty(8, dtype=float)*np.nan
    quality = np.empty(Npnt, dtype=float)*np.nan

    # Index of valid SNR values
    idx = snrx_val > 1

    # Nr. of valid points
    Npnt_valid = np.sum(idx)

    # Proceed if day has enough valid points
    if Npnt_valid >= min_pnt:

        # New variables
        hour_utc = (time_jdb-time_jdn+0.5)*24
        magn_val = -5*np.log10(snrx_val)
        magn_err = 2.5/(np.log(10)*snrx_val)

        # x- and y-variables for linear relation
        x_val = airm_val[idx]
        y_val = magn_val[idx]
        y_err = magn_err[idx]
        x_hat = np.average(x_val, weights=1/y_err**2)
        y_hat = np.average(y_val, weights=1/y_err**2)
        
        # Set up sampler
        sampler = emcee.EnsembleSampler(nwalker, ndim, lnprob, args=(bounds,x_val-x_hat,y_val-y_hat,y_err))
        
        # 1st burn-in chain
        p0 = [[(b[1]+b[0])/2 for b in bounds] + 1e-5*np.random.randn(ndim) for _ in range(nwalker)]
        sampler.run_mcmc(np.array(p0), nstep_b)
        
        # 2nd burn-in chain
        p0 = gauss_bound(sampler.chain)
        sampler.reset()
        sampler.run_mcmc(p0, nstep_b)
        
        # Production chain
        p0 = gauss_bound(sampler.chain)
        sampler.reset()
        sampler.run_mcmc(p0, nstep_p)
        
        # Extract MCMC parameters
        para = np.array(sampler.flatchain[::nthin_p,:])
        c0_hat, c1, jit, mu_hat, sig, Q = np.median(para, axis=0).T

        # Convert parameters
        c0 = c0_hat + y_hat - x_hat*c1
        mu = mu_hat + y_hat
        
        # Proceed if fore- & background correctly identified
        if c0_hat < c0_hat_max:
            
            # Quality (i.e. probability of belonging to foreground)
            norm_val = 0.0
            qual_val = np.zeros(Npnt_valid)
            for ii in range(sampler.chain.shape[1]):
                for jj in range(sampler.chain.shape[0]):
                    ll_fg, ll_bg = sampler.blobs[ii][jj]
                    qual_val += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                    norm_val += 1
            qual_val /= norm_val

            # Store quality
            quality[idx] = qual_val

            ### PLOT
            
            if plot_results:
                
                # Figure: 2x1 mosaic
                fig     = plt.figure(figsize=(21,7))
                gsg_ext = gridspec.GridSpec(1, 2, hspace=0, wspace=0, width_ratios=[1,0.025])
                gsg_int = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsg_ext[0], hspace=0, wspace=0)

                # Valid points
                hour_utc_all = hour_utc[idx]
                airm_val_all = airm_val[idx]
                magn_val_all = magn_val[idx]
                magn_err_all = magn_err[idx]
                
                # Accepted and rejected points
                qual_cut = 0.90
                idx_in     , idx_out      = qual_val > qual_cut , qual_val < qual_cut
                hour_utc_in, hour_utc_out = hour_utc_all[idx_in], hour_utc_all[idx_out]
                airm_val_in, airm_val_out = airm_val_all[idx_in], airm_val_all[idx_out]
                magn_val_in, magn_val_out = magn_val_all[idx_in], magn_val_all[idx_out]
                qual_val_in, qual_val_out = qual_val    [idx_in], qual_val    [idx_out]
                
                # Subplot 1: Magnitude vs. Airmass
                ax00 = plt.subplot(gsg_int[0])
                
                # Elements
                axc0 = ax00.scatter(airm_val_in , magn_val_in , c=qual_val_in , marker='o', s=25, edgecolors='k', cmap=cmap, vmin=0, vmax=1, zorder=100)
                axc0 = ax00.scatter(airm_val_out, magn_val_out, c=qual_val_out, marker='s', s=25, edgecolors='k', cmap=cmap, vmin=0, vmax=1, zorder=100)
                ax00.errorbar(airm_val_all, magn_val_all, magn_err_all, fmt=',k', capsize=2.5)
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
                
                # Subplot 2: Magnitude vs. Time of day
                ax01 = plt.subplot(gsg_int[1], sharey=ax00)
                
                # Elements
                ax01.scatter(hour_utc_in , magn_val_in , c=qual_val_in , marker='o', s=25, edgecolors='k', cmap=cmap, vmin=0, vmax=1, zorder=100)
                ax01.scatter(hour_utc_out, magn_val_out, c=qual_val_out, marker='s', s=25, edgecolors='k', cmap=cmap, vmin=0, vmax=1, zorder=100)
                ax01.errorbar(hour_utc_all, magn_val_all, magn_err_all, fmt=',k', capsize=2.5)
                ax01.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelleft=False)
                
                # x-axis
                ax01.set_xlabel('Time of day in UTC [hr]')
                
                # Colorbar
                cbax = plt.subplot(gsg_ext[1])
                cb   = Colorbar(ax=cbax, mappable=axc0)
                cb.set_label('P(good point)')
                cb.ax.axhline(qual_cut, linestyle='--', color='k')
                
                # Save figure
                fig.tight_layout()
                plt.savefig(fig_name())
                plt.close(fig)

    return c0, c0_hat, c1, jit, mu, mu_hat, sig, Q, quality