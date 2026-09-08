import numpy    as     np
import pandas   as     pd
import pickle
from   tqdm     import tqdm
import warnings
warnings.filterwarnings('ignore')

from typing import Literal

from .compute_berv_and_herv     import compute_berv_and_herv
from .compute_quality           import compute_quality
from .compute_solar_coordinates import compute_solar_coordinates
from .extract_header_keywords   import extract_header_keywords
from .extract_order_by_order_rv import extract_order_by_order_rv

class SOLYARIS:

    def __init__(self, instrument:Literal['espresso', 'harps-n', 'neid'], id:str=None, verbose:bool=True):
        """Create SOLYARIS object.

        Parameters
        ----------
        instrument : Literal['espresso', 'harps-n', 'neid']
            Instrument name.
        id : str, optional
            SOLYARIS object ID, used as the name when saving the object as a pickled file. By default the instrument name.
        verbose : bool, optional
            Whether to print the name of the steps and progress bars. By default True.
        """

        # Instrument and SOLYARIS object ID
        self.instrument = instrument
        if id is None:
            self.id = instrument

        # Verbose
        self.verbose = verbose

        # ESPRESSO
        if self.instrument == 'espresso':
            self.Norder     = 170
            self.obsname    = 'Paranal Observatory'
            self.obscode    = 309

        # HARPS-N
        if self.instrument == 'harps-n':
            self.Norder     = 69
            self.obsname    = 'Roque de los Muchachos'
            self.obscode    = 950

        # NEID
        if self.instrument == 'neid':
            self.Norder     = 122
            self.obsname    = 'Kitt Peak National Observatory'
            self.obscode    = 695

    def add_data(self, files:list[str]=None):
        """Add data.

        Parameters
        ----------
        files : list[str], optional
            List of paths to FITS files. Must be CCF_A files for ESO instruments (ESPRESSO, HARPS, HARPS-N, NIRPS) and L2 files for EXPRES and NEID. By default None.

        Returns
        -------
        None
            Saves files in self.files and nr. of files in self.Nfile.
        """

        # Create 'files' and 'Nfile' properties
        self.files = files
        self.Nfile = len(files)

        return None

    def extract_header_keywords(self):
        """Extract header keywords.

        Returns
        -------
        None
            Saves a pandas.DataFrame with the extracted header keywords in self.table.
        """

        # Print action
        if self.verbose:
            print('Extracting header keywords.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Loop files
        for i in iterable:

            # Extract header keywords
            header_dict = extract_header_keywords(self.files[i], self.instrument, self.Norder)

            # Initiate DataFrame
            if i == 0:
                df = pd.DataFrame(columns=header_dict.keys())

            # Populate DataFrame
            df.loc[i] = header_dict
        
        # Create 'table' property
        self.table = df

        return None

    def extract_order_by_order_rv(self):
        """Extract order-by-order RV.

        Returns
        -------
        None
            Saves order-by-order RV values and errors as 'vrad_val_ORDER' and 'vrad_err_ORDER' columns in self.table,
            where ORDER is the order index starting from 1.
            Saves the summed CCF RV values and errors as 'vrad_val_sum' and 'vrad_err_sum'.
        """

        # Print action
        if self.verbose:
            print('Extracting order-by-order RV.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Empty arrays
        vrad_val = np.empty((self.Nfile, self.Norder+1), dtype=float)
        vrad_err = np.empty((self.Nfile, self.Norder+1), dtype=float)

        # Loop files
        for i in iterable:

            # Extract order-by-order RV
            vrad_val[i], vrad_err[i] = extract_order_by_order_rv(self.files[i], self.instrument, self.Norder)

        # Loop orders
        for i in range(self.Norder):

            # Insert output columns
            self.table[f'vrad_val_{i+1}'] = vrad_val[:,i]
            self.table[f'vrad_err_{i+1}'] = vrad_err[:,i]

        # Insert output columns
        self.table['vrad_val_sum'] = vrad_val[:,self.Norder]
        self.table['vrad_err_sum'] = vrad_err[:,self.Norder]

        return None

    def compute_berv_and_herv(self):
        """Compute BERV (Barycentric-Earth RV) and HERV (Heliocentric RV) corrections.

        Returns
        -------
        None
            Saves BERV and HERV as 'berv_val' and 'herv_val' columns in self.table. Renames BERV from header as 'berv_drs'.
        """

        # Print action
        if self.verbose:
            print('Computing BERV and HERV.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Empty arrays
        berv_val = np.empty(self.Nfile, dtype=float)
        herv_val = np.empty(self.Nfile, dtype=float)

        # Loop files
        for i in iterable:

            # Compute BERV and HERV
            berv_val[i], herv_val[i] = compute_berv_and_herv(self.obsname, self.table.date_obs.values[i], self.table.exp_time.values[i], self.table.photocen.values[i])

        # Rename header columns
        self.table.rename(columns={'berv_val': 'berv_drs'}, inplace=True)

        # Insert output columns
        self.table.insert(self.table.columns.get_loc('berv_drs')+1, 'berv_val', berv_val)
        self.table.insert(self.table.columns.get_loc('berv_drs')+2, 'herv_val', herv_val)

        return None

    def compute_solar_coordinates(self):
        """Compute solar coordinates.

        Returns
        -------
        None
            Saves RA, Dec and airmass as 'alph_val', 'delt_val' and 'airm_val' columns in self.table. Renames corresponding variables from header as 'alph_drs', 'delt_drs' and 'airm_drs'. 
        """

        # Print action
        if self.verbose:
            print('Computing solar coordinates.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Empty arrays
        alph_val = np.empty(self.Nfile, dtype=float)
        delt_val = np.empty(self.Nfile, dtype=float)
        airm_val = np.empty(self.Nfile, dtype=float)

        # Loop files
        for i in iterable:

            # Compute solar coordinates
            alph_val[i], delt_val[i], airm_val[i] = compute_solar_coordinates(self.obsname, self.table.date_obs.values[i], self.table.exp_time.values[i], self.table.photocen.values[i])

        # Rename header columns
        self.table.rename(columns={'alph_val': 'alph_drs'}, inplace=True)
        self.table.rename(columns={'delt_val': 'delt_drs'}, inplace=True)
        self.table.rename(columns={'airm_val': 'airm_drs'}, inplace=True)

        # Insert output columns
        self.table.insert(self.table.columns.get_loc('alph_drs')+1, 'alph_val', alph_val)
        self.table.insert(self.table.columns.get_loc('delt_drs')+1, 'delt_val', delt_val)
        self.table.insert(self.table.columns.get_loc('airm_drs')+1, 'airm_val', airm_val)

        return None

    def compute_quality(self, ref_order:int=1, plot_results:bool=False):
        """Compute quality.

        Parameters
        ----------
        ref_order : int, optional
            Reference order index (starting from 1) for computing the apparent magnitude using the S/N. By default 1.
        plot_results : bool, optional
            Whether to plot the results. By default False.

        Returns
        -------
        None
            Saves quality and associated MCMC parameters as 'quality', 'c0', 'c0_hat', 'c1', 'jit', 'mu', 'mu_hat', 'sig' and 'Q' columns in self.table.
        """

        # Empty arrays
        c0     = np.empty(self.Nfile, dtype=float)
        c0_hat = np.empty(self.Nfile, dtype=float)
        c1     = np.empty(self.Nfile, dtype=float)
        jit    = np.empty(self.Nfile, dtype=float)
        mu     = np.empty(self.Nfile, dtype=float)
        mu_hat = np.empty(self.Nfile, dtype=float)
        sig    = np.empty(self.Nfile, dtype=float)
        Q      = np.empty(self.Nfile, dtype=float)
        qual   = np.empty(self.Nfile, dtype=float)

        # Extract and compute required variables
        time_jdb = self.table[ 'time_jdb'       ].to_numpy(copy=True)
        airm_val = self.table[ 'airm_val'       ].to_numpy(copy=True)
        snrx_val = self.table[f'snr_{ref_order}'].to_numpy(copy=True)
        time_jdn = np.floor(time_jdb + 0.5).astype(int)

        # Instrument modes
        ins_modes = np.unique(self.table.ins_mode.values)

        # Loop instrument modes
        for ins_mode in ins_modes:

            # Index of mode
            i_mode = self.table.ins_mode.values == ins_mode

            # Days of mode
            days = np.unique(time_jdn[i_mode])
            Nday = len(days)

            # Nr. of files of mode
            Nfile = np.sum(i_mode)

            # Print action
            if self.verbose:
                print(f'Computing quality for {ins_mode} mode.')
                print(f'-> Processing {Nfile} files over {Nday} days ...')

            # Iterable
            iterable = tqdm(range(Nday)) if self.verbose else range(Nday)

            # Loop days
            for i in iterable:

                # Index of day
                i_day = time_jdn == days[i]

                # Total index
                ii = i_mode & i_day

                # Compute quality
                c0[ii], c0_hat[ii], c1[ii], jit[ii], mu[ii], mu_hat[ii], sig[ii], Q[ii], qual[ii] = compute_quality(time_jdb[ii], time_jdn[ii], airm_val[ii], snrx_val[ii], ins_mode, plot_results)

        # Insert output columns
        self.table['c0'     ] = c0
        self.table['c0_hat' ] = c0_hat
        self.table['c1'     ] = c1
        self.table['jit'    ] = jit
        self.table['mu'     ] = mu
        self.table['mu_hat' ] = mu_hat
        self.table['sig'    ] = sig
        self.table['Q'      ] = Q
        self.table['quality'] = qual

        return None

    def export_table(self):

        self.table.to_csv(self.id+'.csv', index=False)

        return None

def save(solyaris):

    return pickle.dump(solyaris, open(solyaris.id+'.solyaris', 'wb'))

def load(solyaris):

    return pickle.load(open(solyaris, 'rb'))