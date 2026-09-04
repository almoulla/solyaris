import numpy as np
import pickle

from .compute_berv_and_herv     import compute_berv_and_herv
from .compute_quality_flag      import compute_quality_flag
from .compute_solar_coordinates import compute_solar_coordinates
from .extract_header_keywords   import extract_header_keywords
from .extract_order_by_order_rv import extract_order_by_order_rv

class SOLYARIS:

    def __init__(self, instrument, id=None):

        # Instrument and SOLYARIS object ID
        self.instrument = instrument
        if id is None:
            self.id = instrument

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

    def add_data(self, files=None):

        self.files = files

        return None

    def extract_header_keywords(self):

        print(f'Extracting header keywords ...')
        self.table = extract_header_keywords(self.files, self.instrument, self.Norder)

        return None

    def extract_order_by_order_rv(self):

        print('Extracting order-by-order RV ...')
        vrad_val_ord, vrad_err_ord = extract_order_by_order_rv(self.files, self.instrument, self.Norder)

        for j in range(self.Norder+1):
            self.table[f'vrad_val_{j+1}'] = vrad_val_ord[:,j]
            self.table[f'vrad_err_{j+1}'] = vrad_err_ord[:,j]

        return None

    def compute_berv_and_herv(self):

        print('Computing BERV and HERV ...')
        berv_val, herv_val = compute_berv_and_herv(self.obsname, self.table.date_obs.values, self.table.exp_time.values, self.table.photocen.values)

        self.table.rename(columns={'berv_val': 'berv_drs'}, inplace=True)
        
        self.table.insert(self.table.columns.get_loc('berv_drs')+1, 'berv_val', berv_val)
        self.table.insert(self.table.columns.get_loc('berv_drs')+2, 'herv_val', herv_val)

        return None

    def compute_solar_coordinates(self):

        print('Computing solar coordinates ...')
        alph_val, delt_val, airm_val = compute_solar_coordinates(self.obsname, self.table.date_obs.values, self.table.exp_time.values, self.table.photocen.values)

        self.table.rename(columns={'alph_val': 'alph_drs'}, inplace=True)
        self.table.rename(columns={'delt_val': 'delt_drs'}, inplace=True)
        self.table.rename(columns={'airm_val': 'airm_drs'}, inplace=True)
        
        self.table.insert(self.table.columns.get_loc('alph_drs')+1, 'alph_val', alph_val)
        self.table.insert(self.table.columns.get_loc('delt_drs')+1, 'delt_val', delt_val)
        self.table.insert(self.table.columns.get_loc('airm_drs')+1, 'airm_val', airm_val)

        return None

    def compute_quality_flag(self, ref_order=1):

        keys = ['c0', 'c0_hat', 'c1', 'jit', 'mu', 'mu_hat', 'sig', 'Q', 'qualflag']
        for key in keys:
            self.table[f'{key}'] = np.empty(self.table.shape[0])

        ins_modes = np.unique(self.table.ins_mode.values)
        for ins_mode in ins_modes:

            print(f'Computing quality flag for {ins_mode} mode ...')

            idx = self.table.ins_mode.values == ins_mode
            df  = compute_quality_flag(self.table.time_jdb.values[idx], self.table.airm_val.values[idx], self.table[f'snr_{ref_order}'].values[idx], ins_mode)

            for key in keys:
                self.table.loc[idx, key] = df[f'{key}'].values

        return None

    def export_table(self):

        self.table.to_csv(self.id+'.csv', index=False)

        return None

def save(solyaris):

    return pickle.dump(solyaris, open(solyaris.id+'.solyaris', 'wb'))

def load(solyaris):

    return pickle.load(open(solyaris, 'rb'))