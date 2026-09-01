import pickle

from .compute_berv_and_herv     import compute_berv_and_herv
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

    def add_data(self, files=None):

        self.files = files

        return None

    def extract_header_keywords(self):

        self.table = extract_header_keywords(self.files, self.instrument, self.Norder)

        return None

    def extract_order_by_order_rv(self):

        vrad_val_ord, vrad_err_ord = extract_order_by_order_rv(self.files, self.instrument, self.Norder)

        for j in range(self.Norder+1):
            self.table[f'vrad_val_{j+1}'] = vrad_val_ord[:,j]
            self.table[f'vrad_err_{j+1}'] = vrad_err_ord[:,j]

        return None

    def compute_solar_coordinates(self):
        
        alph_val, delt_val, airm_val = compute_solar_coordinates(self.obsname, self.table.date_obs.values, self.table.exp_time.values, self.table.photocen.values)

        self.table.rename(columns={'alph_val': 'alph_drs'}, inplace=True)
        self.table.rename(columns={'delt_val': 'delt_drs'}, inplace=True)
        self.table.rename(columns={'airm_val': 'airm_drs'}, inplace=True)
        
        self.table.insert(self.table.columns.get_loc('alph_drs')+1, 'alph_val', alph_val)
        self.table.insert(self.table.columns.get_loc('delt_drs')+1, 'delt_val', delt_val)
        self.table.insert(self.table.columns.get_loc('airm_drs')+1, 'airm_val', airm_val)

        return None

    def compute_berv_and_herv(self):

        berv_val, herv_val = compute_berv_and_herv(self.obsname, self.table.date_obs.values, self.table.exp_time.values, self.table.photocen.values)

        self.table.rename(columns={'berv_val': 'berv_drs'}, inplace=True)
        
        self.table.insert(self.table.columns.get_loc('berv_drs')+1, 'berv_val', berv_val)
        self.table.insert(self.table.columns.get_loc('berv_drs')+2, 'herv_val', herv_val)

        return None

    def export_table(self):

        self.table.to_csv(self.id+'.csv', index=False)

        return None

def save(solyaris):

    return pickle.dump(solyaris, open(solyaris.id+'.solyaris', 'wb'))

def load(solyaris):

    return pickle.load(open(solyaris, 'rb'))