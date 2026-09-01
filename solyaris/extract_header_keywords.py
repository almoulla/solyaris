from   astropy.io import fits
import pandas     as     pd
from   tqdm       import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_header_keywords(files, instrument, Norder):

    # Nr. of files
    Nfile = len(files)

    # Loop files
    print(f'Extracting header keywords ...')
    for i in tqdm(range(Nfile)):

        # Load FITS file
        with fits.open(files[i]) as hdul:

            # Header
            header = hdul[0].header

            # ESPRESSO
            if instrument == 'espresso':

                # Extract header keywords
                header_dict = {}
                header_dict['file'    ] = header['ARCFILE'                               ]
                header_dict['targ'    ] = header['HIERARCH ESO OBS TARG NAME'            ]
                header_dict['date'    ] = header['DATE-OBS'                              ][:10]
                header_dict['date_obs'] = header['DATE-OBS'                              ]
                header_dict['ins_mode'] = header['HIERARCH ESO INS MODE'                 ]
                header_dict['obj_type'] = header['HIERARCH ESO PRO REC1 RAW1 CATG'       ]
                header_dict['exp_time'] = header['EXPTIME'                               ]
                header_dict['photocen'] = header['HIERARCH ESO QC TMMEAN USED'           ]
                header_dict['alph_val'] = header['HIERARCH ESO TEL5 TARG ALPHA'          ]
                header_dict['delt_val'] = header['HIERARCH ESO TEL5 TARG DELTA'          ]
                header_dict['airm_val'] = header['HIERARCH ESO TEL5 AIRM START'          ]*(1-header_dict['photocen']) + header['HIERARCH ESO TEL5 AIRM END']*header_dict['photocen']
                header_dict['berv_val'] = header['HIERARCH ESO QC BERV'                  ]
                header_dict['time_jdb'] = header['HIERARCH ESO QC BJD'                   ]
                header_dict['vrad_val'] = header['HIERARCH ESO QC CCF RV'                ]
                header_dict['vrad_err'] = header['HIERARCH ESO QC CCF RV ERROR'          ]
                for j in range(Norder):
                    header_dict[f'snr_{j+1}'] = header[f'HIERARCH ESO QC ORDER{j+1} SNR' ]

        # Initiate DataFrame
        if i == 0:
            df = pd.DataFrame(columns=header_dict.keys())

        # Populate DataFrame
        df.loc[i] = header_dict

    return df