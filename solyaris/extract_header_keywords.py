from   astropy.io import fits
import numpy      as     np
import pandas     as     pd
from   tqdm       import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_header_keywords(files, instrument, Norder):

    # Nr. of files
    Nfile = len(files)

    # Loop files
    for i in tqdm(range(Nfile)):

        # Load FITS file
        with fits.open(files[i]) as hdul:

            # ESPRESSO
            if instrument == 'espresso':

                # Extract header keywords
                header_dict = {}
                header                  = hdul[0].header
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
                header_dict['fwhm_val'] = header['HIERARCH ESO QC CCF FWHM'              ]
                header_dict['fwhm_err'] = header['HIERARCH ESO QC CCF FWHM ERROR'        ]
                header_dict['cont_val'] = header['HIERARCH ESO QC CCF CONTRAST'          ]
                header_dict['cont_err'] = header['HIERARCH ESO QC CCF CONTRAST ERROR'    ]
                header_dict['biss_val'] = header['HIERARCH ESO QC CCF BIS SPAN'          ]
                header_dict['biss_err'] = header['HIERARCH ESO QC CCF BIS SPAN ERROR'    ]
                for j in range(Norder):
                    header_dict[f'snr_{j+1}'] = header[f'HIERARCH ESO QC ORDER{j+1} SNR' ]

            # HARPS-N
            if instrument == 'harps-n':

                # Extract header keywords
                header_dict = {}
                header                  = hdul[0].header
                header_dict['file'    ] = header['FILENAME'                              ]
                header_dict['targ'    ] = header['HIERARCH TNG OBS TARG NAME'            ]
                header_dict['date'    ] = header['DATE-OBS'                              ][:10]
                header_dict['date_obs'] = header['DATE-OBS'                              ]
                header_dict['ins_mode'] = header['HIERARCH TNG INS MODE'                 ]
                header_dict['obj_type'] = header['HIERARCH ESO PRO REC1 RAW2 CATG'       ]
                header_dict['exp_time'] = header['EXPTIME'                               ]
                header_dict['photocen'] = header['HIERARCH TNG EXP_METER_A EXP CENTROID' ]
                header_dict['alph_val'] = header['HIERARCH TNG TEL TARG ALPHA'           ]
                header_dict['delt_val'] = header['HIERARCH TNG TEL TARG DELTA'           ]
                header_dict['airm_val'] = header['AIRMASS'                               ]
                header_dict['berv_val'] = header['HIERARCH TNG QC BERV'                  ]
                header_dict['time_jdb'] = header['HIERARCH TNG QC BJD'                   ]
                header_dict['vrad_val'] = header['HIERARCH TNG QC CCF RV'                ]
                header_dict['vrad_err'] = header['HIERARCH TNG QC CCF RV ERROR'          ]
                header_dict['fwhm_val'] = header['HIERARCH TNG QC CCF FWHM'              ]
                header_dict['fwhm_err'] = header['HIERARCH TNG QC CCF FWHM ERROR'        ]
                header_dict['cont_val'] = header['HIERARCH TNG QC CCF CONTRAST'          ]
                header_dict['cont_err'] = header['HIERARCH TNG QC CCF CONTRAST ERROR'    ]
                header_dict['biss_val'] = header['HIERARCH TNG QC CCF BIS SPAN'          ]
                header_dict['biss_err'] = header['HIERARCH TNG QC CCF BIS SPAN ERROR'    ]
                for j in range(Norder):
                    header_dict[f'snr_{j+1}'] = header[f'HIERARCH TNG QC ORDER{j+1} SNR' ]

            # NEID
            if instrument == 'neid':

                # Extract header keywords
                header_dict = {}
                header                  = hdul[0].header
                header_dict['file'    ] = header['FILENAME'                              ]
                header_dict['targ'    ] = header['OBJECT'                                ]
                header_dict['date'    ] = header['DATE-OBS'                              ][:10]
                header_dict['date_obs'] = header['DATE-OBS'                              ]
                header_dict['ins_mode'] = header['OBS-MODE'                              ]
                header_dict['obj_type'] = header['WAVECAL'                               ]
                header_dict['exp_time'] = header['EXPTIME'                               ]
                header_dict['photocen'] = 0.5
                header_dict['alph_val'] = header['TCSRA'                                 ]
                header_dict['delt_val'] = header['TCSDEC'                                ]
                header_dict['airm_val'] = header['AIRMASS'                               ]
                header_dict['berv_val'] = header['SSBRV100'                              ]
                header                  = hdul[12].header
                header_dict['time_jdb'] = header['CCFJDMOD'                              ]
                header_dict['vrad_val'] = header['CCFRVMOD'                              ]
                header_dict['vrad_err'] = header['DVRMSMOD'                              ]
                header_dict['fwhm_val'] = header['FWHMMOD'                               ]
                header_dict['fwhm_err'] = header['EFWHMMOD'                              ]
                header_dict['cont_val'] = (header['FITMODV0']+header['FITMODV3'])/header['FITMODV0']
                header_dict['cont_err'] = np.nan
                header_dict['biss_val'] = header['BISMOD'                                ]
                header_dict['biss_err'] = header['EBISMOD'                               ]
                header                  = hdul[0].header
                header_dict['snr_drs' ] = header['EXTSNR'                                ]
                Npix = hdul[1].data.shape[1]
                for j in range(Norder):
                    header_dict[f'snr_{j+1}'] = hdul[1].data[j,Npix//2]/np.sqrt(hdul[4].data[j,Npix//2])

        # Initiate DataFrame
        if i == 0:
            df = pd.DataFrame(columns=header_dict.keys())

        # Populate DataFrame
        df.loc[i] = header_dict

    return df