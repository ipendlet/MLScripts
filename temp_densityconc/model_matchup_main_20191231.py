#!/usr/bin/env python
# coding: utf-8
#%%
'''
    Code base used for generating the data used in comparing the different density models
for leave-one-amine-out testing and standard test train splits on the perovskite data.

Some visualization also included.  

Due to the specificity of some of the underlying assumptions in this code it is recommended to contact
the authors prior to using for publication.  This code was designed specifically for the dataset
targeted and should not be assumed to be functional for any subsequent dataset of the perovksite project.

Authors -- Aaron Dharna, Ian Pendleton
 
Usage: execute from main function of code
'''

import os
import re
import copy
import logging
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
if not sklearn.__version__ == '0.21.3':
    raise ValueError("Update Scikit-learn with: pip3 install --upgrade --user scikit-learn")

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split #StratifiedShuffleSplit
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

#get_ipython().run_line_magic('matplotlib', 'inline')
# pd.set_option('display.max_columns', None) # so we can actually see the whole dataset with pd.DataFrame.head()
# pd.set_option('display.max_rows', None)


def buildlogger():
    # create logger with 'initialize'
    logger = logging.getLogger('models')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs event debug messages
    logfile = './AnalysisLog.txt'
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    wh = logging.StreamHandler()
    wh.setLevel(logging.WARN)
    logger.addHandler(fh)
    logger.addHandler(wh)
    logger.info('initializing run')
    return(logger)
logger = buildlogger()


INCHI_TO_CHEMNAME = {'null':'null',
'YEJRWHAVMIAJKC-UHFFFAOYSA-N':'Gamma-Butyrolactone',
'IAZDPXIOMUYVGZ-UHFFFAOYSA-N':'Dimethyl sulfoxide',
'BDAGIHXWWSANSR-UHFFFAOYSA-N':'Formic Acid',
'RQQRAHKHDFPBMC-UHFFFAOYSA-L':'Lead Diiodide',
'XFYICZOIWSBQSK-UHFFFAOYSA-N':'Ethylammonium Iodide',
'LLWRXQXPJMPHLR-UHFFFAOYSA-N':'Methylammonium iodide',
'UPHCENSIMPJEIS-UHFFFAOYSA-N':'Phenethylammonium iodide',
'GGYGJCFIYJVWIP-UHFFFAOYSA-N':'Acetamidinium iodide',
'CALQKRVFTWDYDG-UHFFFAOYSA-N':'n-Butylammonium iodide',
'UUDRLGYROXTISK-UHFFFAOYSA-N':'Guanidinium iodide',
'YMWUJEATGCHHMB-UHFFFAOYSA-N':'Dichloromethane',
'JMXLWMIFDJCGBV-UHFFFAOYSA-N':'Dimethylammonium iodide',
'KFQARYBEAKAXIC-UHFFFAOYSA-N':'Phenylammonium Iodide',
'NLJDBTZLVTWXRG-UHFFFAOYSA-N':'t-Butylammonium Iodide',
'GIAPQOZCVIEHNY-UHFFFAOYSA-N':'N-propylammonium Iodide',
'QHJPGANWSLEMTI-UHFFFAOYSA-N':'Formamidinium Iodide',
'WXTNTIQDYHIFEG-UHFFFAOYSA-N':'1,4-Diazabicyclo[2,2,2]octane-1,4-diium Iodide',
'LCTUISCIGMWMAT-UHFFFAOYSA-N':'4-Fluoro-Benzylammonium iodide',
'NOHLSFNWSBZSBW-UHFFFAOYSA-N':'4-Fluoro-Phenethylammonium iodide',
'FJFIJIDZQADKEE-UHFFFAOYSA-N':'4-Fluoro-Phenylammonium iodide',
'QRFXELVDJSDWHX-UHFFFAOYSA-N':'4-Methoxy-Phenylammonium iodide',
'SQXJHWOXNLTOOO-UHFFFAOYSA-N':'4-Trifluoromethyl-Benzylammonium iodide',
'KOAGKPNEVYEZDU-UHFFFAOYSA-N':'4-Trifluoromethyl-Phenylammonium iodide',
'MVPPADPHJFYWMZ-UHFFFAOYSA-N':'chlorobenzene',
'CWJKVUQGXKYWTR-UHFFFAOYSA-N':'Acetamidinium bromide',
'QJFMCHRSDOLMHA-UHFFFAOYSA-N':'Benzylammonium Bromide',
'PPCHYMCMRUGLHR-UHFFFAOYSA-N':'Benzylammonium Iodide',
'XAKAQFUGWUAPJN-UHFFFAOYSA-N':'Beta Alanine Hydroiodide',
'KOECRLKKXSXCPB-UHFFFAOYSA-K':'Bismuth iodide',
'XQPRBTXUXXVTKB-UHFFFAOYSA-M':'Cesium iodide',
'ZMXDDKWLCZADIW-UHFFFAOYSA-N':'Dimethylformamide',
'BCQZYUOYVLJOPE-UHFFFAOYSA-N':'Ethane-1,2-diammonium bromide',
'IWNWLPUNKAYUAW-UHFFFAOYSA-N':'Ethane-1,2-diammonium iodide',
'PNZDZRMOBIIQTC-UHFFFAOYSA-N':'Ethylammonium bromide',
'QWANGZFTSGZRPZ-UHFFFAOYSA-N':'Formamidinium bromide',
'VQNVZLDDLJBKNS-UHFFFAOYSA-N':'Guanidinium bromide',
'VMLAEGAAHIIWJX-UHFFFAOYSA-N':'i-Propylammonium iodide',
'JBOIAZWJIACNJF-UHFFFAOYSA-N':'Imidazolium Iodide',
'RFYSBVUZWGEPBE-UHFFFAOYSA-N':'iso-Butylammonium bromide',
'FCTHQYIDLRRROX-UHFFFAOYSA-N':'iso-Butylammonium iodide',
'UZHWWTHDRVLCJU-UHFFFAOYSA-N':'iso-Pentylammonium iodide',
'MCEUZMYFCCOOQO-UHFFFAOYSA-L':'Lead(II) acetate trihydrate',
'ZASWJUOMEGBQCQ-UHFFFAOYSA-L':'Lead(II) bromide',
'ISWNAMNOYHCTSB-UHFFFAOYSA-N':'Methylammonium bromide',
'VAWHFUNJDMQUSB-UHFFFAOYSA-N':'Morpholinium Iodide',
'VZXFEELLBDNLAL-UHFFFAOYSA-N':'n-Dodecylammonium bromide',
'PXWSKGXEHZHFJA-UHFFFAOYSA-N':'n-Dodecylammonium iodide',
'VNAAUNTYIONOHR-UHFFFAOYSA-N':'n-Hexylammonium iodide',
'HBZSVMFYMAOGRS-UHFFFAOYSA-N':'n-Octylammonium Iodide',
'FEUPHURYMJEUIH-UHFFFAOYSA-N':'neo-Pentylammonium bromide',
'CQWGDVVCKBJLNX-UHFFFAOYSA-N':'neo-Pentylammonium iodide',
'IRAGENYJMTVCCV-UHFFFAOYSA-N':'Phenethylammonium bromide',
'UXWKNNJFYZFNDI-UHFFFAOYSA-N':'piperazine dihydrobromide',
'QZCGFUVVXNFSLE-UHFFFAOYSA-N':'Piperazine-1,4-diium iodide',
'HBPSMMXRESDUSG-UHFFFAOYSA-N':'Piperidinium Iodide',
'IMROMDMJAWUWLK-UHFFFAOYSA-N':'Poly(vinyl alcohol), Mw89000-98000, >99% hydrolyzed)',
'QNBVYCDYFJUNLO-UHDJGPCESA-N':'Pralidoxime iodide',
'UMDDLGMCNFAZDX-UHFFFAOYSA-O':'Propane-1,3-diammonium iodide',
'VFDOIPKMSSDMCV-UHFFFAOYSA-N':'Pyrrolidinium Bromide',
'DMFMZFFIQRMJQZ-UHFFFAOYSA-N':'Pyrrolidinium Iodide',
'DYEHDACATJUKSZ-UHFFFAOYSA-N':'Quinuclidin-1-ium bromide',
'LYHPZBKXSHVBDW-UHFFFAOYSA-N':'Quinuclidin-1-ium iodide',
'UXYJHTKQEFCXBJ-UHFFFAOYSA-N':'tert-Octylammonium iodide',
'BJDYCCHRZIFCGN-UHFFFAOYSA-N':'Pyridinium Iodide',
'ZEVRFFCPALTVDN-UHFFFAOYSA-N':'Cyclohexylmethylammonium iodide',
'WGYRINYTHSORGH-UHFFFAOYSA-N':'Cyclohexylammonium iodide',
'XZUCBFLUEBDNSJ-UHFFFAOYSA-N':'Butane-1,4-diammonium Iodide',
'RYYSZNVPBLKLRS-UHFFFAOYSA-N':'1,4-Benzene diammonium iodide',
'DWOWCUCDJIERQX-UHFFFAOYSA-M':'5-Azaspiro[4.4]nonan-5-ium iodide',
'YYMLRIWBISZOMT-UHFFFAOYSA-N':'Diethylammonium iodide',
'UVLZLKCGKYLKOR-UHFFFAOYSA-N':'2-Pyrrolidin-1-ium-1-ylethylammonium iodide',
'BAMDIFIROXTEEM-UHFFFAOYSA-N':'N,N-Dimethylethane- 1,2-diammonium iodide',
'JERSPYRKVMAEJY-UHFFFAOYSA-N':'N,N-dimethylpropane- 1,3-diammonium iodide',
'NXRUEVJQMBGVAT-UHFFFAOYSA-N':'N,N-Diethylpropane-1,3-diammonium iodide',
'N/A':'None'}


VERSION_DATA_PATH = './'
CRANK_FILE = '0045.perovskitedata.csv'

def _stratify(data0, data1, out, inchis):

    stratifiedData0 = pd.DataFrame()
    stratifiedData1 = pd.DataFrame()

    stratifiedOut = pd.DataFrame()
    indicies = {}
    for i, x in enumerate(np.unique(inchis.values)):
        z = inchis.values == x
        total_amine0 = data0[z].reset_index(drop=True)
        total_amine1 = data1[z].reset_index(drop=True)

        amine_out = out[z].reset_index(drop=True)

        # this is still experimental and can easily be changed.
        try:
            uniformSamples = np.random.choice(total_amine0.index, size=96, replace=False)
        except Exception:
            uniformSamples = np.random.choice(total_amine0.index, size=96)
        sampled_amine0 = total_amine0.loc[uniformSamples]
        sampled_amine1 = total_amine1.loc[uniformSamples]
        
        sampled_out = amine_out.loc[uniformSamples]

        # save pointer to where this amine lives in the stratified dataset.
        # this isn't needed for random-TTS, but makes doing the Leave-One-Amine-Out 
        # train-test-splitting VERY EASY. 
        indicies[x] = np.array(range(96)) + i*96

        stratifiedData0 = pd.concat([stratifiedData0, sampled_amine0]).reset_index(drop=True)
        stratifiedData1 = pd.concat([stratifiedData1, sampled_amine1]).reset_index(drop=True)
        stratifiedOut = pd.concat([stratifiedOut, sampled_out]).reset_index(drop=True)
    
    stratifiedOut = stratifiedOut.iloc[:,0]
    return stratifiedData0, stratifiedData1, stratifiedOut, indicies

def _prepare(shuffle=0, deep_shuffle=0):
    ''' reads in perovskite dataframe and returns only experiments that meet specific criteria

    --> Data preparation occurs here
    criteria for main dataset include experiment version 1.1 (workflow 1 second generation), only
    reactions that use GBL, and 
    '''
    perov = pd.read_csv(os.path.join(VERSION_DATA_PATH, CRANK_FILE), skiprows=4, low_memory=False)
    logger.info(f'Initial df import with rows = {perov.shape[0]} and columns = {perov.shape[1]}')
    perov = perov[perov['_raw_ExpVer'] == 1.1].reset_index(drop=True)
    logger.info(f'Remove all but workflow 1.1: rows = {perov.shape[0]} and columns = {perov.shape[1]}')

    # only reaction that use GBL as a solvent (1:1 comparisons -- DMF and other solvents could convolute analysis)    
    perov = perov[perov['_raw_reagent_0_chemicals_0_InChIKey'] == "YEJRWHAVMIAJKC-UHFFFAOYSA-N"].reset_index(drop=True)    
    logger.info(f'Only experiments with GBL as the solvent: rows = {perov.shape[0]} and columns = {perov.shape[1]}')
    logger.info(f"Unique LOO groups remaining {len(perov['_rxn_organic-inchikey'].unique())}")

    # removes some anomalous entries with dimethyl ammonium still listed as the organic.
    perov = perov[perov['_rxn_organic-inchikey'] != 'JMXLWMIFDJCGBV-UHFFFAOYSA-N'].reset_index(drop=True)
    logger.info('removing anomalous entries with dimethyl ammonium incorrectly listed: rows = {perov.shape[0]} and columns = {perov.shape[1]}')
    logger.info(f"Unique LOO groups remaining {len(perov['_rxn_organic-inchikey'].unique())}")

    #We need to know which reactions have no succes and which have some
    organickeys = perov['_rxn_organic-inchikey']
    uniquekeys = organickeys.unique()

    df_key_dict = {}
    #find an remove all organics with no successes (See SI for reasoning)
    for key in uniquekeys:
        #build a dataframe name by the first 10 characters of the inchi containing all rows with that inchi
        df_key_dict[str(key)] = perov[perov['_rxn_organic-inchikey'] == key]
    all_groups = []
    successful_groups = []
    failed_groups = []
    for key, value in df_key_dict.items():
        all_groups.append(key)
        if 4 in value['_out_crystalscore'].values.astype(int):
            successful_groups.append(key)
        else:
            failed_groups.append(key)

    logger.info(f'{len(all_groups)} total leave-one-out groups (organoammoniums in this case) detected')
    logger.info(f'{len(failed_groups)} failed leave-one-out groups (organoammoniums in this case) detected')
    logger.info(f'{len(successful_groups)} successful leave-one-out groups (organoammoniums in this case) detected')
    logger.info(f'List of failed inchis: {failed_groups}')
    logger.info(successful_groups)
    #only grab reactions where there were some recorded successes in the amine grouping
    successful_perov = (perov[perov['_rxn_organic-inchikey'].isin(successful_groups)])
    successful_perov = successful_perov[successful_perov['_rxn_organic-inchikey'] != 'JMXLWMIFDJCGBV-UHFFFAOYSA-N'].reset_index(drop=True)

    # we need to do this so we can drop nans and such while keeping the data consistent
    # we couldnt do this on the full perov data since dropna() would nuke the entire dataset (rip)
    all_columns = successful_perov.columns
    
    full_data = successful_perov[all_columns].reset_index(drop=True)

    full_data = full_data.fillna(0).reset_index(drop=True)
    successful_perov = full_data[full_data['_out_crystalscore'] != 0].reset_index(drop=True)
    
    ## Shuffle options for these unique runs
    out_hold = pd.DataFrame()
    out_hold['out_crystalscore'] = successful_perov['_out_crystalscore']
    if shuffle == 1:
        logger.info('Shallow shuffle toggled for this analysis')
        out_hold['out_crystalscore'] = successful_perov['_out_crystalscore']
        successful_perov = successful_perov.reindex(np.random.permutation(successful_perov.index)).reset_index(drop=True)
        successful_perov['_out_crystalscore'] = out_hold['out_crystalscore']
    if deep_shuffle == 1:
        # Only want to shuffle particular columns (some shuffles will break processing), we will attempt to describe each selection in text
        logger.info('deep shuffle toggled for this analysis')

        #build holdout (not shuffled)
        out_hold_deep_df = pd.DataFrame()
        out_hold_deep_df = successful_perov.loc[:, '_raw_model_predicted':'_prototype_heteroatomINT']
        out_hold_deep_df = pd.concat([successful_perov['_rxn_organic-inchikey'], out_hold_deep_df], axis=1) 

        #isolate shuffle set
        shuffle_deep_df = pd.DataFrame()
        shuffle_deep_df = pd.concat([successful_perov.loc[:, 'name':'_rxn_M_organic'], 
                                     successful_perov.loc[:, '_rxn_temperatureC_actual_bulk' : '_feat_Hacceptorcount']], 
                                     axis = 1)
        successful_perov = shuffle_deep_df.apply(np.random.permutation)

        successful_perov.reset_index(drop=True)
        successful_perov = pd.concat([out_hold_deep_df, successful_perov], axis=1)

    logger.info(f'Only experiments with some group successes: rows = {successful_perov.shape[0]} and columns = {successful_perov.shape[1]}')

    successful_perov.rename(columns={"_raw_v0-M_acid": "_rxn_v0-M_acid", "_raw_v0-M_inorganic": "_rxn_v0-M_inorganic", "_raw_v0-M_organic":"_rxn_v0-M_organic"}, inplace=True)

    return successful_perov

def _computeProportionalConc(perovRow, v1=False, chemtype='organic'):
    """Compute the concentration of acid, inorganic, or acid for a given row of a crank dataset
    
    Intended to be pd.DataFrame.applied over the rows of a crank dataset
    
    :param perovRow: a row of the crank dataset
    :param v1: use v1 concentration or v0 
    :param chemtype: in ['organic', 'inorganic', 'acid']
    
    
    Currently hard codes inorganic as PbI2 and acid as FAH. TODO: generalize
    """
    inchis = {
        'organic': perovRow['_rxn_organic-inchikey'],
        'inorganic': 'RQQRAHKHDFPBMC-UHFFFAOYSA-L',
        'acid': 'BDAGIHXWWSANSR-UHFFFAOYSA-N'
    }
    
    speciesExperimentConc = perovRow[f"{'_rxn_M_' if v1 else '_rxn_v0-M_'}{chemtype}"]
    
    reagentConcPattern = f"_raw_reagent_[0-9]_{'v1-' if v1 else ''}conc_{inchis[chemtype]}"
    speciesReagentConc = perovRow.filter(regex=reagentConcPattern)
    
    if speciesExperimentConc == 0: 
        return speciesExperimentConc
    else: 
        return speciesExperimentConc / np.max(speciesReagentConc)

def createDensityDatasets(perov, experiment=0, propConc=False):
    """Return Perovskite Density datasets with _rxn, and _feat columns 
       for density hypothosis 0 (old) and 1 (new)
    
    :param curatedCols: List of Regular expressions that denote which columns 
                        you want like to add on top of _rxn, _feat. 
                        
    :returns: data0:  Dataframe for Density-Model0
              data1:  DataFrame for Density-Model1
              out:    Outcome binarized (crystal vs not-crystal)
              inchis: Series of _rxn_inchikeys for LeaveOneOut masking
    """
    model0C = '_rxn_v0-M_acid _rxn_v0-M_inorganic _rxn_v0-M_organic'.split(' ')
    model1C = '_rxn_M_acid _rxn_M_inorganic _rxn_M_organic'.split(' ')
    
    rxns0 = perov.filter(regex='_rxn_.*').drop(model1C, axis=1)
    
    rxns1 = perov.filter(regex='_rxn_.*').columns.tolist()

    for f in model0C:
        rxns0[f] = perov[f]
    rxns0 = rxns0.columns.tolist()

    
    feats = perov.filter(regex='_feat_.*').columns.tolist()
    model0 = rxns0 + feats
    model1 = rxns1 + feats

    logger.info(f'SolV Headers: {rxns0}')
    logger.info(f'SolUD+ Headers: {rxns1}')

    inchis = pd.DataFrame.from_dict({"inchis":perov['_rxn_organic-inchikey'].values})

    full_data = perov.drop('_rxn_organic-inchikey', axis=1)
    
    curatedCols = []
    if experiment == 1:
        curatedCols = ['_raw_reagent_.*_chemicals_.*_actual_amount$',
                       '_raw_*molweight', 
                       '_feat_vanderwalls_volume', 
                       '_raw_reagent_\d_volume']

        i = []
        for reg in curatedCols:
            i.extend(perov.filter(regex=reg, axis=1).columns.tolist()) 

        model0 = sorted(list(set(i + model0)))
        model1 = sorted(list(set(i + model1)))

    if curatedCols and experiment == 1:
        if '_raw_reagent_5_chemicals_2_actual_amount' in full_data.columns:
            # this has been verified by hand. This column is full of zeros and 'null9.'
            # and that's it. Therefore, nuking this column is okay. 
            full_data['_raw_reagent_5_chemicals_2_actual_amount'] = [0]*full_data.shape[0]

    inchi_counts = {inchi: inchi_df.shape[0] for inchi, inchi_df in inchis.groupby('inchis')}

    inchis = inchis["inchis"].reset_index(drop=True)
    
    model0 = list(set(model0) - set(['_rxn_organic-inchikey']))
    model1 = list(set(model1) - set(['_rxn_organic-inchikey']))
    
    if propConc:
        chemtypes = ['organic', 'inorganic', 'acid']
        for v1 in [False, True]:
            for chemtype in chemtypes:
                 full_data[f'_rxn_proportionalConc_{"v1-" if v1 else ""}{chemtype }'] = perov.apply(_computeProportionalConc, axis=1, v1=v1, chemtype=chemtype)    

        model0 += [f'_rxn_proportionalConc_{c}' for c in chemtypes]
        model1 += [f'_rxn_proportionalConc_v1-{c}' for c in chemtypes]
                           
        model0 = list(set(model0) - set(model0C))
        model1 = list(set(model1) - set(model1C))


    data0 = full_data[model0].reset_index(drop=True)
    data1 = full_data[model1].reset_index(drop=True)
    out = full_data['_out_crystalscore']

    # binarize class
    out = (out == 4).astype(int)
    
    def getRaws(regexes):
        i = []
        for r in regexes:
            i.extend(perov.filter(regex=r).columns)
        return i

    def cleanRaw(fn, v1):
        r2 = perov[getRaws(fn(v1))].reset_index(drop=True).fillna(0)
        return r2[perov['_out_crystalscore'] != 0].reset_index(drop=True)
    
    
    if experiment == 2:
        # These are described as "experiment" level features 
        # final_concentrations, reagent concentrations, reagent volumes
        raw2 = lambda v1: [f"_raw{'_v1-' if v1 else '_'}M_.*_final", 
                   f"_raw_reagent_\d{'_v1-' if v1 else '_'}conc_.*", 
                   "_raw_reagent_\d_volume"]

        data0 = pd.concat([data0, cleanRaw(raw2, False)], axis=1)
        data1 = pd.concat([data1, cleanRaw(raw2, True)], axis=1)
    
    elif experiment == 3:
        # This is more along the lines of "reagent" level features
        # These include all information related to reagents but exclude experimental properties
        raw3 = lambda v1: ['_raw_reagent_\d_volume$', 
                           f"_raw_reagent_\d{'_v1-' if v1 else '_'}conc.*"]
                           
        data0 = pd.concat([data0, cleanRaw(raw3, False)], axis=1)
        data1 = pd.concat([data1, cleanRaw(raw3, True)], axis=1)
        
        data0 = data0.drop(model0C, axis=1)
        data1 = data1.drop(model1C, axis=1)

    return data0, data1, out, inchis, inchi_counts, perov, full_data


# Get the code as described above.
def generateTrainingLists(perov, onehot):
    data0, data1, out, inchis, inchi_counts, perov, full_data = createDensityDatasets(perov, experiment=0)
    data0P, data1P, out, inchis, inchi_counts, perov, full_data = createDensityDatasets(perov,
                                                                                        experiment=0, 
                                                                                        propConc=True)
    data0_1, data1_1, out, inchis, inchi_counts, perov, full_data = createDensityDatasets(perov, experiment=1)
    data0_2, data1_2, out, inchis, inchi_counts, perov, full_data = createDensityDatasets(perov, experiment=2)
    data0_3, data1_3, out, inchis, inchi_counts, perov, full_data = createDensityDatasets(perov, experiment=3)
    model0C = ['_rxn_v0-M_acid', '_rxn_v0-M_inorganic','_rxn_v0-M_organic']
    noConc = data0.drop(model0C, axis=1)
    noConc_plus1 = data0_1.drop(model0C, axis=1)
    noConc_plus2 = data0_2.drop(model0C, axis=1)
    noConc_plus3 = data0_3#.drop(model0C, axis=1)

    all_data_dict = {'data0': data0,
                     'data1': data1,
                     'data0P': data0P,
                     'data1P': data1P,
                     'data0_1': data0_1,
                     'data1_1': data1_1,
                     'data0_2': data0_2,
                     'data1_2': data1_2,
                     'data0_3': data0_3,
                     'data1_3': data1_3,
                     'noConc': noConc,
                     'noConc_plus1': noConc_plus1,
                     'noConc_plus2': noConc_plus2,
                     'noConc_plus3': noConc_plus3
    }

    #if there are things we want to one hot do the rest
    if onehot['setting'] == 1: 
        logger.info('Removing _feats_ and one-hot-encoding inchikeys')
        X = perov[['_rxn_organic-inchikey']]
        hotone = OneHotEncoder(categories='auto')
        hotone.fit(X)
        columnnames =  ['_onehot_%s' % s for s in hotone.categories_[0]]
        hot_df = pd.DataFrame(hotone.transform(X).toarray(), columns = columnnames)
        # drop any of the entries we don't care about (if told)
        feat_cols = [x for x in perov.columns if '_feat_' in x]
        for k in all_data_dict:
            data = all_data_dict[k].drop(feat_cols, axis=1)
            data = pd.concat([data,hot_df], axis=1)
            all_data_dict[k] = data
            
    exps0 = [(all_data_dict['data0'], all_data_dict['data1']), 
             (all_data_dict['noConc_plus1'], all_data_dict['data1P'])]

    exps1 = [
        (all_data_dict['data0_1'], all_data_dict['data0_1']), 
        (all_data_dict['noConc'], all_data_dict['data1_1'])
    ]

    exps2 = [
        (all_data_dict['noConc_plus2'], all_data_dict['data0_2']), 
        (all_data_dict['data1_2'], all_data_dict['data0_3'])
    ]

    exps3 = [
        (all_data_dict['noConc_plus3'], all_data_dict['data1_3'])
    ]

    allexps = [exps0, exps1, exps2, exps3]

    titles0 = [
        'data0.v.data1', 
        'noConc.raw1.v.data1.pM'
    ]

    titles1 = [
        'data0.raw1.v.data0.raw1',
        'noConc.v.data1.raw1'
    ]

    titles2 = [
        'noConc.raw2.v.data0.raw2',
        'data1.raw2.v.data0.raw3'
    ]

    titles3 = [
        'noConc.raw3.v.data1.raw3'
    ]

    alltitles = [titles0, titles1, titles2, titles3]
    
    return allexps, alltitles, out, inchis, inchi_counts, perov, full_data

def std_norm_cols(X, stddict, normdict):
        #only stddict or normdict will have True, not both
        curated_list = []
        columnlist = list(X.columns)
        for header_prefix in stddict:
            if stddict[header_prefix] == 1:
                for column in columnlist:
                    if header_prefix in column:
                        curated_list.append(column)
        for header_prefix in normdict:
            if normdict[header_prefix] == 1:
                for column in columnlist:
                    if header_prefix in column:
                        curated_list.append(column)
        return curated_list


def compare_models(data0_dict, data1_dict, out, clf_in, inchis, stddict, normdict, cv=0, stratify=False):
    """Train test splits data0 and data1, trains and tests clf on both datsets, prints results
    :param data0_dict: dictionary of 'name' and 'data'.   
    :param data1_dict: dictionary of 'name' and 'data'.  Second for the comparisons
    :param out: outcome scores matched against the data0 and data 1
    :param clf: classifier dictionary including the 'estimator' 'opt: bool' and 'param grid' for hyperparameter tuning
    :param cv: 

    """

    
    outcomes = {}
    key = clf_in['estimator'].__str__()
    outcomes[key] = {}
    
    if stratify:
        data0_dict['data'], data1_dict['data'], out, _ = _stratify(data0_dict['data'], data1_dict['data'], out, inchis)
    
    def train_test_and_report(data, out, clf_dict, stddict, normdict, cv=0):
        if cv >= 2:
            # metrics to track
            def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
            def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
            def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
            def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
            def mcc(y_true, y_pred): return matthews_corrcoef(y_true, y_pred)
            def sup1(y_true, y_pred): return np.sum(y_true)
            def sup0(y_true, y_pred): return len(y_true) - np.sum(y_true)
            scoring = {'tp': make_scorer(tp), 
                       'tn': make_scorer(tn),
                       'fp': make_scorer(fp), 
                       'fn': make_scorer(fn), 
                       'sup_1': make_scorer(sup1),
                       'sup_0': make_scorer(sup0),
                       'mcc': make_scorer(mcc), 
                       'precision': 'precision', 
                       'recall': 'recall', 
                       'f1': 'f1'}
            # run cross_validation
            clf = clf_dict['estimator']
            X, X_test, y, y_test = train_test_split(data, out, test_size=0.2, random_state=42)
            
            #get list of colmns to norm / std
            curated_columns = std_norm_cols(X, stddict, normdict)

            # only apply processes to curated
            column_transform = make_column_transformer((StandardScaler(), curated_columns), remainder='passthrough')

            # build pipeline to convert particular columns with header prefixes to norm or std (user specified earlier)
            # Feed the resulting train and test data appropriately to GridCVmethod for optimization
            if 1 in normdict.values():
                column_transform = make_column_transformer((Normalizer(), curated_columns), remainder='passthrough')
                clf_pipe = Pipeline(steps=[('transform', column_transform), ('model', clf)])
            if 1 in stddict.values():
                column_transform = make_column_transformer((StandardScaler(), curated_columns), remainder='passthrough')
                clf_pipe = Pipeline(steps=[('transform', column_transform), ('model', clf)])
            else: 
                clf_pipe = Pipeline(steps=[('transform', None), ('model', clf)])

            if clf_dict['opt'] == 1:
                # rename for handling by gridsearch
                param_grid = {}
                for k, v in clf_dict['param_grid'].items():
                    param_grid[f'model__{k}'] =  v
                clf = GridSearchCV(clf_pipe, param_grid=param_grid, refit=True, cv=5, n_jobs=8)
                clf.fit(X, y)
                clf = clf.best_estimator_
            else:
                clf = clf_pipe
            
            #Assess model performance using optimized parameters from grid (if chosen) on same norm /std /onehot
            logger.info(f'Old error point for interpolation with stratified (debug): train: {data.shape} out: {out.shape}')
            scores = cross_validate(clf, data, out,
                                    cv=KFold(cv, shuffle=True),  #shuffle batched experimental data into descrete experiments
                                    scoring=scoring, 
                                    return_train_score=True,
                                    return_estimator=True)
            return scores, curated_columns

        else:
            clf = clf_dict['estimator']

            X, X_test, y, y_test = train_test_split(data, out, test_size=0.2, random_state=42)

            curated_columns = std_norm_cols(X, stddict, normdict)

            # build pipeline to convert particular columns with header prefixes to norm or std (user specified earlier)
            # Feed the resulting train and test data appropriately to GridCVmethod for optimization
            if 1 in normdict.values():
                column_transform = make_column_transformer((Normalizer(), curated_columns), remainder='passthrough')
                clf_pipe = Pipeline(steps=[('transform', column_transform), ('model', clf)])
            if 1 in stddict.values():
                column_transform = make_column_transformer((StandardScaler(), curated_columns), remainder='passthrough')
                clf_pipe = Pipeline(steps=[('transform', column_transform), ('model', clf)])
            else: 
                clf_pipe = Pipeline(steps=[('transform', None), ('model', clf)])

            if clf_dict['opt'] == 1:
                # rename for handling by gridsearch
                param_grid = {}
                for k, v in clf_dict['param_grid'].items():
                    # '__' (double underscore) specfies step to apply param grid to
                    param_grid[f'model__{k}'] =  v
                clf = GridSearchCV(clf_pipe, param_grid=param_grid, refit=True, cv=5, n_jobs=8)
                clf.fit(X, y)
                clf = clf.best_estimator_
            else:
                clf = clf_pipe

            pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, pred)
            cr = classification_report(y_test, pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, pred)
            matt_coeff = matthews_corrcoef(y_test, pred)
            return cm, cr, precision, recall, f1, support, matt_coeff, clf, curated_columns
        
    # END DEF
    ###############################################################################
    # Actual body of fn here. 
    if cv >= 2:
        for data_index, d_dict in enumerate([data0_dict, data1_dict]):
            outcomes[key][data_index] = {}

            d = d_dict['data']
            data_name = d_dict['name']

            cvScores, curated_columns = train_test_and_report(d, out, clf_in, stddict, normdict, cv=cv)
            clfs = cvScores['estimator']
            print(data_name,', stratify=',stratify, ', ', clfs, file=open('STTSModelInfo.txt', 'a'))

            cm = [cvScores[f'test_{i}'] for i in ['tp', 'fp', 'fn', 'tn']]
            cr = ""
            precision = cvScores['test_precision']
            recall = cvScores['test_recall']
            f1 = cvScores['test_f1']
            support_pos = cvScores['test_sup_1'] # positive class
            support_neg = cvScores['test_sup_0'] # negative class
            matthew_coef = cvScores['test_mcc']
            
            def extractCM(i, inputCm):
                # extract across fibers of tensor
                return np.array([[inputCm[0][i], inputCm[1][i]], 
                                 [inputCm[2][i], inputCm[3][i]]])

            for i in range(cv):
                # save info
                outcomes[key][data_index][i] = {}
                outcomes[key][data_index][i]['confMat'] = extractCM(i, cm)
                outcomes[key][data_index][i]['classRep'] = cr
                outcomes[key][data_index][i]['classifier'] = key.split("(")[0]
                outcomes[key][data_index][i]['dataIndex'] = data_index
                outcomes[key][data_index][i]['fold'] = i
                outcomes[key][data_index][i]['precision'] = [0, precision[i]]
                outcomes[key][data_index][i]['recall'] = [0, recall[i]]
                outcomes[key][data_index][i]['f1'] = [0, f1[i]]
                outcomes[key][data_index][i]['support'] = [support_neg[i], support_pos[i]]
                outcomes[key][data_index][i]['matthewCoef'] = matthew_coef[i]
                if key.split("(")[0] == "GradientBoostingClassifier":
                    x = clfs[i][1].feature_importances_

                    #reorder column headers from pipeline operations (report correctly!)
                    old_order = list(d.columns)
                    temp_headers = [col for col in old_order if col not in curated_columns]
                    # if no columns are selected for the pipeline, no columns will be moved
                    hold_curated = list(curated_columns)
                    hold_curated.extend(temp_headers)
                    hold_curated = np.array(hold_curated)

                    # sort descending [::-1]
                    outcomes[key][data_index][i]['featImportance'] = x[np.argsort(x)[::-1]]
                    outcomes[key][data_index][i]['orderFeatByImport'] = list(hold_curated[np.argsort(x)[::-1]])
                    
        df0 = pd.DataFrame.from_dict(outcomes[key][0], orient='index')
        df1 = pd.DataFrame.from_dict(outcomes[key][1], orient='index')
        df = pd.concat([df0, df1]).reset_index(drop=True)
        
        return df
    else:
        for data_index, d_dict in enumerate([data0_dict, data1_dict]):
            d = d_dict['data']
            data_name = d_dict['name']
            cm, cr, precision, recall, f1, support, matthew_coef, clf, curated_columns = train_test_and_report(d,
                                                                                                   out,
                                                                                                   clf_in,
                                                                                                   stddict,
                                                                                                   normdict)
            # save info
            print(data_name, ', stratify=',stratify, ', ', clf, file=open('STTSModelInfo.txt', 'a'))
            outcomes[key][data_index] = {}
            outcomes[key][data_index]['confMat'] = cm
            outcomes[key][data_index]['classRep'] = cr
            outcomes[key][data_index]['classifier'] = key.split("(")[0]
            outcomes[key][data_index]['dataIndex'] = data_index
            outcomes[key][data_index]['fold'] = cv
            outcomes[key][data_index]['precision'] = precision
            outcomes[key][data_index]['recall'] = recall
            outcomes[key][data_index]['f1'] = f1
            outcomes[key][data_index]['support'] = support
            outcomes[key][data_index]['matthewCoef'] = matthew_coef
            if key.split("(")[0] == "GradientBoostingClassifier":
                x = clf['model'].feature_importances_

                #reorder column headers from pipeline operations (report correctly!)
                old_order = list(d.columns)
                temp_headers = [col for col in old_order if col not in curated_columns]
                # if no columns are selected for the pipeline, no columns will be moved
                hold_curated = list(curated_columns)
                hold_curated.extend(temp_headers)
                hold_curated = np.array(hold_curated)

                # sort descending [::-1]
                outcomes[key][data_index]['featImportance'] = list(x[np.argsort(x)[::-1]])
                outcomes[key][data_index]['orderFeatByImport'] = list(hold_curated[np.argsort(x)[::-1]])
                    
            
        df = pd.DataFrame.from_dict(outcomes[key], orient='index')
        return df

def train_random_tts_models(classifiers, strat, data0, data1, out, inchis, stddict, normdict, cv=0):
    
    results = pd.DataFrame()

    for clf in classifiers:
        df = compare_models(data0, data1, out, clf, inchis, stddict, normdict, cv=cv, stratify=strat)
        results = pd.concat([results, df]).reset_index(drop=True)

    return results

def amine_split(data0, data1, out, inchis, stratify=False):
    """Generate leave one amine out TTS for each amine in inchis
    """
    if stratify is True:
        logger.info('uniformally sampling 96 experiments for each amine')
        data0, data1, out, indicies = _stratify(data0, data1, out, inchis)
        for amine in inchis.unique():
            is_amine = indicies[amine]
            train = list(set(range(data0.shape[0])) - set(indicies[amine]))
            X_test_0 = data0.iloc[is_amine]
            X_train_0 = data0.iloc[train]

            X_test_1 = data1.iloc[is_amine]
            X_train_1 = data1.iloc[train]

            y_test = out.iloc[is_amine].values.squeeze()
            y_train = out.iloc[train].values.squeeze()

            yield (X_train_0, X_test_0, y_train, y_test), (X_train_1, X_test_1, y_train, y_test)
    
    if stratify is False:
        logger.info('samples built from all experiments for each amine')
        for amine in inchis.unique():
            is_amine = (inchis == amine).values
            X_test_0 = data0[is_amine]
            X_train_0 = data0[~is_amine]

            X_test_1 = data1[is_amine]
            X_train_1 = data1[~is_amine]

            y_test = out[is_amine]
            y_train = out[~is_amine]

            yield (X_train_0, X_test_0, y_train, y_test), (X_train_1, X_test_1, y_train, y_test)

def amine_compare_models(fold, data_index, X, X_test, y, y_test, clf, inchi, data_columns):
    """Train test and report for one amine model with clf
    
    return outcomes dict
    """
    outcomes = {}
    key_flat = clf['model'].__str__()
    key = clf.__str__()
    outcomes[key] = {}
    
    # train a classifier on this data
    # get confusion matrix and classification report
    def train_test_and_report(X, X_test, y, y_test, clf):
        clf.fit(X, y)
        pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        cr = classification_report(y_test, pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, pred)
        matt_coeff = matthews_corrcoef(y_test, pred)
        return cm, cr, precision, recall, f1, support, matt_coeff
    
    #save data into a dict (eventually dataframe)
    cm, cr, precision, recall, f1, support, matthew_coef = train_test_and_report(X, 
                                                                                  X_test, 
                                                                                  y, 
                                                                                  y_test, 
                                                                                  clf)
    outcomes[key][data_index] = {}
    outcomes[key][data_index]['confMat'] = cm
    outcomes[key][data_index]['classRep'] = cr
    outcomes[key][data_index]['classifier'] = key.split("(")[0]
    outcomes[key][data_index]['dataIndex'] = data_index
    outcomes[key][data_index]['inchi'] = inchi
    outcomes[key][data_index]['precision'] = precision
    outcomes[key][data_index]['recall'] = recall
    outcomes[key][data_index]['fold'] = fold
    outcomes[key][data_index]['f1'] = f1
    outcomes[key][data_index]['support'] = support
    outcomes[key][data_index]['matthewCoef'] = matthew_coef
    
    if key_flat.split("(")[0] == "GradientBoostingClassifier":
        x = clf['model'].feature_importances_

        # columns are reorganized during pipeline (clf)
        old_order = list(X.columns)
        temp_headers = [col for col in old_order if col not in data_columns]
        # if no columns are selected for the pipeline, no columns will be moved
        hold_curated = list(data_columns)
        hold_curated.extend(temp_headers)
        hold_curated = np.array(hold_curated)

        outcomes[key][data_index]['featImportance'] = x[np.argsort(x)[::-1]]
        outcomes[key][data_index]['orderFeatByImport'] = hold_curated[np.argsort(x)[::-1]]
            
    return outcomes

def train_all_loocv_models(classifiers, data0_dict, data1_dict, out, inchis, cv=1, stratify=False):
    """Get LOOCV results for each inchikey

    :param classifiers: list of sci-kit learn classifiers
    :param data0: first data sample to be compared
    :param data1: second data sample to be compared
    :param inchis: list of inchi keys determined by the sample


    """
    data0 = data0_dict['data']
    data1 = data1_dict['data']

    # returns list of data0 and data1 (xtrain, xtest, ytrain,ytest) for each unique amine
    mycv_return = list(amine_split(data0, data1, out, inchis, stratify=stratify))
    # repack list for gridsearhCV 
    testtrain_index = []
    testtrain_0 = []
    testtrain_1 = []
    for tts0, tts1 in mycv_return: #tts0 and tts1 are split the same way, just need indexes so one list suffices
        trainindexes_0 = tts0[0].index.values
        testindexes_0 = tts0[1].index.values
        testtrain_index.append([trainindexes_0, testindexes_0])
        testtrain_0.append(tts0)
        testtrain_1.append(tts1)


    # this is ugly as the original code was developed from the pretense of comparison.  
    # We have to carry two unique models through LOO after CV to ensure the best hyperparameters are used
#    X_train, X_test, y_train, y_test = train_test_split(data, out, test_size=0.2, random_state=42)
    clf_run_dict = {}
    for clf_dict in classifiers:
        clf = clf_dict['estimator']
        clf_run_dict['data0_clf'] = clf #each dataset / gridCV model may be optimized differently, this stores both independent (pipelines)
        clf_run_dict['data1_clf'] = clf

        #get list of column *headers* to norm / std does not dictate which rows (indexes) are used
        # pipeline handles rows (any example of the datastructure is sufficient)
        curated_columns = std_norm_cols(data0, stddict, normdict)
        curated_columns_2 = std_norm_cols(data1, stddict, normdict)
        clf_run_dict['data0_column'] = curated_columns
        clf_run_dict['data1_column'] = curated_columns_2

        # build pipeline to convert particular columns with header prefixes to norm or std (user specified earlier)
        # Feed the resulting train and test data appropriately to GridCVmethod for optimization
        if 1 in normdict.values():
            column_transform = make_column_transformer((Normalizer(), clf_run_dict['data0_column']), remainder='passthrough')
            column_transform_2 = make_column_transformer((Normalizer(), clf_run_dict['data1_column']), remainder='passthrough')
            clf_pipe = Pipeline(steps=[('transform', column_transform), ('model', clf_run_dict['data0_clf'])])
            clf_pipe_2 = Pipeline(steps=[('transform', column_transform_2), ('model', clf_run_dict['data1_clf'])])
        elif 1 in stddict.values():
            column_transform = make_column_transformer((StandardScaler(), clf_run_dict['data0_column']), remainder='passthrough')
            column_transform_2 = make_column_transformer((StandardScaler(), clf_run_dict['data1_column']), remainder='passthrough')
            clf_pipe = Pipeline(steps=[('transform', column_transform), ('model', clf_run_dict['data0_clf'])])
            clf_pipe_2 = Pipeline(steps=[('transform', column_transform_2), ('model', clf_run_dict['data1_clf'])])
        else: 
            clf_pipe = Pipeline(steps=[('transform', None), ('model', clf_run_dict['data0_clf'])])
            clf_pipe_2 = Pipeline(steps=[('transform', None), ('model', clf_run_dict['data1_clf'])])

        if clf_dict['opt'] == 1:
            # rename for handling by gridsearch
            param_grid = {}
            for k, v in clf_dict['param_grid'].items():
                param_grid[f'model__{k}'] =  v

            clf_new_1 = GridSearchCV(clf_pipe, param_grid=param_grid, refit=True, cv=testtrain_index, n_jobs=8)
            clf_new_1.fit(data0, out)
            clf_1 = clf_new_1.best_estimator_    
            clf_run_dict['data0_clf'] = clf_1
            print(data0_dict['name'], ', stratify=',stratify, ', ',  clf_1, file=open('LOOModelInfo.txt', 'a'))

            clf_2 = GridSearchCV(clf_pipe_2, param_grid=param_grid, refit=True, cv=testtrain_index, n_jobs=8)
            clf_2.fit(data1, out)
            clf_2 = clf_2.best_estimator_    
            clf_run_dict['data1_clf'] = clf_2
            print(data1_dict['name'], ', stratify=',stratify, ', ',  clf_2, file=open('LOOModelInfo.txt', 'a'))
        else:
            clf_run_dict['data0_clf'] = clf_pipe
            print(data0_dict['name'], ', stratify=',stratify, ', ',  clf_pipe, file=open('LOOModelInfo.txt', 'a'))
            clf_run_dict['data1_clf'] = clf_pipe_2
            print(data1_dict['name'], ', stratify=',stratify, ', ',  clf_pipe_2, file=open('LOOModelInfo.txt', 'a'))

    
    results = pd.DataFrame()
    # hard code this, we aren't going to use more than 1 model at a time moving forward
#    clf_dict = classifiers[0]
    for fold in range(cv):
        for inchi, leave_one_amine_split in zip(inchis.unique(), mycv_return):
            # for each dataset (model0, model1 concentrations) 
            for conc_model_i, (X_train, X_test, y_train, y_test) in enumerate(leave_one_amine_split):
                print(f'data{conc_model_i} with columns:', X_train.columns, file=open('LOOModelInfo_dataset.txt', 'a'))
                run_clf = clf_run_dict[f'data{conc_model_i}_clf']
                naming_columns = clf_run_dict[f'data{conc_model_i}_column']
                res = amine_compare_models(fold, conc_model_i, X_train, X_test, y_train, y_test, run_clf, inchi, naming_columns)
                df = pd.DataFrame.from_dict(res[run_clf.__str__()], orient='index')
                results = pd.concat([results, df]).reset_index(drop=True)

    return results

def get_success_scores(results): 
    for col in ['precision', 'recall', 'f1']:
        results[col + '_success'] = results[col].apply(lambda x: x[1])
    
    results['matthewCoef_success'] = results['matthewCoef']
    return results

def translate_inchi_key(results):
    return results['inchi'].apply(lambda x: INCHI_TO_CHEMNAME[str(x)])

def cleanResults(df, inchis, inchi_counts):
    df['classifier'] = df['classifier'].str.split("(").apply(lambda x: x[0])
    lens_c = [n + " -- (" + str(inchis.shape[0] - inchi_counts[i]) + ", " + str(inchi_counts[i]) + ")" 
              for i, n in zip(df['inchi'], 
                              df['Chemical Name'])
             ]
    df['ChemicalName -- (train, test)'] = lens_c
    return df

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def generateResults(classifiers, perov, stddict, normdict, onehot, cv=2, extrpl=1, intrpl=1, plot_results=0):
    """Run the entire notebook. Generate results for the comparions enumerated in `genearteTrainingLists()`

    :param classifiers: type=dict classifiers with 'estimators' 'opt' (hyperparameter or no) and 'param grid'
    :param cv: number of cross validation folds to be used (adds about 30 minutes to each run)
    :param extrpl: bool, default=1, disable extrapolitive comparisons with 0
    :param intrpl: bool, default=1, disable interpolitive comparisons with 0
    :param plot_results: bool, default=0, enable plots with 1

    :return: None
    """
    print('Generating training and testing data samples')
    expList, titleList, out, inchis, inchi_counts, perov, full_data = generateTrainingLists(perov, onehot)
    # for each experiment list and title list
    for exps, titles in zip(expList, titleList):
        # for paired: dataset_i, dataset_j, title_i_j 
        for (d0_data, d1_data), title in zip(exps, titles):
            # for results stratified vs not-stratified
            os.mkdir(f'./results/{title}')
            print(f'Working on {title}')
            d0 = {}
            d1 = {}
            d0['name'] = title.split('v')[0]
            d1['name'] = title.split('v')[1]
            d0['data'] = d0_data
            d1['data'] = d1_data
            for strat in [False, True]:
                # extrapolative and Leave-one-amine-out mean the same
                if extrpl == 1:
                    logger.info('extrapolative training: {}'.format(title))
                    logger.info('stratified: {}'.format(strat))
                    #TODO: ADD stddict and normdict functions
                    results = train_all_loocv_models(classifiers,
                                                     data0_dict=d0, 
                                                     data1_dict=d1, 
                                                     out=out, 
                                                     inchis=inchis, 
                                                     cv=cv, 
                                                     stratify=strat)
                    # clean up results some and 
                    results['Chemical Name'] = translate_inchi_key(results)
                    # remove degenerate experiments
                    # get scores for the class we care about (recall on whether or not we made a crystal)
                    results = get_success_scores(results)
                    # remove excess information; create LOO-id with train/test information
                    results = cleanResults(results, inchis, inchi_counts)
                    if strat:
                        results.to_csv(f'./results/{title}/results_strat_{title}_2019.08.20.csv', index=False)
                    else:
                        results.to_csv(f'./results/{title}/results_{title}_2019.08.20.csv', index=False)
                    # copy the results dataframe and jitter the values so that 
                    #  they can be seen in the pictures we make
                    r = results.copy()
                    r['precision_success'] = rand_jitter(results['precision_success'])
                    r['recall_success'] = rand_jitter(results['recall_success'])
                    r['f1_success'] = rand_jitter(results['f1_success'])
                    r['matthewCoef_success'] = rand_jitter(results['matthewCoef_success'])

                if intrpl == 1: 
                    logger.info('interpolative training: {}'.format(title))
                    interpolativeResults = train_random_tts_models(classifiers, 
                                                                   strat,
                                                                   d0, 
                                                                   d1, 
                                                                   out, 
                                                                   inchis,
                                                                   stddict,
                                                                   normdict,
                                                                   cv=cv) 
                    interpolativeResults = get_success_scores(interpolativeResults)
                    # if startified, say so in the filename
                    if strat:
                        interpolativeResults.to_csv(f'./results/{title}/random_strat_TTS_results.csv', index=False)
                    else:
                        interpolativeResults.to_csv(f'./results/{title}/randomTTS_results.csv', index=False)

                # make a bunch of pictures (and save them) for each cv fold. 
                # in old versions of the code.  not all that useful... so removed
                #if plot_results == 1:
                #    for fold in range(cv):
                #        cool_plot(r, measure='precision', title=title, fold=fold, strat=strat)
                #        cool_plot(r, measure='recall', title=title, fold=fold, strat=strat)
                #        cool_plot(r, measure='f1', title=title, fold=fold, strat=strat)
                #        cool_plot(r, measure='matthewCoef', title=title, fold=fold, strat=strat)



if __name__ == '__main__':
    ''' Toggles for controlling various run parameters.  Useful for reproducing reported results

    The available options to change are outlined 
    Note: One run of the loop takes about 30 minutes on a standard machine.  Increasing cv increases by about 30 minutes per fold. 
    '''
    if os.path.isdir('./results'):
        # results folder must be empty!!
        print('Move or remove results folder before starting')
    os.mkdir('./results')

    # DATASETS #
    ## Check datasets enabled in `generateTrainingLists` each dataset will generate a folder in results
    ## To remove datasets, comment out the dataset 'exp' list and title list options :w

    normdict = {}
    stddict = {}
    onehot = {}
    # MODELING OPTIONS #
    cv = 6              # Number of cross validation folds (For all data) after hyperparameter tuning 
                        ## See line approx 747 for more gridsearchCV cv values
    extrpl = 1          # Run extrapolative models (Leave one amine/organic out)
    intrpl = 1          # Run interpolative models (Standard test train splits )
    plot_results = 0    # Turn mass plotting on (plots each of the model comparisons to png images)
    shuffle = 0         # Shuffles correspondence between x and y  (cannot be used with deep shuffle or unshuffled data)
    deep_shuffle = 0    # Shuffles each column of the dataset independently (cannot be used concurrently with data_shuffle or unshuffled data)
    hyperparam_opt = 1  # Enables gridsearch_CV, takes a LONG time (depending on parameters specified)
    # Enable only norm or std (not both at the same time... makes no sense) any combination of norm0-2 is fine same with std0-2
    normdict['_feat_'] = 0    # Forces normalization of _feats_* 
    normdict['_rxn_'] = 0     # Forces normalization of _rxns_* 
    normdict['_raw_'] = 0     # Forces normalization of _raws_* (post calculations of proportional values)
    stddict['_feat_'] = 0     # Forces standardization of _feats_* 
    stddict['_rxn_'] = 0      # Forces standardization of _rxns_* 
    stddict['_raw_'] = 0      # Forces standardization of _raws_* (post calculations of proportional values)
    # for no one hot set drop=0 and empty the keep list
    # Specifically for analyzing how stts models are handling the _feats_
    onehot['setting'] = 0 # CAUTION DROPS _feats_ from dataframe leaves only encoding from organic inchikey
    
    # MODEL SPECIFICATION #
    #### KNN
    knn = KNeighborsClassifier(leaf_size=30, metric='minkowski',
                      metric_params=None, n_jobs=8, p=20)
    knn_param_grid = {'weights': ['uniform','distance'],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                      'n_neighbors': range(3,9,2)
    }
    knn_dict = {'estimator': knn,
                'opt': hyperparam_opt,
                'param_grid': knn_param_grid
    }
    #### GBT
    #default based on gbt testing, but will be ignored if hyperparam_opt = 1
    gbt = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=6,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=4, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='auto',
                           random_state=42, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
    # uncomment next line to overwrite gbt and reset to default
    gbt = GradientBoostingClassifier(random_state=42) 
    gbt_param_grid = {'min_samples_split': range(2,10,2),
                      'min_samples_leaf': range(2,7,2),
                      'n_estimators': [100,500],
                      'max_depth': range(2,7,2),
                      'learning_rate': [0.05,0.10,0.15,0.20]
    }
    gbt_dict = {'estimator': gbt,
                'opt': hyperparam_opt,
                'param_grid': gbt_param_grid
    }

    #### LSVC
    lSVC = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
          verbose=0)
    lSVC_param_grid = {'tol': [0.001, 0.0001, 0.00001],
                 'dual': [True, False],
                 'fit_intercept': [True, False],
                 'loss': ['hinge', 'squared_hinge'],
                 'penalty':['l1','l2']
    }
    lSVC_dict = {'estimator': lSVC,
                'opt': hyperparam_opt,
                'param_grid': lSVC_param_grid
    }
    #### SVC
    svmSVC = SVC(C=1.0, gamma='auto', coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200, 
                 class_weight=None, verbose=False, max_iter=-1, 
                 random_state=42)
    svmSVC_param_grid = {'kernel': ['poly', 'rbf'],
                         'degree': [2,3,4],
    }
    svmSVC_dict = {'estimator': svmSVC,
                   'opt': hyperparam_opt,
                   'param_grid': svmSVC_param_grid
    }

    # enter the dictionary of the model of choice here
    classifiers = [svmSVC_dict]

    if len(classifiers) > 1:
        print('We tried to automate everything, but we can only really do 1 classifier at a time.')
        print('Please use one classifier_dictionary per folder per group of user defined settings (i.e. norms, stds, etc).')  
        print('That will enable data workup, else good luck!')
        sys.exit()

    if 1 in normdict.values() and 1 in stddict.values():
        print('Only specify either normalization or standarization, not both')
        sys.exit()
    for i in normdict:
        logger.info(f'Std/norm dict: {i}, {normdict[i]}') 
    for i in stddict:
        logger.info(f'Std/norm dict: {i}, {stddict[i]}') 

    logger.info(f"Run Parameters: cv:{cv}, extrapolation:{extrpl}, interpolation:{intrpl}")
    logger.info(f"Run Parameters: shuffle:{shuffle}, deep shuffle:{deep_shuffle}, hyperparamopt:{hyperparam_opt}")
    logger.info(f"Run Parameters: Classifier: {classifiers}")

    # PIPELINE START # 
    print("Starting Run")
    perov = _prepare(shuffle, deep_shuffle)
    print("Dataset successfully parsed")
    generateResults(classifiers, perov, stddict, normdict, onehot, cv, extrpl, intrpl, plot_results)
    print('Run log written to AnalysisLog.txt')
