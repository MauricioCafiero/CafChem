from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint
from skfp.preprocessing import MolFromSmilesTransformer
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
from skfp.fingerprints import RDKitFingerprint 
from skfp.fingerprints import AtomPairFingerprint
from skfp.fingerprints import MordredFingerprint, RDKit2DDescriptorsFingerprint
from skfp.fingerprints import MACCSFingerprint, PubChemFingerprint
from skfp.fingerprints import FunctionalGroupsFingerprint
from skfp.preprocessing import ConformerGenerator
from skfp.fingerprints import (
    AutocorrFingerprint,
    E3FPFingerprint,
    MORSEFingerprint,
    RDFFingerprint,
)
from skfp.model_selection import scaffold_train_test_split, FingerprintEstimatorGridSearch
from skfp.distances import fraggle_distance, mcs_distance, bulk_fraggle_distance, bulk_mcs_distance
from skfp.filters import BeyondRo5Filter, LipinskiFilter, BMSFilter, BrenkFilter
from skfp.filters import GlaxoFilter, PfizerFilter, SureChEMBLFilter, ZINCBasicFilter
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

def clean_smiles(smiles_list: list[str]):
  '''
  '''
  ions_to_clean = ['[Na+].', '[Cl-].', '[Ca+].', '[K+].', '.[Na+]', '.[Cl-]', '.[Ca+]', '.[K+]', '[Br-].', '[I-].', '[F-].',
                   '.[Br-]', '.[I-]', '.[F-]']
  clean_smiles = []
  for smiles in smiles_list:
    for ion in ions_to_clean:
      smiles = smiles.replace(ion,"")
    clean_smiles.append(smiles)
  return clean_smiles
    

class get_fingerprints():
  '''
  '''
  def __init__(self, smiles_list: list[str], target_list: list[float], transform_flag: bool, n_jobs=-1):
    '''
    '''
    self.smiles_list = clean_smiles(smiles_list)
    self.target_list = target_list
    self.transform_flag = transform_flag
    self.n_jobs = n_jobs

    if transform_flag:
      self.y = np.log10(target_list)
    else:
      self.y = target_list
    
    mols_from_smiles = MolFromSmilesTransformer()
    self.mols_raw = mols_from_smiles.transform(smiles_list)

    print(f'Welcome to the CafChem SKFP generator. The types of 2D fingerprints available are:')
    print(f'1. ECFP')
    print(f'2. Atom_Pair')
    print(f'3. Mordred')
    print(f'4. RDKit_2D')
    print(f'5. MACCS')
    print(f'6. PubChem')
    print(f'7. Functional_Groups')
    print(f'8. RDKitFingerprint')
    print('And the 3D fingerprints available are:')
    print(f'9. E3FP')
    print(f'10. Autocorr')
    print(f'11. MORSE')
    print(f'12. RDF')
    print('To create fingerprints, call the create method of this class with the FP type as the first arguement.')
    print('To perform a FP parameter gridsearch, call the get_grid method with the number of the FP as the only argument, \
then call fp_gridsearch with a fp type, a model, a model grid and the fp grid as arguements.')

    self.types_3d = ['E3FP', 'Autocorr', 'MORSE', 'RDF']
    self.types_2d = ['ECFP', 'Atom_Pair', 'Mordred', 'RDKit_2D', 'MACCS', 'PubChem']
    self.fp_hash = {
        'ECFP': ECFPFingerprint(n_jobs=self.n_jobs),
        'Atom_Pair': AtomPairFingerprint(n_jobs=self.n_jobs),
        'Mordred': MordredFingerprint(n_jobs=self.n_jobs),
        'RDKit_2D': RDKit2DDescriptorsFingerprint(n_jobs=self.n_jobs),
        'MACCS': MACCSFingerprint(n_jobs=self.n_jobs),
        'PubChem': PubChemFingerprint(n_jobs=self.n_jobs),
        'Functional_Groups': FunctionalGroupsFingerprint(n_jobs=self.n_jobs),
        'RDKitFingerprint': RDKitFingerprint(n_jobs=self.n_jobs),
        'E3FP': E3FPFingerprint(n_jobs=self.n_jobs),
        'Autocorr': AutocorrFingerprint(n_jobs=self.n_jobs),
        'MORSE': MORSEFingerprint(n_jobs=self.n_jobs),
        'RDF': RDFFingerprint(n_jobs=self.n_jobs)
    }
  
  def get_grid(self, fp_int: int):
    '''
    '''
    self.fp_int = fp_int

    grid_list = [
        {'fp_size': [1024, 2048], 'radius': [2, 4, 6]},
        {'fp_size': [1024, 2048], 'min_distance': [1, 2, 3], 'max_distance': [20,30,40]},
        {'use_3D': [True, False]},
        {'normalized': [True, False]},
        {},
        {},
        {},
        {'fp_size': [1024, 2048], 'max_path': [5, 7, 9]},
        {'fp_size': [1024, 2048], 'radius_multiplier': [1.1, 1.1718, 3.0]},
        {'use_3D': [True, False]},
        {},
        {}
    ]

    return grid_list[fp_int-1]

  def create(self, fp_type: str, many_conf = True, num_confs = 5):
    '''
    '''
    self.fp_type = fp_type
    self.many_conf = many_conf
    self.num_confs = num_confs

    if self.fp_type in self.types_3d:
      if self.many_conf:
        conformer_gen = ConformerGenerator(num_conformers=self.num_confs, optimize_force_field="UFF", n_jobs=self.n_jobs)
      else:
        conformer_gen = ConformerGenerator(n_jobs=self.n_jobs)
      
      self.mols = conformer_gen.transform(self.mols_raw)
    else:
      self.mols = self.mols_raw
    
    fp = self.fp_hash[self.fp_type]

    self.mols_train, self.mols_test, self.y_train, self.y_test = scaffold_train_test_split(
        self.mols, self.y, test_size=0.2)
  
    self.X_train = fp.transform(self.mols_train)
    self.X_test = fp.transform(self.mols_test)

    print(f'Fingerprints created. The feature array size per molecule is: {self.X_train.shape[1]}')

    return self.X_train, self.X_test, self.y_train, self.y_test
  
  def fp_gridsearch(self, fp_type: str, model, model_grid, fp_grid, many_conf = True, num_confs = 5):
    '''
    '''
    self.fp_type = fp_type
    self.model = model
    self.model_grid = model_grid
    self.fp_grid = fp_grid
    self.many_conf = many_conf
    self.num_confs = num_confs

    fp = self.fp_hash[self.fp_type]

    fp_params = fp_grid # {"fp_size": [1024, 2048], "max_path": [5, 7, 9]}

    gs_cv = GridSearchCV(
        estimator=self.model,
        param_grid=self.model_grid,
    )

    fp_cv = FingerprintEstimatorGridSearch(fp, self.fp_grid, gs_cv)

    if self.fp_type in self.types_3d:
      if self.many_conf:
        conformer_gen = ConformerGenerator(num_conformers=self.num_confs, optimize_force_field="UFF", n_jobs=self.n_jobs)
      else:
        conformer_gen = ConformerGenerator(n_jobs=self.n_jobs)
      
      self.mols = conformer_gen.transform(self.mols_raw)
      self.mols_train, self.mols_test, self.y_train, self.y_test = scaffold_train_test_split(
          self.mols, self.y, test_size=0.2)
      fp_cv.fit(self.mols_train, self.y_train)
      y_pred = fp_cv.predict(self.mols_test)
      r2 = r2_score(self.y_test, y_pred)
    else:
      self.smiles_train, self.smiles_test, self.y_train, self.y_test = scaffold_train_test_split(
          self.smiles_list, self.y, test_size=0.2)
      fp_cv.fit(self.smiles_train, self.y_train)
      y_pred = fp_cv.predict(self.smiles_test)
      r2 = r2_score(self.y_test, y_pred)

    print(f"R2 score for best estimator: {r2:.3f}")
    print(f"Best FP parameters: {fp_cv.best_fp_params_}")
    print(f"CV results: {fp_cv.cv_results_}")

    return r2
  
  def get_distance_types(self):
    '''
    '''
    print('Distance types available include:')
    print('1. Fraggle_distance') #Computes the Fraggle distance between two RDKit Mol objects by subtracting similarity value from 1
    print('2. MCS_distance.') #Computes the Maximum Common Substructure (MCS) distance between two RDKit Mol objects by subtracting similarity value from 1.
    print('3. Bulk Fraggle.')
    print('4. Bulk MCS.')
    print('Call the binary_distances method with the distance type.')


  def get_distances(self, distance_type: str, smiles_list: list[str]):
    '''
    '''
    self.distance_type = distance_type
    if 'bulk' in self.distance_type.lower():
      bulk_flag = True
    else:
      bulk_flag = False
    
    distance_hash = {
        'Fraggle_distance': fraggle_distance,
        'MCS_distance': mcs_distance,
        'Bulk_Fraggle': bulk_fraggle_distance,
        'Bulk_MCS': bulk_mcs_distance
    }

    smiles_list = clean_smiles(smiles_list)
    mols = MolFromSmilesTransformer().transform(smiles_list)

    if not bulk_flag:
      dist = distance_hash[self.distance_type](mols[0], mols[1])
    else:
      dist = distance_hash[self.distance_type](mols)

    return dist

class filters():
  '''
  '''
  def __init__(self, smiles_list: list[str]):
    '''
    '''
    self.smiles_list = clean_smiles(smiles_list)

    print('Available filters include:')
    print('1. Lipinski: filters molecules that obey Lipinskis rules')
    print('2. BeyondRo5: filters molecules that obey BeyondRo5 rules')
    print('3. BMS: aims to remove molecules containing certain functional groups to filter out random noise, “promiscuous” compounds, and frequent hitters.')
    print('4. Brenk: Designed to filter out molecules containing substructures with undesirable pharmacokinetics or toxicity')
    print('5. Glaxo: Designed at Glaxo Wellcome (currently GSK) to filter out molecules with reactive functional groups, unsuitable leads (i.e. compounds which would not be initially followed up), and unsuitable natural products (i.e., derivatives of natural product compounds known to interfere with common assay procedures).')
    print('6. Pfizer: Based on observation that compounds exhibiting low partition coefficient (clogP) and high topological polar surface area (TPSA) are roughly 2.5 times more likely to be free of toxicity issues in the tested conditions')
    print('7. SureChEMBL: Based on structural alerts, i.e. toxicophores. Filters out compounds likely to be toxic.')
    print('8. ZINC: Designed to keep only drug-like molecules, removing molecules with unwanted functional groups.')

    self.filt_hash = {
      'Lipinski': LipinskiFilter(),
      'BeyondRo5': BeyondRo5Filter(),
      'BMS': BMSFilter(),
      'Brenk': BrenkFilter(),
      'Glaxo': GlaxoFilter(),
      'Pfizer': PfizerFilter(),
      'SureChEMBL': SureChEMBLFilter(),
      'ZINC': ZINCBasicFilter()
    }

  def get_filter(self, filter_name: str):
    '''
    '''
    self.filter_name = filter_name

    filt = self.filt_hash[self.filter_name]

    return filt.transform(self.smiles_list)