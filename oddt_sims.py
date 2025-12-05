from rdkit.Chem import AllChem, rdDepictor, rdDistGeom
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_3d = True
from rdkit.Chem import Draw

from rdkit import rdBase
from rdkit.Chem import rdMolAlign
import py3Dmol
import os
import numpy as np
from rdkit import RDConfig
from rdkit.Chem.Features.ShowFeats import _featColors as featColors
from rdkit.Chem.FeatMaps import FeatMaps

fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef'))

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

rdDepictor.SetPreferCoordGen(True)

import oddt
from oddt.interactions import (close_contacts,
                               hbonds,
                               distance,
                               halogenbonds,
                               halogenbond_acceptor_halogen,
                               pi_stacking,
                               salt_bridges,
                               pi_cation,
                               hydrophobic_contacts)

from oddt.shape import usr, usr_cat, electroshape, usr_similarity

class conformers():
  '''
    Class for generating conformers of a list of molecules
  '''
  def __init__(self,smiles: list, num_confs: int):
    '''
      read in a SMILES string and generate num_confs of conformers

      Args:
        smiles: list of SMILES strings
        num_confs: number of conformers to generate
      Returns:
        None
    '''
    self.smiles = smiles
    self.num_confs = num_confs

  def get_confs(self, use_random = False):
    '''
      Generates the conformers with or without using random coordinates.

      Args:
        use_random: boolean, whether to use random coordinates or not
      Returns:
        the embedded molecule objects containing the conformers
    '''
    ps = AllChem.ETKDGv3()
    ps.randomSeed=0xf00d
    ps.numThreads = 2
    if use_random == True:
      ps.useRandomCoords = True

    mols = [Chem.MolFromSmiles(x) for x in self.smiles]
    mols = [Chem.AddHs(m) for m in mols]

    embedded_mols = []
    for i,m in enumerate(mols):
      if not (i+1)%10:
        print(f'Doing {i+1} of {len(mols)}')
      m = Chem.Mol(m)
      AllChem.EmbedMultipleConfs(m,self.num_confs,ps)
      embedded_mols.append(m)
    
    self.embedded_mols = embedded_mols

    return self.embedded_mols
  
  def make_ref_conf(self, ref_smiles: str, folder = '', ref_file_base = None, use_random = False):
    '''
      prepares the reference by generating conformers and writing them to a file.

      Args:
        ref_smiles: SMILES string of the reference molecule
        use_random: boolean, whether to use random coordinates or not
        folder: location for ref conformer files
        ref_file_base: base name for ref conformer files
      Returns:
        none; writes files
    '''
    if folder != '':
      if not os.path.exists(folder):
        os.makedirs(folder)
    #if the folder exists, remove files only, make sure not to remove folders
    if os.path.exists(folder):
      for file in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, file)):
          os.remove(os.path.join(folder, file))

    ps = AllChem.ETKDGv3()
    ps.randomSeed=0xf00d
    ps.numThreads = 2
    if use_random == True:
      ps.useRandomCoords = True

    mol = Chem.MolFromSmiles(ref_smiles)
    molH = Chem.AddHs(mol)

    m = Chem.Mol(molH)
    AllChem.EmbedMultipleConfs(m,self.num_confs,ps)
    embedded_mols = m

    expanded_confs = []
    for i in range(embedded_mols.GetNumConformers()):
      m = Chem.Mol(embedded_mols,confId=i)
      expanded_confs.append(m)

    if ref_file_base == None:
      sdf_file_base = folder + '/ref_sdf'
    sdf_file_base = ref_file_base.replace('.sdf','')

    for i, mol in enumerate(expanded_confs):
      ref_file  = f'{folder}/{ref_file_base}_{i}.sdf'
      writer = Chem.SDWriter(ref_file)
      writer.write(mol)
      writer.close()
    
    self.ref_location = folder
    self.ref_base = ref_file_base

  def expand_all_confs(self):
    '''
      Expands all molecule objects by extracting each conformer into it's own molecule object.

      Args:
        None
      Returns:
        A list of the expanded molecule objects containing the conformers
    '''
    expanded_confs = []
    for mol_idx in range(len(self.embedded_mols)):
      temp_list = []
      for i in range(self.embedded_mols[mol_idx].GetNumConformers()):
        m = Chem.Mol(self.embedded_mols[mol_idx],confId=i)
        temp_list.append(m)
      expanded_confs.append(temp_list)
    self.expanded_confs = expanded_confs[0]

    return expanded_confs
  
  def xyz_to_sdf(self, folder = '', sdf_file_base = None):
    '''
      Takes and xyz file and produces an sdf file.

        Args:
          xyz_file: file to process
          folder: location for sdf files
          sdf_file_base (optional): name for sdf file
        Returns:
          None; writes file
    '''
    if folder != '':
      if not os.path.exists(folder):
        os.makedirs(folder)
    #if the folder exists, remove files only, make sure not to remove folders
    if os.path.exists(folder):
      for file in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, file)):
          os.remove(os.path.join(folder, file))

    if sdf_file_base == None:
      sdf_file_base = folder + 'new_sdf'
    sdf_file_base = sdf_file_base.replace('.sdf','')

    for i, mol in enumerate(self.expanded_confs):
      sdf_file  = f'{folder}/{sdf_file_base}_{i}.sdf'
      writer = Chem.SDWriter(sdf_file)
      writer.write(mol)
      writer.close()
    
    self.sdf_location = folder
    self.sdf_base = sdf_file_base

  def sim_function(self, m, comp_type: str):
    '''
    applies a particular similarity function to the mol object

    Args:
      m: mol object 
      comp_type: type of similarity function to use
    
    Returns:
      the similarity function applied to the mol object

    '''
    if comp_type == 'usr':
      return usr(m)
    elif comp_type == 'usr_cat':
      return usr_cat(m)
    elif comp_type == 'electroshape':
      return electroshape(m)

  def compare_mols(self, comp_type: str):
    ''' 
    Reads in previously created SDF files for reference and test molecule conformers
    and calculates similarity of each pair using one of three similarity functions.
    USR: Ultrafast shape recognition
    USR_CAT: USR with pharmacophoric constraints
    ELECTROSHAPE: fast molecular similarity calculations incorporating shape, chirality and electrostatics

    Args:
      comp_type: type of similarity function to use
    
    '''
    path = self.sdf_location
    files = os.listdir(path)            
    filenames = [file for file in files if (os.path.splitext(file)[1]==".sdf")]

    ref_path = self.ref_location
    ref_files = os.listdir(ref_path)         
    ref_filenames = [file for file in ref_files if (os.path.splitext(file)[1]==".sdf")]
    ref_mols = []
    for ref_file in ref_filenames:
      mol = next(oddt.toolkit.readfile('sdf',f'{ref_path}/{ref_file}'))
      ref_mols.append(mol)

    mols = []
    for test_file in filenames:
      mol = next(oddt.toolkit.readfile('sdf',f'{path}/{test_file}'))
      mols.append(mol)
    
    best_list = []
    for i,rm in enumerate(ref_mols):
      #print(f'Testing reference conformation {i} -------------------------------------------')
      reference_shape = self.sim_function(rm, comp_type)
      scores = []
      idx_tuples = []
      for j, tm in enumerate(mols):
        query_shape = self.sim_function(tm, comp_type)
        similarity = usr_similarity(query_shape, reference_shape)
        scores.append(similarity)
        idx_tuples.append((i,j))
      
      best_idx = np.argmax(scores)
      best_score = scores[best_idx]
      best_tuple = idx_tuples[best_idx]
      #print(f'The best match for reference conformation {best_tuple[0]} and test conformation {best_tuple[1]}: {best_score:.3f}')
      total_tuple = (best_score, best_tuple)
      best_list.append(total_tuple)
    
    print('=======================================================================================')
    print('=======================================================================================')
    overall_best = np.argmax([x[0] for x in best_list])
    overall_score = best_list[overall_best][0]
    overall_tuple = best_list[overall_best][1]
    print(f'The overall best match is reference conformation {overall_tuple[0]} and test conformation {overall_tuple[1]}: {overall_score:.3f}')
    
    os.system(f'cp {self.ref_location}/{self.ref_base}_{overall_tuple[0]}.sdf {self.ref_base}_{overall_tuple[0]}_{self.sdf_base}_{overall_tuple[1]}_REF.sdf')
    os.system(f'cp {self.sdf_location}/{self.sdf_base}_{overall_tuple[1]}.sdf {self.ref_base}_{overall_tuple[0]}_{self.sdf_base}_{overall_tuple[1]}_TEST.sdf')

    suppl = Chem.SDMolSupplier(f'{self.ref_base}_{overall_tuple[0]}_{self.sdf_base}_{overall_tuple[1]}_REF.sdf')
    for mol in suppl:
      ref_mol = mol
    
    suppl = Chem.SDMolSupplier(f'{self.ref_base}_{overall_tuple[0]}_{self.sdf_base}_{overall_tuple[1]}_TEST.sdf')
    for mol in suppl:
      test_mol = mol
    
    mols_to_print = [ref_mol, test_mol]
    legends = ['Reference', 'Test']
    img = Draw.MolsToGridImage(mols_to_print, legends = legends, molsPerRow=2, subImgSize=(200,200))

    return best_list, img

  
  