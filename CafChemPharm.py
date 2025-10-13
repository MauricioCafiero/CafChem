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
  
  def expand_conf(self, mol_idx: int):
    '''
      Expands the molecule object by extracting each conformer into it's own molecule object.

      Args:
        mol_idx: index of the molecule object to expand
      Returns:
        the expanded molecule objects containing the conformers
    '''
    expanded_confs = []
    for i in range(self.embedded_mols[mol_idx].GetNumConformers()):
      m = Chem.Mol(self.embedded_mols[mol_idx],confId=i)
      expanded_confs.append(m)

    return expanded_confs
  
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

    return expanded_confs
  
  def get_XYZ_strings(self, mol_idx):
    '''
      Generates XYZ strings for each conformer of a molecule object.

      Args:
        mol_idx: index of the molecule object to generate XYZ strings for
      Returns:
        A list of XYZ strings for each conformer of the molecule object
    '''
    expanded_confs = self.expand_all_confs()
    xyz_strings = []

    for conf in expanded_confs[mol_idx]:
      xyz_string = Chem.MolToXYZBlock(conf)
      xyz_strings.append(xyz_string)
    
    self.xyz_strings = xyz_strings

    return self.xyz_strings
  
  def make_xyz_files(self, mol_idx):
    '''
      Generates XYZ files for each conformer of a molecule object. The files are saved in the current directory.

      Args:
        mol_idx: index of the molecule object to generate XYZ files for
      Returns:
        None
    '''
    xyz_strings = self.get_XYZ_strings(mol_idx)
    
    for i,xyz in enumerate(xyz_strings):
      with open(f'conf_{i}.xyz','w') as f:
        f.write(xyz)

class evaluate_pharmacophore():
  '''
    Class for evaluating pharmacophore features of a known and test molecule
  '''
  def __init__(self,known: str, test: str, draw_flag = True):
    '''
      Reads SMILES strings for a known active and a test molecule, finds the pharmacophore features
      for each and makes comparisons.

      Args:
        known: SMILES string for the known active
        test: SMILES string for the test molecule
        draw_flag: boolean, whether to draw the molecules or not
      Returns:
        None
    '''
    self.known = Chem.MolFromSmiles(known)
    self.test = Chem.MolFromSmiles(test)
    self.draw_flag = draw_flag
    self.keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'LumpedHydrophobe')

    self.smiles = [known, test]
    self.mols = [Chem.MolFromSmiles(x) for x in self.smiles]
    
  def align_confs(self):
    '''
      Generates a conformer for each molecule and aligns the conformers.

      Args:
        None
      Returns:
        The embedded molecule objects containing the aligned conformers
    '''

    self.mols = [Chem.AddHs(m) for m in self.mols]
    ps = AllChem.ETKDGv3()
    #ps.randomSeed = 0xf00d  # we seed the RNG so that this is reproducible
    for m in self.mols:
        AllChem.EmbedMolecule(m,ps)

    o3d = rdMolAlign.GetO3A(self.mols[1],self.mols[0])
    o3d.Align()

    if self.draw_flag == True:
      draw_mol(self.mols)

    return self.mols

  def get_feat_lists_maps(self, draw = 'maps'):
    '''
      creates feature lists and maps for the known and test molecules.

      Args:
        draw: string, which feature maps to draw. Can be 'maps' or 'feats'
      Returns:
        The feature lists and maps for the known and test molecules
    '''
    self.feat_lists = []
    for m in self.mols:
      rawFeats = fdef.GetFeaturesForMol(m)
      self.feat_lists.append([f for f in rawFeats if f.GetFamily() in self.keep])
    
    self.feat_maps = [FeatMaps.FeatMap(feats = x,weights=[1]*len(x),params=fmParams) for x in self.feat_lists]

    if draw == 'maps' and self.draw_flag == True:
      drawFeatMap(self.mols[0], self.feat_maps[0])
      drawFeatMap(self.mols[1], self.feat_maps[1])
    elif draw == 'feats' and self.draw_flag == True:
      draw_feats(self.mols[0], self.feat_lists[0])
      draw_feats(self.mols[1], self.feat_lists[1])

    return self.feat_lists, self.feat_maps
  
  def get_feat_score(self, print_flag = True):
    '''
      Calculates the pharmacophore feature score for the test molecule. 
      The score is normalized by the number of features in the known and test molecules.

      Args:
        None
      Returns:
        The pharmacophore feature scores for the test molecule
    '''

    self.test_score_norm = self.feat_maps[1].ScoreFeats(self.feat_maps[0].GetFeatures())/min(self.feat_maps[1].GetNumFeatures(),self.feat_maps[0].GetNumFeatures())
    self.test_score = self.feat_maps[1].ScoreFeats(self.feat_maps[0].GetFeatures())/self.feat_maps[0].GetNumFeatures()

    if print_flag == True:
      print(f'Test score: {self.test_score}')
      print(f'Test score normalized: {self.test_score_norm}')

    return self.test_score, self.test_score_norm
  
  def get_feat_comparison(self):
    '''
      Compares the pharmacophore features of the known and test molecules.

      Args:
        None
      Returns:
        A string with the pharmacophore features of the known and test molecules
    '''
    feat_hash = {'Donor': 'Hydrogen bond donors', 'Acceptor': 'Hydrogen bond acceptors', 
               'NegIonizable': 'Negatively ionizable groups', 'PosIonizable': 'Positively ionizable groups', 
               'ZnBinder': 'Zinc Binders', 'Aromatic': 'Aromatic rings', 'LumpedHydrophobe': 'Hydrophobic/non-polar groups' }

    feats_known = {}
    feats_test = {}

    for feat in self.feat_lists[0]:
      if feat.GetFamily() not in feats_known.keys():
        feats_known[feat.GetFamily()]  = 1
      else:
        feats_known[feat.GetFamily()] += 1
  
    for feat in self.feat_lists[1]:
      if feat.GetFamily() not in feats_test.keys():
        feats_test[feat.GetFamily()]  = 1
      else:
        feats_test[feat.GetFamily()] += 1
    
    for keep_feat in self.keep:
      if keep_feat not in feats_known.keys():
        feats_known[keep_feat] = 0
      if keep_feat not in feats_test.keys():
        feats_test[keep_feat] = 0
    
    known_string = "known"
    test_string = "test"
    props_string = f"The Pharmacophore Features are as follows: \n"
    props_string += "============================================== \n"
    props_string += f"{known_string:30} {test_string:30}"
    for known_feat, test_feat in zip(feats_known.keys(),feats_test.keys()):    
      props_string += f"\n{known_feat:20}: {feats_known[known_feat]:10} | {test_feat:20}: {feats_test[test_feat]:10}"
    
    return props_string

class evaluate_pharmacophore_all_confs():
  '''
    Class for evaluating pharmacophore features of a known and all conformations of a test molecule
  '''
  def __init__(self,known: str, test: str, num_confs: int):
    '''
      Reads SMILES strings for a known active and a test molecule, finds the pharmacophore features
      for each conformation of the test molecule and makes comparisons.

      Args:
        known: SMILES string for the known active
        test: SMILES string for the test molecule
      Returns:
        None
    '''
    self.known = known
    self.test = test
    self.num_confs = num_confs
    self.keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'LumpedHydrophobe')
    self.smiles = [known, test]
    self.mols = [Chem.MolFromSmiles(x) for x in self.smiles]
    
  def align_confs(self):
    '''
      Generates a conformer for the known molecules and num_confs conformers for the test molecule.
      Aligns the conformers.

      Args:
        None
      Returns:
        known: the embedded molecule object containing the known active
        test_confs: the embedded molecule objects containing the conformers of the test molecule
    '''

    test_conformers = conformers([self.test],self.num_confs)
    embedded_mols = test_conformers.get_confs()
    self.expanded_confs = test_conformers.expand_conf(0)

    ps = AllChem.ETKDGv3()
    ps.randomSeed=0xf00d
    known_Hs = Chem.AddHs(self.mols[0])
    self.known_embedded = Chem.Mol(known_Hs)
    AllChem.EmbedMolecule(self.known_embedded,ps)

    for conf in self.expanded_confs:
      o3d = rdMolAlign.GetO3A(conf,self.known_embedded)
      o3d.Align()

    return self.known_embedded, self.expanded_confs

  def get_feat_lists_maps(self):
    '''
      creates feature lists and maps for the known and test molecules.

      Args:
        None
      Returns:
        The feature lists and maps for the known and test molecules
    '''
    self.feat_lists = []
    rawFeats = fdef.GetFeaturesForMol(self.known_embedded)
    self.feat_lists.append([f for f in rawFeats if f.GetFamily() in self.keep])

    for m in self.expanded_confs:
      rawFeats = fdef.GetFeaturesForMol(m)
      self.feat_lists.append([f for f in rawFeats if f.GetFamily() in self.keep])
    
    self.feat_maps = [FeatMaps.FeatMap(feats = x,weights=[1]*len(x),params=fmParams) for x in self.feat_lists]

    return self.feat_lists, self.feat_maps
  
  def get_feat_score(self):
    '''
      Calculates the pharmacophore feature score for each conformation of the test molecule. 
      The scores are normalized by the number of features in the known and test molecules.

      Args:
        None
      Returns:
        props_string: The pharmacophore feature scores for each conformation of the test molecule
    '''
    test_scores = []
    test_scores_norm = []
    for i in range(1,len(self.feat_maps),1):
      test_score_norm = self.feat_maps[i].ScoreFeats(self.feat_maps[0].GetFeatures())/min(self.feat_maps[i].GetNumFeatures(),self.feat_maps[0].GetNumFeatures())
      test_score = self.feat_maps[i].ScoreFeats(self.feat_maps[0].GetFeatures())/self.feat_maps[0].GetNumFeatures()
      test_scores.append(test_score)
      test_scores_norm.append(test_score_norm)
      print(f'Test score for conformation {i-1}: {test_score}')
      print(f'Test score normalized for conformation {i-1}: {test_score_norm}')
      print("===============================================================")
    
    max_idx = np.argmax(test_scores)
    print(f'The best conformation is {max_idx}')

    feat_hash = {'Donor': 'Hydrogen bond donors', 'Acceptor': 'Hydrogen bond acceptors', 
               'NegIonizable': 'Negatively ionizable groups', 'PosIonizable': 'Positively ionizable groups', 
               'ZnBinder': 'Zinc Binders', 'Aromatic': 'Aromatic rings', 'LumpedHydrophobe': 'Hydrophobic/non-polar groups' }

    feats_known = {}
    feats_test = {}

    for feat in self.feat_lists[0]:
      if feat.GetFamily() not in feats_known.keys():
        feats_known[feat.GetFamily()]  = 1
      else:
        feats_known[feat.GetFamily()] += 1
  
    for feat in self.feat_lists[max_idx+1]:
      if feat.GetFamily() not in feats_test.keys():
        feats_test[feat.GetFamily()]  = 1
      else:
        feats_test[feat.GetFamily()] += 1
    
    for keep_feat in self.keep:
      if keep_feat not in feats_known.keys():
        feats_known[keep_feat] = 0
      if keep_feat not in feats_test.keys():
        feats_test[keep_feat] = 0
    
    known_string = "known"
    test_string = "best conformation"
    props_string = f"The Pharmacophore Features are as follows: \n"
    props_string += "============================================== \n"
    props_string += f"{known_string:30} {test_string:30}"
    for known_feat, test_feat in zip(feats_known.keys(),feats_test.keys()):    
      props_string += f"\n{known_feat:20}: {feats_known[known_feat]:10} | {test_feat:20}: {feats_test[test_feat]:10}"

    print(props_string)
    
    return test_scores, test_scores_norm, props_string

def draw_mol(ms, p=None, confId=-1, removeHs=True,colors=('cyanCarbon','redCarbon','blueCarbon')):
  '''
    Draws the molecules overlaid on each other

    Args:
      ms: list of molecule objects
      p: py3Dmol view object
      confId: index of the conformer to draw
      removeHs: boolean, whether to remove hydrogen atoms
      colors: list of colors to use for each molecule
    Returns:
      p: py3Dmol view object
  '''
  if p is None:
      p = py3Dmol.view(width=400, height=400)
  p.removeAllModels()
  for i,m in enumerate(ms):
      if removeHs:
          m = Chem.RemoveHs(m)
      IPythonConsole.addMolToView(m,p,confId=confId)
  for i,m in enumerate(ms):
      p.setStyle({'model':i,},
                      {'stick':{'colorscheme':colors[i%len(colors)]}})
  p.zoomTo()
  return p.show()


def colorToHex(rgb):
  '''
    Converts RGB to Hex

    Args:
      rgb: list of RGB values
    Returns:
      Hex value
  '''
  rgb = [f'{int(255*x):x}' for x in rgb]
  return '0x'+''.join(rgb)

def draw_feats(m, feats, p=None, confId=-1, removeHs=True):
  '''
    Draws the pharmacophore features of a molecule

    Args:
      m: molecule object
      feats: list of pharmacophore features
      p: py3Dmol view object
      confId: index of the conformer to draw
      removeHs: boolean, whether to remove hydrogen atoms
    Returns:
      p: py3Dmol view object
  '''
  if p is None:
      p = py3Dmol.view(width=400, height=400)
  p.removeAllModels()
  if removeHs:
      m = Chem.RemoveHs(m)
  IPythonConsole.addMolToView(m,p,confId=confId)
  for feat in feats:
      pos = feat.GetPos()
      clr = featColors.get(feat.GetFamily(),(.5,.5,.5))
      p.addSphere({'center':{'x':pos.x,'y':pos.y,'z':pos.z},'radius':.5,'color':colorToHex(clr)});
  p.zoomTo()
  return p.show()

def drawFeatMap(m, fMap, p=None, confId=-1, removeHs=True):
  '''
    Draws the pharmacophore feature map of a molecule

    Args:
      m: molecule object
      fMap: pharmacophore feature map
      p: py3Dmol view object
      confId: index of the conformer to draw
      removeHs: boolean, whether to remove hydrogen atoms
    Returns:
      p: py3Dmol view object
  '''
  if p is None:
      p = py3Dmol.view(width=400, height=400)
  p.removeAllModels()
  if removeHs:
      m = Chem.RemoveHs(m)
  IPythonConsole.addMolToView(m,p,confId=confId)
  for feat in fMap.GetFeatures():
      pos = feat.GetPos()
      clr = featColors.get(feat.GetFamily(),(.5,.5,.5))
      p.addSphere({'center':{'x':pos.x,'y':pos.y,'z':pos.z},'radius':feat.weight*.5,'color':colorToHex(clr)});
  p.zoomTo()
  return p.show()