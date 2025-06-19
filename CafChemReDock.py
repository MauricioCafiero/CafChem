import numpy as np
import pandas as pd
import deepchem as dc
import time
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem, Draw
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from dockstring import load_target
from google.colab import files
import os
import py3Dmol
import ase.io
from ase import Atoms
from fairchem.core import FAIRChemCalculator, pretrained_mlip

global HMGCR_data
HMGCR_data = {
        "file_location":"CafChem/HMGCR_dude_QM_site.xyz",
        "charge": 3,
        "spin": 1
        }

def save_pose(pose_mol, pose_score, name,saved_index):
  pose_mol.SetProp('_Name',str(pose_score))
  sdf_filename = f"{name}_{saved_index}.sdf"
  w = Chem.SDWriter(sdf_filename)
  w.write(pose_mol)
  w.close
  #files.download(sdf_filename)
  print(f"SDF file written for score {pose_score}")

def dock_dataframe(filename: str, target_protein: str, num_cpus: int, key = "SMILES",):
  '''
    Dock all SMILES strings in a CSV file to a target protein using DockString. 
    Accepts a filename for the CSV file, converts it to a dataframe, and then uses
    the provided key to loop over all SMILES to dock. Saves all poses in SDF files. 

      Args:
        filename: the CSV file name
        key: the column title for the SMILES strings 
        target_protein: the target protein to dock in; must be included in the 
          DockString database
        num_cpus: the number of CPUs to use for docking
      Returns:
        None; poses are saved in SDF files.
  '''
  df = pd.read_csv(filename)
  target = load_target(target_protein)
  count = df.shape[0]

  print("===============================================")
  print(f"Docking {count} molecules in {target_protein}.")

  i = 0
  saved_index = 1
  for smile in df[key]:
    try:
      print(f"Docking molecule {i+1}.")
      score, aux = target.dock(smile, num_cpus = num_cpus)
      save_pose(aux['ligand'],score,"trial",saved_index)
      saved_index += 1
    except:
      print(f"Molecule {i} could not be docked!")
    i += 1

def dock_list(smiles_list: list, target_protein: str, num_cpus: int):
  '''
    Dock all SMILES strings in a list to a target protein using DockString. 
    Loop over all SMILES in the list to dock. Saves all poses in SDF files. 

      Args:
        smiles_list: a list of SMILES strings
        target_protein: the target protein to dock in; must be included in the 
          DockString database
        num_cpus: the number of CPUs to use for docking
      Returns:
        None; poses are saved in SDF files.
  '''
  target = load_target(target_protein)
  count = len(smiles_list)

  print("===============================================")
  print(f"Docking {count} molecules in {target_protein}.")

  i = 0
  saved_index = 1
  for smile in smiles_list:
    try:
      print(f"Docking molecule {i+1}.")
      score, aux = target.dock(smile, num_cpus = num_cpus)
      save_pose(aux['ligand'],score,"trial",saved_index)
      saved_index += 1
    except:
      print(f"Molecule {i} could not be docked!")
    i += 1

def dock_smiles(smile: str, target_protein: str, num_cpus: int):
  '''
    Dock a single SMILES string to a target protein using DockString.

      Args:
        smile: a single SMILES string
        target_protein: the target protein to dock in; must be included in the 
        DockString database
        num_cpus: the number of CPUs to use for docking
      Returns:
        None; poses are saved in SDF files.
  '''
  target = load_target(target_protein)
  print("===============================================")
  print(f"Docking molecule.")
  try: 
    score, aux = target.dock(smile, num_cpus = num_cpus)
    save_pose(aux['ligand'],score,"trial",1)
  except:
    print(f"Molecule could not be docked!")

def uma_interaction(filename_base: str, target_obj: str, calculator: FAIRChemCalculator,
                    charge: int, spin: int):
  '''
    retrieve a molecule from an SDF file, add protons, and calculate the energy
    using Meta's UMA MLIP.
    
    Args:
      filename_base: the SDF filename without the sdf extension
      target: the target protein to dock in followed by _data
      calculator: the FAIRCHEM calculator
      charge: charge of the molecule
      spin: spin multiplicity of the molecule
    Returns:
      None
  '''
  filename = filename_base + ".sdf"
  suppl = Chem.SDMolSupplier(filename)

  total_xyz_list = []
  for mol in suppl:
    '''
    In most cases this loop will only be over one molecule.
    adds protons to the structure and then makes and XYZ string, 
    and an ASE atoms object.
    '''
    xyz_list = []
    atoms_list = ""
    template = mol
    molH = Chem.AddHs(mol)
    AllChem.ConstrainedEmbed(molH,template, useTethers=True)
    xyz_string = f"{molH.GetNumAtoms()}\n\n"
    for atom in molH.GetAtoms():
      atoms_list += atom.GetSymbol()
      pos = molH.GetConformer().GetAtomPosition(atom.GetIdx())
      temp_tuple = (pos[0], pos[1], pos[2])
      xyz_list.append(temp_tuple)
      xyz_string += f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}\n"

    # build atoms object and calculate energy
    atoms = Atoms(atoms_list,xyz_list) 
    atoms.info.update({"spin": spin, "charge": charge})
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    print(f"Energy of ligand is: {0.0367493*energy:.3f} kcal/mol")

    #calculate energy of active site
    as_mol = ase.io.read(target_obj["file_location"], format="xyz")
    as_mol.info.update({"spin": target_obj["spin"], "charge": target_obj["charge"]})
    as_mol.calc = calculator
    as_energy = as_mol.get_potential_energy()
    print(f"Energy of active site is: {0.0367493*as_energy:.3f} kcal/mol")

    #calculate energy of the compex
    pl_complex = atoms + as_mol
    total_spin = spin + target_obj["spin"] - 1
    total_charge = charge + target_obj["charge"]
    pl_complex.info.update({"spin": total_spin, "charge": total_charge})
    pl_complex.calc = calculator
    print(f"The size of the complex is: {len(pl_complex)}")
    pl_complex_energy = pl_complex.get_potential_energy()
    print(f"Energy of complex is: {0.0367493*pl_complex_energy:.3f} kcal/mol")

    #calculate the interaction energy
    print("===========================================================")
    print(f"Energy difference is: {0.0367493*(pl_complex_energy-as_energy-energy):.3f} kcal/mol")

    # Save the XYZ string(s) and pass back for visualization
    total_xyz_list.append(xyz_string)

  return total_xyz_list

def visualize_molecule(xyz_string: str):
  '''
    input an XYZ string to vosualize the molecule in 3D
  '''
  viewer = py3Dmol.view(width=800, height=400)
  viewer.addModel(xyz_string, "xyz")  
  viewer.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
  viewer.zoomTo()
  viewer.show()