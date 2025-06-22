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
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from fairchem.core import FAIRChemCalculator, pretrained_mlip

global HMGCR_data
HMGCR_data = {
        "file_location":"CafChem/HMGCR_dude_QM_site.xyz",
        "charge": 3,
        "spin": 1,
        "constraints": [1, 11, 16, 24, 33, 41, 54, 60, 72, 83, 92, 98, 107, 124, 132, 140, 148, 159, 168, 181],
        "size": 331
        }

def save_pose(pose_mol, pose_score, name,saved_index):
  '''
    Save the bext pose (lowest score) from a docking run as an SDF file. 
    
    Args:
        pose_mol: the mol object for the best pose
        pose_score: the score for the best pose
        name: a name to use for the SDF file
        saved_index: a number to attach to the name
    
    Returns:
        None; SDF file is saved.
  '''
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
        scores: a list of scores; poses are saved in SDF files.
  '''
  df = pd.read_csv(filename)
  target = load_target(target_protein)
  count = df.shape[0]

  print("===============================================")
  print(f"Docking {count} molecules in {target_protein}.")

  i = 0
  saved_index = 1
  scores = []
  for smile in df[key]:
    try:
      print(f"Docking molecule {i+1}.")
      score, aux = target.dock(smile, num_cpus = num_cpus)
      scores.append(score)
      save_pose(aux['ligand'],score,"trial",saved_index)
      saved_index += 1
    except:
      print(f"Molecule {i} could not be docked!")
      scores.append(0.0)
    i += 1
  return scores  

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
        scores: a list of scores; poses are saved in SDF files.
  '''
  target = load_target(target_protein)
  count = len(smiles_list)

  print("===============================================")
  print(f"Docking {count} molecules in {target_protein}.")

  i = 0
  saved_index = 1
  scores = []
  for smile in smiles_list:
    try:
      print(f"Docking molecule {i+1}.")
      score, aux = target.dock(smile, num_cpus = num_cpus)
      scores.append(score)
      save_pose(aux['ligand'],score,"trial",saved_index)
      saved_index += 1
    except:
      print(f"Molecule {i} could not be docked!")
      score.append(0.0)
    i += 1
  return scores

def dock_smiles(smile: str, target_protein: str, num_cpus: int):
  '''
    Dock a single SMILES string to a target protein using DockString.

      Args:
        smile: a single SMILES string
        target_protein: the target protein to dock in; must be included in the 
        DockString database
        num_cpus: the number of CPUs to use for docking
      Returns:
        score; poses are saved in SDF files.
  '''
  target = load_target(target_protein)
  print("===============================================")
  print(f"Docking molecule.")
  try: 
    score, aux = target.dock(smile, num_cpus = num_cpus)
    save_pose(aux['ligand'],score,"trial",1)
  except:
    print(f"Molecule could not be docked!")
  return score

def uma_interaction(filename_base: str, target_obj: str, calculator: FAIRChemCalculator,
                    charge: int, spin: int, optFlag = False):
  '''
    retrieve a molecule from an SDF file, add protons, and calculate the energy
    using Meta's UMA MLIP.
    
    Args:
      filename_base: the SDF filename without the sdf extension
      target: the target protein to dock in followed by _data
      calculator: the FAIRCHEM calculator
      charge: charge of the molecule
      spin: spin multiplicity of the molecule
      optFlag = True -> optimizes the complex, False -> does not optimize the complex
    Returns:
      ie: the interaction energy between the lignd and the protein active site.
      total_xyz_list: a list with the xyz strings for the unoptimized ligands
      complex structures are saved as XYZ files
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

    '''
      Put together atoms objects for the ligand and the protein. Combine them 
      into a single ASE atoms object for the pl complex. Optimize the structure
      and calculate the energy of the pl complex. Optimization uses a 
      constraints list from the target object. Separate the optimized complex
      into new ASE atoms objects for the ligand and the protein. Calculate the
      energy of each.
    '''
    atoms1 = Atoms(atoms_list,xyz_list)     
    as_mol1 = ase.io.read(target_obj["file_location"], format="xyz")
    pl_complex = as_mol1 + atoms1
    total_spin = spin + target_obj["spin"] - 1
    total_charge = charge + target_obj["charge"]
    pl_complex.info.update({"spin": total_spin, "charge": total_charge})
    pl_complex.calc = calculator

    combo_size = len(pl_complex)
    as_size = len(as_mol1)

    print(f"The size of the complex is: {combo_size}")
    c = FixAtoms(indices = target_obj["constraints"])
    pl_complex.set_constraint(c)
    if optFlag: 
      dyn = BFGS(pl_complex)
      dyn.run(fmax=0.1)
      optimized_energy = pl_complex.get_potential_energy()
      print(f"Optimized energy of complex is: {0.0367493*optimized_energy:.3f} ha")
      ase.io.write("optimized_complex.xyz", images=pl_complex, format="xyz")
    else:
      optimized_energy = pl_complex.get_potential_energy()
      print(f"Energy of complex is: {0.0367493*optimized_energy:.3f} ha")
      ase.io.write("total_complex.xyz", images=pl_complex, format="xyz")

    # build atoms objects 
    atoms = pl_complex[as_size:]
    print(f"The size of the ligand is: {len(atoms)}")
    atoms.info.update({"spin": spin, "charge": charge})
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    print(f"Energy of ligand is: {0.0367493*energy:.3f} ha")

    #calculate energy of active site
    as_mol = pl_complex[:as_size]
    print(f"The size of the active site is: {len(as_mol)}")
    as_mol.info.update({"spin": target_obj["spin"], "charge": target_obj["charge"]})
    as_mol.calc = calculator
    as_energy = as_mol.get_potential_energy()
    print(f"Energy of active site is: {0.0367493*as_energy:.3f} ha")
  
    #calculate the interaction energy
    print("===========================================================")
    ie = 23.06035*(optimized_energy-as_energy-energy)
    print(f"Energy difference is: {ie:.3f} kcal/mol")

    # Save the XYZ string(s) and pass back for visualization
    total_xyz_list.append(xyz_string)

  return ie, total_xyz_list

def old_uma_interaction(filename_base: str, target_obj: str, calculator: FAIRChemCalculator,
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
      total_xyz_list: a list of xyz strings for each molecule in the SDF file
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
    print(f"Energy of ligand is: {0.0367493*energy:.3f} ha")

    #calculate energy of active site
    as_mol = ase.io.read(target_obj["file_location"], format="xyz")
    as_mol.info.update({"spin": target_obj["spin"], "charge": target_obj["charge"]})
    as_mol.calc = calculator
    as_energy = as_mol.get_potential_energy()
    print(f"Energy of active site is: {0.0367493*as_energy:.3f} ha")

    #calculate energy of the compex
    pl_complex = atoms + as_mol
    total_spin = spin + target_obj["spin"] - 1
    total_charge = charge + target_obj["charge"]
    pl_complex.info.update({"spin": total_spin, "charge": total_charge})
    pl_complex.calc = calculator
    print(f"The size of the complex is: {len(pl_complex)}")
    pl_complex_energy = pl_complex.get_potential_energy()
    print(f"Energy of complex is: {0.0367493*pl_complex_energy:.3f} ha")

    #calculate the interaction energy
    print("===========================================================")
    print(f"Energy difference is: {23.06035*(pl_complex_energy-as_energy-energy):.3f} kcal/mol")

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

def complexG16(in_file: str, target_obj: dict, charge: int, spin: int):
  '''
    Reads in the complex.xyz file generated by the uma_interaction function
    and converts it to a G16 input file.

    Args:
      in_file: the complex.xyz file
      target_obj: the target information
      charge: the charge of the ligand
      spin: the spin multiplicity of the ligand
    Returns:
      None; writes the G16 input file.
  '''
  f = open(in_file,"r")
  lines = f.readlines()
  f.close()

  total_charge = target_obj["charge"] + charge
  total_spin = target_obj["spin"] + spin - 1
  as_size = target_obj["size"]

  header = "%chk=/scratch2/gaussian/complex.chk\n"
  header += "%mem=32GB\n"
  header += "%nprocshared=32\n"
  header += "#p counterpoise=2 wB97XD def2tzvpp scf=xqc\n\n"
  header += "complex\n\n"
  header += f'{total_charge} {total_spin} {target_obj["charge"]} {target_obj["spin"]} {charge} {spin}\n'
  
  g = open("complex.gjf","w")
  g.write(header)
  for line in lines[2:as_size+2]:
    line = line.strip("\n") + " 1\n"
    g.write(line)
  for line in lines[as_size+2:]:
    line = line.strip("\n") + " 2\n"
    g.write(line)
  g.write("\n")
  g.close()