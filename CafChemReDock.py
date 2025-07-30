import numpy as np
import pandas as pd
import deepchem as dc
import time
import random
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
        "file_location":"CafChem/data/HMGCR_dude_QM_site.xyz",
        "charge": 3,
        "spin": 1,
        "constraints": [1, 11, 16, 24, 33, 41, 54, 60, 72, 83, 92, 98, 107, 124, 132, 140, 148, 159, 168, 181],
        "size": 331
        }

global DRD2_data
DRD2_data = {
        "file_location":"CafChem/data/DRD2_dude_QM_site.xyz",
        "charge": -1,
        "spin": 1,
        "constraints": [1, 10, 18, 27, 33, 42, 54, 62, 78, 89, 101, 110],
        "size": 216
        }

global MAOB_data
MAOB_data = {
        "file_location":"CafChem/data/MAOB_dude_QM_site.xyz",
        "charge": -1,
        "spin": 1,
        "constraints": [1, 7, 12, 17, 22, 31, 39, 44, 51, 61, 67, 84, 89, 94, 111, 120, 129, 134, 139, 147, 161, 172, 180, 188, 197, 206, 218, 235, 242, 250, 256, 265, 272, 281, 295, 306, 322, 335, 342, 356, 361, 370, 375, 389, 398, 408],
        "size": 809
        }

global MAOBnoFAD_data
MAOBnoFAD_data = {
        "file_location":"CafChem/data/MAOBnoFAD_dude_QM_site.xyz",
        "charge": 1,
        "spin": 1,
        "constraints": [1, 7, 12, 17, 22, 31, 39, 44, 51, 61, 67, 84, 89, 94, 111, 120, 129, 134, 139, 147, 161, 172, 180, 188, 197, 206, 218, 235, 242, 250, 256, 265, 272, 281, 295, 306, 322, 335, 341, 355, 360, 369, 374, 388, 397, 407],
        "size": 727
        }

global ADRB2_data
ADRB2_data = {
        "file_location":"CafChem/data/ADRB2_dude_QM_site.xyz",
        "charge": -2,
        "spin": 1,
        "constraints": [1, 16, 25, 33, 41, 49, 58, 66, 78, 84, 92, 100, 107, 122, 133, 145, 156, 170, 179, 189],
        "size": 349
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

class solvation():
  '''
  Class to hold all functions related to adding waters to a molecule
  '''
  def __init__(self, filename: str, how_many_water_radii: int):
    '''
    Add randomly placed waters to a molecule

      Args:
        filename: XYZ file of molecule
        how_many_water_radii: number of water radii around the molecules for the
                              box dimensions
    '''
    self.filename = filename
    self.how_many_water_radii = how_many_water_radii
    self.water_vdw_rad = 1.7
    print("add_waters class initialized")
  
  def add_box(self, val):
    new_val = val + self.how_many_water_radii*self.water_vdw_rad
    return new_val
    
  def sub_box(self, val):
    new_val = val - self.how_many_water_radii*self.water_vdw_rad
    return new_val
  
  def calc_distance(self,water: list, atom: list):
    '''
    Calculate the distance between the O in the newly added water and an atom
    in the molecule.

      Args:
        water: list of XYZ coordinates for the O atom in water
        atom: list of XYZ coordinates for an atom in the molecule
      Returns:
        distance: distance between the two atoms
    '''
    distance = 0.0
    for i in range(3):
        distance += (float(water[i])-float(atom[i]))**2
    distance = np.sqrt(distance)
    return distance

  def get_box_size(self):
    '''
      Get the box size for the molecule by finding the maximum dimensions and then
      adding a specified number of water van der Waals radii.

        Args:
          None
        Returns:
          max_values: list of maximum XYZ dimensions
          min_values: list of minimum XYZ dimensions
    '''
    f = open(self.filename,"r")
    lines = f.readlines()
    f.close()
  
    max_values = np.zeros((3))
    min_values = np.zeros((3))
    x_list = []
    y_list = []
    z_list = []
    for row in lines[2:]:
        parts = row.split()
        x_list.append(parts[1]) 
        y_list.append(parts[2])
        z_list.append(parts[3]) 
    max_values[0] = max(x_list)
    max_values[1] = max(y_list)
    max_values[2] = max(z_list)
    min_values[0] = min(x_list)
    min_values[1] = min(y_list)
    min_values[2] = min(z_list)
    
    max_values = [self.add_box(val) for val in max_values]
    min_values = [self.sub_box(val) for val in min_values]

    dims = ["x","y","z"]
    print(f"Maximum dimensions after augmentation are:")
    for dim,maxes,mins in zip(dims,max_values,min_values):
        print(f"{dim} - Max: {maxes}, Min: {mins}")
    
    volume = 1.0
    for big,small in zip(max_values,min_values):
        volume *= (big - small)
    print(f"Volume is {volume} A^3")
    
    return max_values, min_values
  
  def get_water_coordinates(self,o_coordinates: list):
    '''
    Takes coordinates for an oxygen atom, adds two H-atoms, translates the molecule
    to the new location and adds a random rotation.

      Args:
        o_coordinates: oxygen atom coordinates
      Returns:
        rot_water_xyz: string of coordinates for the water molecule
        rot_water_coordinates: list of coordinates for the water molecule
    '''
    o_xyz = np.asarray([0.0, 0.0, 0.0]).reshape(3,1)
    h1_xyz = np.asarray([0.580743,  0.000000,  0.758810]).reshape(3,1)
    h2_xyz = np.asarray([0.580743,  0.000000,  -0.758810]).reshape(3,1)

    theta_x = random.uniform(0,2*np.pi)
    theta_y = random.uniform(0,2*np.pi)
    theta_z = random.uniform(0,2*np.pi)
    
    x_rotation_matrix = np.asarray(([1.0,0.0,0.0],[0.0,np.cos(theta_x),-np.sin(theta_x)],[0.0,np.sin(theta_x),np.cos(theta_x)])).reshape(3,3)
    y_rotation_matrix = np.asarray(([np.cos(theta_y),0.0,-np.sin(theta_y)],[0.0,1.0,0.0],[np.sin(theta_y),0.0,np.cos(theta_y)])).reshape(3,3)
    z_rotation_matrix = np.asarray(([np.cos(theta_z),-np.sin(theta_z),0.0],[np.sin(theta_z),np.cos(theta_z),0.0],[0.0,0.0,1.0])).reshape(3,3)
    
    rot_h1_xyz = np.matmul(x_rotation_matrix,h1_xyz)
    rot_h1_xyz = np.matmul(y_rotation_matrix,rot_h1_xyz)
    rot_h1_xyz = np.matmul(z_rotation_matrix,rot_h1_xyz)
    rot_h2_xyz = np.matmul(x_rotation_matrix,h2_xyz)
    rot_h2_xyz = np.matmul(y_rotation_matrix,rot_h2_xyz)
    rot_h2_xyz = np.matmul(z_rotation_matrix,rot_h2_xyz)
        
    for i in range(3):
        o_xyz[i] = o_xyz[i] + o_coordinates[i]
        rot_h1_xyz[i] = rot_h1_xyz[i] + o_coordinates[i]
        rot_h2_xyz[i] = rot_h2_xyz[i] + o_coordinates[i]
    
    rot_water_xyz =  f"O {o_xyz[0].item()}   {o_xyz[1].item()}    {o_xyz[2].item()}\n"
    rot_water_xyz += f"H {rot_h1_xyz[0].item()}   {rot_h1_xyz[1].item()}    {rot_h1_xyz[2].item()}\n"
    rot_water_xyz += f"H {rot_h2_xyz[0].item()}   {rot_h2_xyz[1].item()}    {rot_h2_xyz[2].item()}\n"

    rot_water_coordinates = []
    rot_water_coordinates.append([o_xyz[0], o_xyz[1], o_xyz[2]])
    rot_water_coordinates.append([rot_h1_xyz[0], rot_h1_xyz[1], rot_h1_xyz[2]])
    rot_water_coordinates.append([rot_h2_xyz[0], rot_h2_xyz[1], rot_h2_xyz[2]])

    return rot_water_xyz, rot_water_coordinates

  def add_waters(self, max_waters: int, stopping_criteria = 10):
    '''
    Adds water molecules up to max waters. Tries randomly adding waters and fails 
    if it is too close to an existing atom. 

      Args:
        max_waters: maximum waters to add
        stopping_criteria: number of failed attempts before stopping
      Returns:
        molecule_text: string of coordinates for the molecule with waters
    '''
    f = open(self.filename,"r")
    lines = f.readlines()
    f.seek(0)
    molecule_text = f.read()
    f.close()
  
    max_values, min_values = self.get_box_size()

    waters_to_add = []
    add_water = True

    water_counter = 0
    fail_counter = []
    
    for _ in range(max_waters):
        add_water = True

        if len(fail_counter) >= stopping_criteria:
            if 0 not in fail_counter[-stopping_criteria:]:
                print("Stopping criteria met! Exiting water addition")
                break
        
        new_water = []
        for i in range(3):
            new_water.append(random.uniform(min_values[i],max_values[i]))
           
        for row in lines[2:]:
            parts = row.split()
            mol_vec = [parts[1],parts[2],parts[3]]
            distance = self.calc_distance(new_water, mol_vec)
            if distance < self.water_vdw_rad:
                #print(f"distance: {distance} is close to another atom, breaking loop")
                fail_counter.append(1)
                add_water = False
                break
                
        if len(waters_to_add) > 0:
          for water in waters_to_add:
            for row in water:
                distance = self.calc_distance(new_water,row)
                if distance < self.water_vdw_rad:
                    #print(f"distance: {distance} is close to another water, breaking loop")
                    fail_counter.append(1)
                    add_water = False
                    break
                
        if add_water:
            water_counter += 1
            fail_counter.append(0)
            new_water_string, new_water_coordinates = self.get_water_coordinates(new_water)
            molecule_text += new_water_string
            waters_to_add.append(new_water_coordinates)

    old_length = int(lines[0])
    new_length = old_length + 3*water_counter
    molecule_text = f"{new_length}" + molecule_text[2:]

    print(f"Added {water_counter}/{max_waters} waters.")
    print("==================================================")
    return molecule_text

def smiles_to_atoms(smiles: str, charge = 0, spin = 1): # -> ase.Atoms:
  '''
  receives a smiles string and returns an ASE atoms object. Adds Hs to
  molecule, optimizes with MMFF by RDKit. Makes and XYZ string, 
  and an ASE atoms object.

    Args:
      smiles: SMILES string for molecule
      charge: charge of molecule
      spin: spin multiplicity of molecule

    Returns:
      atoms: ASE atoms object
  '''
  xyz_list = []
  atoms_list = ""
  mol = Chem.MolFromSmiles(smiles)
  molH = Chem.AddHs(mol)
  AllChem.EmbedMolecule(molH)
  AllChem.MMFFOptimizeMolecule(molH)
  xyz_string = f"{molH.GetNumAtoms()}\n\n"
  for atom in molH.GetAtoms():
    atoms_list += atom.GetSymbol()
    pos = molH.GetConformer().GetAtomPosition(atom.GetIdx())
    temp_tuple = (pos[0], pos[1], pos[2])
    xyz_list.append(temp_tuple)
    xyz_string += f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}\n"

  atoms = Atoms(atoms_list,xyz_list)  
  atoms.info['charge'] = charge
  atoms.info['spin'] = spin

  return atoms  

def XYZ_to_atoms(xyz_file: str, charges = None, spins = None) -> ase.Atoms:
  '''
  receives an XYZ file with one or more molecules and returns a list of ASE atoms objects.
  Updates charge and spin info for each molecule.

    Args:
      xyz_file: XYZ file with one or more molecules
      charges: list of charges for each molecule
      spins: list of spin multiplicities for each molecule

    Returns:
      all_mols: list of ASE atoms objects
  '''
  all_mols = []
  for mol in ase.io.iread(xyz_file, format="xyz"):
    all_mols.append(mol)

  if charges == None:
    charges = [0] * len(all_mols)
  if spins == None:
    spins = [1] * len(all_mols)
  
  for mol, charge, spin in zip(all_mols, charges, spins):
    mol.info['charge'] = charge
    mol.info['spin'] = spin

  return all_mols

def opt_energy(mol: ase.Atoms, calculator: FAIRChemCalculator, opt_flag = True,
               constraints_flag = False, constraints_list = []):
  '''
  Receives an ASE atoms object and calculates the energy using the UMA MLIP.
  if prompted, optimizes the structure and returns the optimized energy.

    Args:
      mol: ASE atoms object
      calculator: FAIRChemCalculator object
      opt_flag: boolean indicating whether to optimize the structure
    Returns:
      energy: energy of molecule in Hartree
  '''
  mol.calc = calculator
  initial_energy = mol.get_potential_energy()
  print(f"Initial energy: {0.0367493*initial_energy:.6f} ha")
  if opt_flag:

    if constraints_flag == True and len(constraints_list) != 0:
      c = FixAtoms(indices = constraints_list)
      mol.set_constraint(c)

    opt = BFGS(mol)
    opt.run(fmax=0.10)
    energy = mol.get_potential_energy()
    print(f"Final energy: {0.0367493*energy:.6f} ha")
    print(f"Energy difference: {0.0367493*(energy-initial_energy):.6f} ha")
    return 0.0367493*energy

  return 0.0367493*initial_energy

def atoms_to_xyz(mol: ase.Atoms, filename: str):
  '''
    Receives an atoms object and a filename and writes an XYZ file.
    
        Args:
            mol: atoms object
            filename: filename for writing
        Returns:
            None; writes file
  '''
  ase.io.write(filename+".xyz", mol, format="xyz")

def sdf_to_xyz(sdf_file: str, xyz_file = None):
  '''
    Takes and sdf file and produces an xyz file.

      Args:
        sdf_file: file to process
        xyz_file (optional): name for xyz file
  '''
  suppl = Chem.SDMolSupplier(sdf_file)

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
      xyz_string += f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}\n"
    total_xyz_list.append(xyz_string)

    if xyz_file == None:
      xyz_file = sdf_file.replace(".sdf",".xyz").replace(".SDF",".xyz")
    
    f = open(xyz_file,"w")
    f.write(xyz_string)
    f.close()

  return total_xyz_list

def xyz_to_sdf(xyz_file: str, sdf_file = None):
  '''
    Takes and xyz file and produces an sdf file.

      Args:
        xyz_file: file to process
        sdf_file (optional): name for sdf file
      Returns:
        None; writes file
  '''
  if sdf_file == None:
    sdf_file = xyz_file.replace(".xyz",".sdf").replace(".XYZ",".sdf")

  mol = Chem.MolFromXYZFile(xyz_file)
  writer = Chem.SDWriter(sdf_file)
  writer.write(mol)
  writer.close()

def rescoring (df_raw, ref_col: str, comp_col: str, step_size: int ):
  '''
    rescores variables, compares rankings of different variables and returns accuracy, produces image of ranked molecules.
    Args: 
      df_raw: dataframe with data to analyse
      ref_col: column to be ranked against
      comp_col: column to be ranked
      step_size: number of variables in group for accuracy measurement
    Returns:
      accuracy: if the same two molecules appear in the same group
      img: images of the molecues ranked, next to each other for comparison 
  '''
  
  df = df_raw.copy()
  ref_list = df[ref_col].to_list() 
  smiles_list = df["smiles"].to_list()

  df.sort_values(by=[comp_col], inplace=True)
  new_list = df["smiles"].to_list() 
  number_correct = 0

  total_number = len(smiles_list) - (len(smiles_list)%step_size)
  ref_set = set()
  comp_set = set()
  for i in range(0, total_number, step_size):
    for j in range(i, i+step_size, 1):
      ref_set.add(smiles_list[j])
      comp_set.add(new_list[j])
   
    for smile in comp_set:
      if smile in ref_set:
        number_correct += 1
    ref_set = set()
    comp_set = set()
  
  mols = []
  legends = []
  for ref,comp in zip(smiles_list, new_list):
    mol_1 = Chem.MolFromSmiles(ref)
    mol_2 = Chem.MolFromSmiles(comp)
    mols.append(mol_1)
    mols.append(mol_2)
    legends.append("experimental")
    legends.append(comp_col)
  img = Draw.MolsToGridImage(mols, legends = legends, molsPerRow=2, subImgSize=(300,300))
  pic = img.data

  accuracy = number_correct/total_number
  return accuracy, img
