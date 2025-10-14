import pandas as pd
import matplotlib.pyplot as plt
from ase import units
import ase.io
from ase import Atoms
from ase.optimize import LBFGSLineSearch, BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.constraints import FixAtoms
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import shutil
import numpy as np
from ase.units import Bohr, Hartree

from skala.ase import Skala

def smiles_to_atoms(smiles: str, charge = 0, spin = 1) -> ase.Atoms:
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
  #atoms.positions = atoms.get_positions() * Bohr

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
    #mol.positions = mol.get_positions() * Bohr
    mol.info['charge'] = charge
    mol.info['spin'] = spin

  return all_mols

def atoms_to_xyz(mol: ase.Atoms, filename: str, file_flag=False):
  '''
    Receives an atoms object and a filename and writes an XYZ file.
    
        Args:
            mol: atoms object
            filename: filename for writing
            file_flag: boolean indicating whether to write to file
        Returns:
            XYZ string
            None; writes file
  '''
  #atoms.positions = atoms.get_positions() / Bohr
  if file_flag == True:
    ase.io.write(filename+".xyz", mol, format="xyz")
    #atoms.positions = atoms.get_positions() * Bohr
  else:
    atom_symbols = mol.get_chemical_symbols()
    atom_positions = mol.get_positions()
    xyz_string = f"{mol.get_global_number_of_atoms()}\n\n"
    i=0
    for atom_symbol, atom_position in zip(atom_symbols, atom_positions):
      if i != len(atom_symbols)-1:
        xyz_string += f"{atom_symbol} {atom_position[0]} {atom_position[1]} {atom_position[2]}\n"
        i += 1
      else:
        xyz_string += f"{atom_symbol} {atom_position[0]} {atom_position[1]} {atom_position[2]}"
    #atoms.positions = atoms.get_positions() * Bohr
    return xyz_string

def opt_energy(mol: ase.Atoms, opt_flag = True,
               constraints_flag = False, constraints_list = [], opt_type='BFGS'):
  '''
  Receives an ASE atoms object and calculates the energy using the UMA MLIP.
  if prompted, optimizes the structure and returns the optimized energy.

    Args:
      mol: ASE atoms object
      opt_flag: boolean indicating whether to optimize the structure
      constraints_flag: boolean indicating whether to apply constraints
      constraints_list: list of indices of atoms to apply constraints to
      opt_type: type of optimization algorithm to use, options are 'BFGS' and 'LBFGSLineSearcg'
    Returns:
      energy: energy of molecule in Hartree
  '''
  initial_energy = mol.get_potential_energy()
  print(f"Initial energy: {0.0367493*initial_energy:.6f} ha")
  if opt_flag:

    if constraints_flag == True and len(constraints_list) != 0:
      c = FixAtoms(indices = constraints_list)
      mol.set_constraint(c)

    if opt_type == 'BFGS':
      opt = BFGS(mol)
    else:
      opt = LBFGSLineSearch(mol)
    opt.run(fmax=0.10)
    energy = mol.get_potential_energy()
    print(f"Final energy: {0.0367493*energy:.6f} ha")
    print(f"Energy difference: {0.0367493*(energy-initial_energy):.6f} ha")
    return 0.0367493*energy

  return 0.0367493*initial_energy

def get_current_xyz(mol: ase.Atoms, filename):
  '''
  receives an ASE atoms object and saves the current structure as an XYZ file.

    Args:
      mol: ASE atoms object
      filename: name of file to save
    Returns:
      None; saves XYZ file
  '''
  mol.positions = mol.get_positions() / Bohr
  mol.write(filename+".xyz", format="xyz")

def calc_dipole(atoms):
  '''
  '''
  dipole = atoms.get_dipole_moment()

  magnitude = np.linalg.norm(dipole)
  axes = ['x','y','z']

  print(f"Dipole moment magnitude: {magnitude:.3f}")
  print("Dipole moment vector:")
  print("=====================================================")
  for axis, comp in zip(axes,dipole):
    print(f"{axis}-component: {comp:8.3f}") 
  
  return dipole

def calculate_vibrations(mol: ase.Atoms, struc_type = "nonlinear"):
  '''
  receives an ASE atoms object and calculates the vibrational frequencies using the UMA MLIP.

    Args:
      mol: ASE atoms object
      struc_type: linear or non-linear
    Returns:
      list of vibrational frequencies
  '''
  vib = Vibrations(mol)
  vib.run()
  vib_energies = vib.get_energies()

  real_vibs = []
  number_imaginary = 0
  for vib in vib_energies:
    if vib.imag < 0.00001:
      real_vibs.append(8065.56*vib.real.item())
    else:
      number_imaginary += 1
  
  num_atoms = mol.get_global_number_of_atoms()
  struct_type = struc_type.lower()
  if struct_type == "non-linear" or struct_type == "nonlinear":
    non_vib_dof = 6 - number_imaginary
  else:
    non_vib_dof = 5 - number_imaginary
  vib_dof = 3*num_atoms - non_vib_dof

  out_text = ""
  for i,vib in enumerate(real_vibs[3*num_atoms - vib_dof:]):
    out_text += f"Vibrational frequency {i+1}: {vib:.3f} cm-1\n"
  
  out_text += f"Also calculated the following low frequency motions:\n"
  for i,vib in enumerate(real_vibs[:3*num_atoms - vib_dof]):
    out_text += f"Frequency {i+1}: {vib:.3f} cm-1\n"
  
  out_text += f"Also found the following number of imaginary frequencies: {number_imaginary}"

  shutil.rmtree('vib')
  print(out_text)
  return vib_energies, real_vibs

def visualize_molecule(xyz_string: str):
  '''
    input an XYZ string to vosualize the molecule in 3D
  '''
  viewer = py3Dmol.view(width=800, height=400)
  viewer.addModel(xyz_string, "xyz")  
  viewer.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
  viewer.zoomTo()
  viewer.show()

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

def test_units(lines, a, b):
  '''
    Tests the units of the xyz file.

    Args:
      lines: xyz string
      a: index of first atom
      b: index of second atom
    Returns:
      None; prints distance
  '''
  distance = 0.0
  v1 = lines[1+a][2:].split()
  v2 = lines[1+b][2:].split()
  for comp1, comp2 in zip(v1,v2):
    #print((float(comp1)-float(comp2))**2)
    distance += (float(comp1)-float(comp2))**2
  distance = np.sqrt(distance)
  print(distance)