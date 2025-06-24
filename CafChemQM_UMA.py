import pandas as pd
import matplotlib.pyplot as plt
from ase import units
import ase.io
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from ase.md import MDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import shutil
import numpy as np

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

def get_current_xyz(mol: ase.Atoms, filename):
  '''
  receives an ASE atoms object and saves the current structure as an XYZ file.

    Args:
      mol: ASE atoms object
      filename: name of file to save
    Returns:
      None; saves XYZ file
  '''
  mol.write(filename+".xyz", format="xyz")

def calculate_vibrations(mol: ase.Atoms, calculator: FAIRChemCalculator, struc_type = "nonlinear"):
  '''
  receives an ASE atoms object and calculates the vibrational frequencies using the UMA MLIP.

    Args:
      mol: ASE atoms object
      calculator: FAIRChem
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

def calculate_thermodynamics(energy: float, all_vibs: list, atoms: ase.Atoms,
                             struc_type = "nonlinear", spin = 0, sym = 1, temp = 310.15,
                             pressure = 101325.0):
  '''
  receives energy, a list of raw vibrations, an atoms object, and other properties and
  calculates the Gibbs energy, enthalpy and entropy with the Ideal Gas approximation.

    Args:
      energy: energy of molecule in Hartree
      all_vibs: list of raw vibrational frequencies
      atoms: ASE atoms object
      struc_type: linear or non-linear
      spin: spin of molecule
      sym: symmetry number of molecule
      temp: temperature in K
      pressure: pressure in Pa
    Returns:
      H: enthalpy in kJ/mol
      S: entropy in kJ/molK
      G: Gibbs energy in kJ/mol
  '''
  thermo = IdealGasThermo(vib_energies = all_vibs, potentialenergy = energy/0.0367493, atoms = atoms,
                       geometry = struc_type, symmetrynumber = sym, spin = spin)
  H = 96.485*thermo.get_enthalpy(temperature=temp)
  S = 96.485*thermo.get_entropy(temperature=temp, pressure = pressure)
  G = 96.485*thermo.get_gibbs_energy(temperature=temp, pressure = pressure)

  return H, S, G

def reaction_thermo(rxn: dict, calculator: FAIRChemCalculator, temp = 310.15, pressure = 101325.0):
  '''
  Expects a dictionary containing the followin information for reactants and products: SMILES
  strings, stoichiometric coefficients, charges, spins, and structure types. Calculates energy, 
  vibrations and thrmodynamic properties for each molecule.

    Args:
      rxn: dictionary containing the following information for reactants and products: SMILES
      strings, stoichiometric coefficients, charges, spins, symmetry and structure types.
      calculator: FAIRChemCalculator object
      temp: temperature in K
      pressure: pressure in Pa
    Returns
      Grxn: Gibbs energy of reaction in kJ/mol
      Hrxn: Enthalpy of reaction in kJ/mol
      Srxn: Entropy of reaction in kJ/molK
  '''
  gibbs_reactants = 0 
  gibbs_products = 0
  enthalpy_reactants = 0
  entropy_reactants = 0
  enthalpy_products = 0
  entropy_products = 0

  for mol,stoich, charge, spin, struct, sym in zip(rxn['reactants'], rxn['reactant stoich'], 
                                              rxn['reactant_charge'], rxn['reactant_spin'], rxn['reactant_struct'],
                                              rxn['reactant_sym']):
    atoms = smiles_to_atoms(mol, charge, spin)
    energy = opt_energy(atoms, calculator  = calculator, opt_flag = True)
    all_vibs, _ = calculate_vibrations(atoms, calculator, struc_type = struct)
    H, S, G = calculate_thermodynamics(energy, all_vibs, atoms, struc_type = struct, spin = (spin-1)/2, sym = sym)
    gibbs_reactants += G*stoich
    enthalpy_reactants += H*stoich
    entropy_reactants += S*stoich

  print(f"Reactants Gibbs: {gibbs_reactants:.3f} kJ/mol")
  print(f"Reactants Enthalpy: {enthalpy_reactants:.3f} kJ/mol")
  print(f"Reactants Entropy: {entropy_reactants:.3f} kJ/molK")
  
  for mol,stoich, charge, spin, struct, sym in zip(rxn['products'], rxn['product_stoich'], 
                                      rxn['product_charge'], rxn['product_spin'], rxn['product_struct'],
                                      rxn['reactant_sym']):
    atoms = smiles_to_atoms(mol, charge, spin)
    energy = opt_energy(atoms, calculator  = calculator, opt_flag = True)
    all_vibs, _ = calculate_vibrations(atoms, calculator, struc_type = struct)
    H, S, G = calculate_thermodynamics(energy, all_vibs, atoms, struc_type = struct, spin = (spin-1)/2, sym = sym)
    gibbs_products += G*stoich
    enthalpy_products += H*stoich
    entropy_products += S*stoich
  
  print(f"Products Gibbs: {gibbs_products:.3f} kJ/mol")
  print(f"Products Enthalpy: {enthalpy_products:.3f} kJ/mol")
  print(f"Products Entropy: {entropy_products:.3f} kJ/molK")
  print("========================================================")
  Grxn = gibbs_products - gibbs_reactants 
  Hrxn = enthalpy_products - enthalpy_reactants
  Srxn = entropy_products - entropy_reactants
  print(f"Reaction Gibbs: {Grxn:.3f} kJ/mol")
  print(f"Reaction Enthalpy: {Hrxn:.3f} kJ/mol")
  print(f"Reaction Entropy: {Srxn:.3f} kJ/molK")
  print("========================================================")

  return Grxn, Hrxn, Srxn

def run_dynamics(filename: str, mol: ase.Atoms, calculator: FAIRChemCalculator,
    method = "velocity",
    temperature =  298.15, timestep = 0.5 * units.fs, friction = 0.01 / units.fs, 
    cell_size = 25.0, total_steps = 100, traj_interval = 1, log_interval = 1):
  '''
  receives a filename (for saving), an atoms object, a calculator, a method specification 
  (velocity verlet or langevin) and performs a dynamics simulation.

    Args:
      filename: name of file to save
      mol: ASE atoms object
      calculator: FAIRChemCalculator object
      method: velocity verlet or langevin
      temperature: temperature in K (for Langevin)
      timestep: timestep in fs (for both)
      cell_size: size of cell in Angstroms (for langevin)
      friction: friction coefficient (for Langevin)
    
    Returns:
      None; saves XYZ file, log file and displays energy vs time plot
  '''
  output_file = filename + "_md.xyz"
  log_file = filename + "_md_nvt.log"

  
  mol.calc = calculator

  if method == "langevin":
    mol.set_cell([cell_size] * 3)
    MaxwellBoltzmannDistribution(mol, temperature_K=temperature)
    dyn = Langevin(mol, timestep, temperature_K=temperature, friction=friction)
  elif method == "velocity":
    dyn = VelocityVerlet(mol,timestep)

  dyn.attach(
      lambda: ase.io.write(output_file, mol, append=True, format = "xyz"), interval=traj_interval
  )
  dyn.attach(MDLogger(dyn, mol, "md_nvt.log"), interval=log_interval)

  dyn.run(steps=total_steps)

  df = pd.read_table("md_nvt.log", sep="\s+")
  x = df["Etot[eV]"].to_list()
  y = df["Time[ps]"].to_list()

  plt.plot(y, x)
  plt.xlabel("Time (fs)")
  plt.ylabel("Energy (eV)")
  plt.show()

def show_frame(xyz_file: str, frame_number = 1):
  '''
  receives an XYZ file and a frame number and shows the frame in 3D

    Args:
      xyz_file: XYZ file with one or more molecules
      frame_number: frame number to show in 3D
    Returns:
      None; shows frame in 3D
  '''
  with open(xyz_file, "r") as f:
    lines = f.readlines()

  num_atoms = int(lines[0].strip())  # Number of atoms from the first line
  frame_start = frame_number * (num_atoms + 2)  # Start of the desired frame
  frame_lines = lines[frame_start : frame_start + num_atoms + 2]  # Extract frame
  visualize_molecule("".join(frame_lines))

def visualize_molecule(xyz_string: str):
  '''
    input an XYZ string to vosualize the molecule in 3D
  '''
  viewer = py3Dmol.view(width=800, height=400)
  viewer.addModel(xyz_string, "xyz")  
  viewer.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
  viewer.zoomTo()
  viewer.show()