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
from ase.constraints import FixAtoms
from ase.md import MDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import shutil
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip

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
    temperature =  298.15, timestep = 1.0 * units.fs, friction = 0.01 / units.fs, 
    cell_size = 25.0, total_steps = 200, traj_interval = 1, log_interval = 1):
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
  mol.get_positions()
  
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
  dyn.attach(MDLogger(dyn, mol, log_file), interval=log_interval)

  dyn.run(steps=total_steps)

  df = pd.read_table(log_file, sep="\s+")
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