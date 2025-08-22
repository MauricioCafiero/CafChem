import psi4
import numpy
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_psi4(smiles: str, charge = 0, spin = 1):
  '''
  receives a smiles string and returns a psi4 input. Adds Hs to
  molecule, optimizes with MMFF by RDKit. Makes and XYZ string.

    Args:
      smiles: SMILES string for molecule
      charge: charge of molecule
      spin: spin multiplicity of molecule

    Returns:
      p4_input: psi4 geometry object
  '''
  xyz_list = []
  atoms_list = ""
  mol = Chem.MolFromSmiles(smiles)
  molH = Chem.AddHs(mol)
  AllChem.EmbedMolecule(molH)
  AllChem.MMFFOptimizeMolecule(molH)
  xyz_string = f"{charge} {spin}\n"
  for atom in molH.GetAtoms():
    atoms_list += atom.GetSymbol()
    pos = molH.GetConformer().GetAtomPosition(atom.GetIdx())
    xyz_string += f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}\n"

  
  p4_input = psi4.geometry(xyz_string)
  return p4_input
  
def XYZ_to_psi4(file_path: str, charge = 0, spin = 1):
  '''
  receives a path to an XYZ file and returns a psi4 input.

    Args:
      file_path: path to XYZ file
      charge: charge of molecule
      spin: spin multiplicity of molecule

    Returns:
      p4_input: psi4 geometry object
  '''
  f = open(file_path, "r")
  xyz_list = f.readlines()
  f.close()
  
  xyz_list = xyz_list[2:]
  xyz_string = f"{charge} {spin}\n"
  xyz_string += "\n".join(xyz_list)

  p4_input = psi4.geometry(xyz_string)
  return p4_input

def XYZ_to_dimer(file_path: str, atoms_1: int, charge1 = 0, spin1 = 1,
                 charge2 = 0, spin2 = 1):
  '''
  receives a path to an XYZ file and returns a psi4 input.

    Args:
      file_path: path to XYZ file
      charge: charge of molecule
      spin: spin multiplicity of molecule

    Returns:
      p4_input: psi4 geometry object
  '''
  f = open(file_path, "r")
  xyz_list = f.readlines()
  f.close()
  
  xyz_list_1 = xyz_list[2:2+atoms_1]
  xyz_list_2 = xyz_list[2+atoms_1:]
  
  xyz_string = f"{charge1} {spin1}\n"
  xyz_string += "\n".join(xyz_list_1)
  xyz_string += "\n"
  xyz_string += "--\n"
  xyz_string += f"{charge2} {spin2}\n"
  xyz_string += "\n".join(xyz_list_2)

  p4_input = psi4.geometry(xyz_string)
  return p4_input

def psi4_to_XYZ(p4geom, file_path: str):
  '''
    Takes a Psi4 geometry object and returns an XYZ file.

    Args:
      p4geom: Psi4 geometry object
      file_path: path to save XYZ file
    Returns:
      None
  '''
  p4geom.save_xyz_file(file_path, 1)
  

class psi4_calc():
  '''
    Class to run Psi4 calculations.
  '''
  def __init__(self, memory: int, num_threads: int, basis: str, functional: str):
    '''
      Runs energy, optimization and SAPT calculations using Psi4

        Args:
          memory: memory in GB
          num_threads: number of threads to use
          basis: basis set to use
          functional: functional to use
    '''
    self.memory = memory
    self.num_threads = num_threads
    self.basis = basis
    self.functional = functional
	
  def calc_energy(self,molecule, return_wfn = False):
    '''
      Runs energy calculation using Psi4

        Args:
          molecule: the psi4 molecule object to calculate
          return_wfn: whether or not to return the wave function
        Returns:
          energy: the energy in hartree
          kcals: the energy in kilocalories
    '''
    total_mem = self.memory*1E9
    psi4.set_memory(total_mem)
    psi4.set_num_threads(self.num_threads)
    
    energy = psi4.energy(f"scf/{self.basis}", dft_functional = self.functional,
                              molecule=molecule, return_wfn=return_wfn)
							  
    return energy, energy*627.5095
    
  def optimize(self, molecule, return_wfn = False):
    '''
      Runs geometry optimization using Psi4

        Args:
          molecule: the psi4 molecule object to calculate
          return_wfn: whether or not to return the wave function
        Returns:
          energy: the energy in hartree
          kcals: the energy in kilocalories
          molecule: the optimized molecule
    '''
    total_mem = self.memory*1E9
    psi4.set_memory(total_mem)
    psi4.set_num_threads(self.num_threads)
    
    energy = psi4.optimize(f"scf/{self.basis}", dft_functional = self.functional,
                              molecule=molecule)
							  
    return energy, energy*627.5095, molecule
  
  def sapt(self, molecule):
    '''
      Runs SAPT calculation using Psi4. Zeroth order using the jun-cc-pvdz
      basis set.

        Args:
          molecule: the psi4 molecule object to calculate
        Returns:
          energy: the energy in hartree
          kcals: the energy in kilocalories
        Prints: 
          SAPT component energies in hartrees and percentages
    '''
    psi4.set_options({'scf_type': 'df', 'freeze_core': True})

    energy = psi4.energy('sapt0/jun-cc-pvdz', molecule=molecule)

    sapt_electrostatic = psi4.variable('SAPT elst ENERGY')
    sapt_exchange = psi4.variable('SAPT exch ENERGY')
    sapt_dispersion = psi4.variable('SAPT DISP ENERGY')
    sapt_induction = psi4.variable('SAPT IND ENERGY')
    total_sapt = psi4.variable('SAPT TOTAL ENERGY')

    percent_electrostatic = 100*sapt_electrostatic/total_sapt
    percent_exchange = 100*sapt_exchange/total_sapt
    percent_dispersion = 100*sapt_dispersion/total_sapt
    percent_induction = 100*sapt_induction/total_sapt

    print("Printing SAPT energies:")
    print("=====================================================")
    print(f"Electrostatic: {sapt_electrostatic:20.6f} ha {percent_electrostatic:10.2f}%")
    print(f"Exchange:      {sapt_exchange:20.6f} ha {percent_exchange:10.2f}%")
    print(f"Dispersion:    {sapt_dispersion:20.6f} ha {percent_dispersion:10.2f}%")
    print(f"Induction:     {sapt_induction:20.6f} ha {percent_induction:10.2f}%")
    print(f"Total SAPT:    {total_sapt:20.6f} ha")

    return energy, energy*627.5095