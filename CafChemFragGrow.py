import random
import re
import numpy as np
import copy
import math
import py3Dmol
import os
import ase.io
import torch
from ase.calculators.calculator import all_changes
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase import Atoms, Atom
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip

global HMGCR_data
HMGCR_data = {
        "file_location":"CafChem/data/HMGCR_dude_QM_site.xyz",
        "name": "HMGCR",
        "charge": 3,
        "spin": 1,
        "constraints": [1, 11, 16, 24, 33, 41, 54, 60, 72, 83, 92, 98, 107, 124, 132, 140, 148, 159, 168, 181],
        "size": 331
        }

global DRD2_data
DRD2_data = {
        "file_location":"CafChem/data/DRD2_dude_QM_site.xyz",
        "name": "DRD2",
        "charge": -1,
        "spin": 1,
        "constraints": [1, 10, 18, 27, 33, 42, 54, 62, 78, 89, 101, 110],
        "size": 216
        }

global MAOB_data
MAOB_data = {
        "file_location":"CafChem/data/MAOB_dude_QM_site.xyz",
        "name": "MAOB",
        "charge": -1,
        "spin": 1,
        "constraints": [1, 7, 12, 17, 22, 31, 39, 44, 51, 61, 67, 84, 89, 94, 111, 120, 129, 134, 139, 147, 161, 172, 180, 188, 197, 206, 218, 235, 242, 250, 256, 265, 272, 281, 295, 306, 322, 335, 342, 356, 361, 370, 375, 389, 398, 408],
        "size": 809
        }

global MAOBnoFAD_data
MAOBnoFAD_data = {
        "file_location":"CafChem/data/MAOBnoFAD_dude_QM_site.xyz",
        "name": "MAOBnoFAD",
        "charge": 1,
        "spin": 1,
        "constraints": [1, 7, 12, 17, 22, 31, 39, 44, 51, 61, 67, 84, 89, 94, 111, 120, 129, 134, 139, 147, 161, 172, 180, 188, 197, 206, 218, 235, 242, 250, 256, 265, 272, 281, 295, 306, 322, 335, 341, 355, 360, 369, 374, 388, 397, 407],
        "size": 727
        }

global ADRB2_data
ADRB2_data = {
        "file_location":"CafChem/data/ADRB2_dude_QM_site.xyz",
        "name": "ADRB2",
        "charge": -2,
        "spin": 1,
        "constraints": [1, 16, 25, 33, 41, 49, 58, 66, 78, 84, 92, 100, 107, 122, 133, 145, 156, 170, 179, 189],
        "size": 349
        }

def define_fragments():
    '''
    defines fragment objects, including:
        number of atoms, 
        fragment name,
        atoms list,
        coordinates list,
        fragment charge,
        fragment size (vdW size).

        Arguments: none
        Returns:
            frags: List of fragment objects
    '''
    frags  = []

    frags.append(
        {
            "num_atoms": 3,
            "name": "water",
            "atoms": ['O', 'H', 'H'],
            "coords": [
                [-0.21318180666666686, -0.3016452799999998, 0.0],
                [0.7468181933333335, -0.3016452799999998, 0.0],
                [-0.5336363866666667, 0.6032905600000005, 0.0]
            ],
            "charge": 0,
            "spin": 1,
            "size": 0.0
        }
    )

    frags.append(
        {
            "num_atoms": 2,
            "name": "hydrogen fluoride",
            "atoms": ['F', 'H'],
            "coords": [
                [0.43999999999999995, 0.0, 0.0],
                [-0.44000000000000006, 0.0, 0.0]
            ],
            "charge": 0,
            "spin": 1,
            "size": 0.0
        }
    )

    frags.append(
        {
            "num_atoms": 9,
            "name": "cyclopropyl",
            "atoms": ['C', 'H', 'H', 'C', 'H', 'H', 'C', 'H', 'H'],
            "coords": [
                [0.8409265555555555, 0.21681944444444448, -0.00021977777777777897],
                [1.4172875555555555, 0.3662924444444445, 0.9194202222222222],
                [1.4196975555555555, 0.3649514444444445, -0.9184847777777778],
                [-0.6084474444444444, 0.6198084444444445, -4.977777777777897e-05],
                [-1.0262474444444445, 1.0455204444444444, -0.9190577777777779],
                [-1.0253374444444445, 1.0459924444444444, 0.9189832222222222],
                [-0.23288344444444448, -0.8366265555555555, 0.00016022222222222102],
                [-0.3929504444444445, -1.4104465555555556, -0.9193117777777778],
                [-0.3920454444444445, -1.4123115555555557, 0.9185602222222222]
            ],
            "charge": 0,
            "spin": 1,
            "size": 0.0
        }
    )

    frags.append(
        {
            "num_atoms": 4,
            "name": "acetylene",
            "atoms": ['C', 'H', 'C', 'H'],
            "coords": [
                [-0.6006, 0.0, 0.0],
                [-1.6706, 0.0, 0.0],
                [0.6006, 0.0, 0.0],
                [1.6705999999999999, 0.0, 0.0],
            ],
            "charge": 0,
            "spin": 1,
            "size": 0.0
        }
    )

    frags.append(
        {
            "num_atoms": 6,
            "name": "methanol",
            "atoms": ["H","H","H","C","O","H"],
            "coords": [
                [0.7118493333333333, -0.42401449999999996, -0.8989041666666667],
                [0.7658053333333334, 1.1194095, -0.0008381666666668064],
                [0.7119883333333334, -0.4225884999999999, 0.8997058333333333],
                [0.33305833333333335, 0.10613550000000005, 6.83333333317826e-06],
                [-1.0598266666666665, 0.24917050000000007, 7.833333333207015e-06],
                [-1.4628746666666665, -0.6281125, 2.1833333333276528e-05]
            ],
            "charge": 0,
            "spin": 1,
            "size": 0.0
        }
    )

    frags.append(
        {
            "num_atoms": 12,
            "name": "phenyl",
            "atoms": ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            "coords": [
                [-0.6976150833333326, -1.2080162499999998, 0.0004925833333333333],
                [0.6975449166666672, -1.2080162499999998, 0.0004925833333333333],
                [1.395082916666667, -0.0002652499999997726, 0.0004925833333333333],
                [0.6974289166666674, 1.20824375, -0.0007064166666666667],
                [-0.6973960833333326, 1.2081657500000003, -0.0011854166666666666],
                [-1.3949970833333327, -4.024999999963086e-05, -0.00018941666666666673],
                [-1.2473740833333327, -2.16033325, 0.0009425833333333333],
                [1.2470529166666675, -2.1605292499999997, 0.0018075833333333334],
                [2.494762916666667, -0.0001852499999996926, 0.0011265833333333332],
                [1.2476289166666672, 2.1603867500000007, -0.0007654166666666667],
                [-1.2475180833333328, 2.16044675, -0.002138416666666667],
                [-2.4946010833333325, 0.00014275000000019133, -0.00036941666666666676]
            ],
            "charge": 0,
            "spin": 1,
            "size": 0.0
        }
    )
    
    return frags

def add_box(val, vdw_distance):
    '''
        Adds an extra vdW distance to a dimension

        Args:
            val: the existing dimension
            vdw_distance: distance to add
        Returns:
            new val: val + vdw distance
    '''
    new_val = val + vdw_distance
    return new_val
    
def sub_box(val, vdw_distance):
    '''
        Adds an extra vdW distance to a dimension

        Args:
            val: the existing dimension
            vdw_distance: distance to add
        Returns:
            new val: val - vdw distance
    '''
    new_val = val - vdw_distance
    return new_val

def calc_distance(ref,atom):
    '''
        Calculates the distance between two points

        Args:
            ref: a reference point
            atom: the new atom coordinates
        Returns:
            distance: the distance
    '''
    distance = 0.0
    for i in range(3):
        distance += (ref[i]-atom[i])**2
    distance = np.sqrt(distance)
    return distance

def get_frag_coordinates(new_origin: list, which_frag: dict):
    '''
        Given an origin set of coordinates for a new fragment, this function 
        reads in the standard fragment coordinates, rotates them randomly
        in each direction, and then translates them to the new origin.

        Args:
            new_origin: the location of the new fragment
            which_frag: a dictionary for the fragment in question.
    '''

    how_many_atoms = which_frag["num_atoms"]

    theta_x = random.uniform(0,2*np.pi)
    theta_y = random.uniform(0,2*np.pi)
    theta_z = random.uniform(0,2*np.pi)
    
    x_rotation_matrix = np.asarray(([1.0,0.0,0.0],[0.0,np.cos(theta_x),-np.sin(theta_x)],[0.0,np.sin(theta_x),np.cos(theta_x)])).reshape(3,3)
    y_rotation_matrix = np.asarray(([np.cos(theta_y),0.0,-np.sin(theta_y)],[0.0,1.0,0.0],[np.sin(theta_y),0.0,np.cos(theta_y)])).reshape(3,3)
    z_rotation_matrix = np.asarray(([np.cos(theta_z),-np.sin(theta_z),0.0],[np.sin(theta_z),np.cos(theta_z),0.0],[0.0,0.0,1.0])).reshape(3,3)

    atoms_list = []
    for vec in which_frag["coords"]:
        temp_vec = np.array(vec)

        rot_temp = np.matmul(x_rotation_matrix,temp_vec)
        rot_temp = np.matmul(y_rotation_matrix,rot_temp)
        rot_temp = np.matmul(z_rotation_matrix,rot_temp)

        temp_vec = rot_temp + new_origin
        temp_vec = list(temp_vec)
        atoms_list.append(temp_vec)

    return atoms_list

def show_fragment(frag_to_show: dict):
    '''
        accepts a fragment dictionary and displays atoms list and coordinates

        Args:
            frag_to_show: a fragment dictionary object
        Returns:
            None; prints fragment info
    '''
    test_list = []
    formula = ""
    for i,atom in enumerate(frag_to_show["atoms"]):
        geotup = (frag_to_show["coords"][i][0],frag_to_show["coords"][i][1],frag_to_show["coords"][i][2])
        test_list.append(geotup)
        formula += atom
    
    print(str(formula))
    #print(test_list)

    return formula,test_list

def get_fragment_cutoff(frag_to_calc: dict, vdw_frac: float):
    '''
        Accepts a fragment dictionary and calculates the spatial extent, aka, 
        the minnimum distance between the fragment and any atoms of the binding site.

        Args:
            frag_to_calc: fragment dictionary to calculate
            vdw_frac: a vdW distance (fraction of the spatial extent) to add to the cartesian extent
        Returns:
            cutoff: the distance value
    '''
    formula, test_list = show_fragment(frag_to_calc)
    atoms1 = Atoms(formula, test_list)
    
    distances = []
    for i in range(frag_to_calc["num_atoms"]):
        for j in range(i+1,frag_to_calc["num_atoms"],1):
            dis_vec = atoms1.positions[i] - atoms1.positions[j]
            dis = np.linalg.norm(dis_vec)
            distances.append(float(dis))
    
    maxval = max(distances)
    cutoff = maxval*(0.5 + vdw_frac)
    frag_to_calc["size"] = cutoff
    print(f"cutoff is {cutoff}")

    return cutoff

def get_binding_site_xyz(path_to_site: str):
    '''
        opens a XYZ file with a binding site and loads it

        Args:
            path_to_site: path to the binding site XYZ file
        Returns:
            all_molecules: array with binding site coordinates
            atoms_symbols: list of atoms symbols
    '''
    f = open(path_to_site, "r")

    start_token = ""
    start_token = re.compile(r"^\d\d(\d)?\s")
    
    total_input = f.readlines()
    f.close()

    number_molecules = 0
    for line in total_input:
        start_match = start_token.search(line)
        if start_match:
            number_molecules += 1
            molecule_size = line.strip("\n")
    
    molecule_size = int(molecule_size)
    print(f"There are {number_molecules} molecules with size: {molecule_size}")
    
    all_molecules = []
    temp_array = np.zeros((molecule_size,3))
    all_molecules.append(temp_array)

    current_line = 0
    atom_symbols = []
    
    internal_i = 0
    current_line += 2
    print(f"for {current_line}, {current_line+int(molecule_size)}")
    
    for j in range(current_line,current_line+molecule_size,1):
        temp_vec = total_input[j].split()   
        atom_symbols.append(temp_vec[0])
        for k in range(1,4,1):
            all_molecules[0][internal_i,k-1] = temp_vec[k]
        internal_i += 1
        current_line += 1

    return all_molecules,atom_symbols

def get_binding_site_dims(all_molecules: list, vdw_distance: float):
    '''
        Uses the array of atom positions to find the maximum and minimum dimensions

        Args:
            all_molecules: array of atom coordinates
            vdw_distance: extra distance to add to binding site extent
        Returns:
            max_values: np.array of maximum values
            min_values: np.array of minimum values
    '''
    max_values = np.zeros((3))
    min_values = np.zeros((3))
    x_list = []
    y_list = []
    z_list = []
    for row in all_molecules[0]:
        x_list.append(row[0]) 
        y_list.append(row[1])
        z_list.append(row[2]) 
    max_values[0] = np.max(x_list)
    max_values[1] = np.max(y_list)
    max_values[2] = np.max(z_list)
    min_values[0] = np.min(x_list)
    min_values[1] = np.min(y_list)
    min_values[2] = np.min(z_list)
    
    max_values = [add_box(val, vdw_distance) for val in max_values]
    min_values = [sub_box(val, vdw_distance) for val in min_values]
    
    dims = ["x","y","z"]
    print(f"Maximum dimensions after augmentation are:")
    for dim,maxes,mins in zip(dims,max_values,min_values):
        print(f"{dim} - Max: {maxes:10.3f}, Min: {mins:10.3f}")

    volume = 1.0
    for big,small in zip(max_values,min_values):
        volume *= (big - small)
    print(f"Volume is {volume} A^3")

    return max_values, min_values

def grow_fragments(all_molecules: list, frags: list, atom_symbols: list, number_tries: int, calculator,
                   bs_object: dict, max_values: list, min_values: list):
    '''
        Accepts the binding site coordinates and fragment dictionaries and 
        tries to add the fragments to the binding site.

        Args:
            all_molecules: list of binding site atom coordinates
            frags: list of fragment dictionaries
            number_tries: how many times at attempt fragment placement
            calculator: ASE calculator
            bs_object: dictionary for the binding site
            max_values: np.array of maximum values
            min_values: np.array of minimum values
        Returns:
            new_molecules: list of positions of successfully placed fragments
    '''
    try:
      os.mkdir("temp_files")
    except:
      print("temp_files directory already exists")
    
    new_molecules = []
    ies = []

    centers = []
    sigmas = []
    for i in range(3):
        centers.append((max_values[i] + min_values[i])/2)
        sigmas.append((max_values[i] - min_values[i])/4)

    for mi,molecule in enumerate(all_molecules):
        for frag in frags:
            new_sheet = []
            ie_sheet = []
            
            how_many_added = 0
            for _ in range(number_tries):
                add_mol = True
                new_mol_origin = np.empty((3))
                cutoff = frag["size"]
                
                for i in range(3):
                    new_mol_origin[i] = np.random.normal(loc = centers[i], scale = sigmas[i])
                
                for i in range(3):
                    distance = abs(new_mol_origin[i] - centers[i])
                    if distance > sigmas[i]*2:
                        add_mol = False
                        break   
                #calc distance between new_mol origin and centers and if greater than sigmas*2 then reject
        
                for row in molecule:
                    distance = calc_distance(new_mol_origin,row)
                    if distance < cutoff:
                        #print(f"distance: {distance} is close to another atom, breaking loop")
                        add_mol = False
                        break
                         
                if add_mol:
                    new_vec = get_frag_coordinates(new_mol_origin, frag)
                    new_vec = np.array(new_vec)
                    single_frag = np.append(molecule, new_vec, axis=0)
                    ie = calc_frag_energy(single_frag, frag, atom_symbols, calculator, bs_object["charge"],
                                          bs_object["spin"],frag["charge"],frag["spin"])
                    if ie < 0.0:
                        how_many_added += 1
                        print(f"adding fragment: {frag['name']}")
                        new_sheet.append(single_frag)
                        ie_sheet.append(ie)
                    # else:
                    #     print("fragment rejected due to repulsive energy.")

            print(f"Added {how_many_added} {frag['name']} fragments")
            new_molecules.append(new_sheet)
            ies.append(ie_sheet)
                    
    return new_molecules, ies

def save_xyz_files(new_molecules: list, frags: list, bs_object: dict, atom_symbols: list):
    '''
        accepts the new_molecules array and creates an XYZ file for each fragment placement

        Args:
            new_molecules: list of arrays of coordinates of the binding site and fragment
            frag: the dictionary for the fragment in question
            bs_object: dictionary for the binding site
            atom_symbols: list of atom symbols
        Returns:
            None; saves XYZ files
    '''
    try:
      os.mkdir("frag_files")
    except:
      print("frag_files directory already exists")
    
    try:
        files = os.listdir("frag_files")            
        files_to_remove = [file for file in files if (os.path.splitext(file)[1]==".xyz")]
        for file in files_to_remove:
          os.remove(f"frag_files/{file}")
    except:
        print("frag_files directory is empty")

    for k,frag in enumerate(frags):
        for j in range(len(new_molecules[k])):
            mol_file = f"frag_files/{bs_object['name']}_w_{frag['name']}{j}.xyz"
        
            all_symbols = atom_symbols + frag["atoms"]
            f = open(mol_file,"w")
            f.write(f"{len(all_symbols)}\n")
            f.write("\n")
            for i in range(len(new_molecules[k][j])):
                row_string = f"{all_symbols[i]}"
                for coord in new_molecules[k][j][i]:
                    row_string += f"    {coord}"
                if i != len(new_molecules[k][j]): 
                    f.write(row_string+"\n")
                else:
                    f.write(row_string)
            f.close()
            #print("File Written")
    
        print(f"{len(new_molecules[k])} files written for {frag['name']}.")

def calc_frag_energy(new_molecule: list, frag: dict, atom_symbols: list, calculator,
                     bs_charge, bs_spin, ligand_charge, ligand_spin):
    '''
        Calculates the energies of the fragments in the binding site

        Args:
            new_molecules: list of arrays of coordinates of the binding site and fragment
            frag: the dictionary for the fragment in question
            atom_symbols: list of atom symbols
            calculator: ASE calculator
            bs_charge: binding site charge
            bs_spin: binding site spin
            ligand_charge: ligand charge
            ligand_spin: ligand spin
        Returns:
            ies: the interaction energies between the binding site and the fragment
    '''
    path = "temp_files/"
    test_files = ["complex.xyz"]
    
    all_symbols = atom_symbols + frag["atoms"]

    f = open(path+test_files[0],"w")
    f.write(f"{len(all_symbols)}\n")
    f.write("\n")
    
    for i in range(len(new_molecule)):
        row_string = f"{all_symbols[i]}"
        for coord in new_molecule[i]:
            row_string += f"    {coord}"
        if i != len(new_molecule): 
            f.write(row_string+"\n")
        else:
            f.write(row_string)
    f.close()

    atoms_tot = ase.io.read(path+test_files[0], format="xyz")
    total_spin = bs_spin + ligand_spin - 1
    total_charge = bs_charge + ligand_charge
    atoms_tot.info.update({"spin": total_spin, "charge": total_charge})
    os.remove(path+test_files[0])

    bs_length = len(new_molecule) - frag["num_atoms"]
    atoms_bs = atoms_tot[:bs_length]
    atoms_bs.info.update({"spin": bs_spin, "charge": bs_charge})
    atoms_l = atoms_tot[bs_length:]
    atoms_l.info.update({"spin": ligand_spin, "charge": ligand_charge})
    atoms_to_calc = [atoms_tot,atoms_bs,atoms_l]

    results = []
    for atoms in atoms_to_calc:
        atoms.calc = calculator
        results.append(atoms.get_potential_energy())
        
    ie = 23.06035*(results[0] - results[1] - results[2])
    
    return ie

def view_frag_pose(frag_idx: int, pose_idx: int, frags: list, bs: dict):
    '''
      Displays a fragment pose

      Args:
        frag_idx: index of the fragment to view
        pose_idx: index of the pose to view
        frags: list of fragment dictionaries
        bs: dictionary for the binding site
      Returns:
        None; displays fragment pose
    '''
    frag_name = frags[frag_idx]["name"]
    view_file = mol_file = f"frag_files/{bs['name']}_w_{frag_name}{pose_idx}.xyz"
    f = open(view_file,"r")
    lines = f.readlines()
    mol_data = "".join(lines)
    f.close

    viewer = py3Dmol.view(width=800, height=400)
    viewer.addModel(mol_data, "xyz")  # Add the trajectory frame

    for i in range(bs["size"]):
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5}})

    for i in range(bs["size"],bs['size']+frags[frag_idx]["num_atoms"],1): 
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5, 'color': 'green'}})

    viewer.zoomTo()
    viewer.show()

def get_best_poses(ies: list, frags: list):
    '''
      Accepts the list of interaction energies and determines the best pose for each fragment

      Args:
        ies: list of interaction energies
        frags: list of fragment dictionaries
      Returns:
        best_pose_for_fragments: list of best poses for each fragment
    '''
    best_pose_for_fragments = []
    for energy,frag in zip(ies,frags):
        min = np.min(energy)
        min_idx = np.argmin(energy)
        best_pose_for_fragments.append(min_idx)
        print(f"best pose for {frag['name']} is: {min:.3f} at location: {min_idx}")

    return best_pose_for_fragments

def combine_best_poses(frags: list, bs: dict, new_molecules: list, best_pose_for_fragments: list):
  '''
    Accepts the new_molecules array and combines the best poses for each fragment

    Args:
      frags: list of fragment dictionaries
      bs: dictionary for the binding site
      new_molecules: list of arrays of coordinates of the binding site and fragment
      best_pose_for_fragments: list of best poses for each fragment
    Returns:
      combined_poses: list of combined best poses
      combined_atoms: list of combined atoms
      (also generates the combined.xyz file in the frag_files/ folder)
  '''
  combined_poses = []
  combined_atoms = []

  for idx_frag, frag in enumerate(frags):
      total_molecule = new_molecules[idx_frag][best_pose_for_fragments[idx_frag]]
      just_frag_array = total_molecule[bs["size"]:].tolist()
      # just_frag = []
      # for val in just_frag_array:
      #     just_frag.append(val.item())
      just_frag_array = list(just_frag_array)
      combined_poses = [*combined_poses, *just_frag_array]
      combined_atoms = [*combined_atoms, *frag["atoms"]]

  
  path = "frag_files/"
  test_files = ["combined.xyz"]
  
  all_symbols = combined_atoms

  f = open(path+test_files[0],"w")
  f.write(f"{len(all_symbols)}\n")
  f.write("\n")
  
  for i in range(len(combined_poses)):
      row_string = f"{all_symbols[i]}"
      for coord in combined_poses[i]:
          row_string += f"    {coord}"
      if i != len(combined_poses): 
          f.write(row_string+"\n")
      else:
          f.write(row_string)
  f.close()
  
  return combined_poses, combined_atoms

def view_combined_poses(bs: dict, frags: list):
    '''
      Displays the combined best poses

      Args:
        bs: dictionary for the binding site
        frags: list of fragment dictionaries
      Returns:
        None; displays fragment pose
    '''
    
    view_file = "frag_files/combined.xyz"
    bs_file = bs["file_location"]
    lines = []

    f = open(view_file,"r")
    lines1 = f.readlines()

    g = open(bs_file,"r")
    lines2 = g.readlines()

    lines.append(f"{int(lines1[0]) + int(lines2[0])}\n")
    lines.append("\n")

    for line in lines2[2:]:
      lines.append(line)
    for line in lines1[2:]:
      lines.append(line)

    f.close
    g.close
    mol_data = "".join(lines)
    f.close

    total_frags_length = 0
    for frag in frags:
      total_frags_length += frag["num_atoms"]

    viewer = py3Dmol.view(width=800, height=400)
    viewer.addModel(mol_data, "xyz")  # Add the trajectory frame

    for i in range(bs["size"]):
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5}})

    for i in range(bs["size"],bs['size']+total_frags_length,1): 
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5, 'color': 'green'}})
    viewer.zoomTo()
    viewer.show()

def view_all_poses_for_frag(which_frag: int, frags: list, bs: dict, new_molecules: list,
                            best_pose_for_fragments: list):
  '''
    Accepts the new_molecules array and combines the best poses for each fragment

    Args:
      which_frag: index of the fragment to view
      frags: list of fragment dictionaries
      bs: dictionary for the binding site
      new_molecules: list of arrays of coordinates of the binding site and fragment
      best_pose_for_fragments: list of best poses for each fragment
    Returns:
      None
      (generates the frag_combined.xyz file in the frag_files/ folder)
  '''
  combined_poses = []
  combined_atoms = []

  for sheet in new_molecules[which_frag]:
      just_frag_array = sheet[bs["size"]:].tolist()
      just_frag_array = list(just_frag_array)
      combined_poses = [*combined_poses, *just_frag_array]
      combined_atoms = [*combined_atoms, *frags[which_frag]["atoms"]]

  
  path = "frag_files/"
  test_files = [f"{frag['name']}_combined.xyz"]
  
  all_symbols = combined_atoms

  f = open(path+test_files[0],"w")
  f.write(f"{len(all_symbols)}\n")
  f.write("\n")
  
  for i in range(len(combined_poses)):
      row_string = f"{all_symbols[i]}"
      for coord in combined_poses[i]:
          row_string += f"    {coord}"
      if i != len(combined_poses): 
          f.write(row_string+"\n")
      else:
          f.write(row_string)
  f.close()

  bs_file = bs["file_location"]
  lines = []

  f = open(path+test_files[0],"r")
  lines1 = f.readlines()

  g = open(bs_file,"r")
  lines2 = g.readlines()

  lines.append(f"{int(lines1[0]) + int(lines2[0])}\n")
  lines.append("\n")

  for line in lines2[2:]:
    lines.append(line)
  for line in lines1[2:]:
    lines.append(line)

  f.close
  g.close
  mol_data = "".join(lines)

  total_frags_length = 0
  for _ in range(len(new_molecules[which_frag])):
    total_frags_length += frag["num_atoms"]

  viewer = py3Dmol.view(width=800, height=400)
  viewer.addModel(mol_data, "xyz")  # Add the trajectory frame

  for i in range(bs["size"]):
    viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5}})

  location_best_pose = bs['size'] + best_pose_for_fragments[which_frag]*frags[which_frag]["num_atoms"]
  range_best_pose = [i for i in range(location_best_pose,location_best_pose+frags[which_frag]["num_atoms"])]

  for i in range(bs["size"],bs['size']+total_frags_length,1): 
    if i in range_best_pose:
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5, 'color': 'green'}})
    else:
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5, 'color': 'pink'}})
  viewer.zoomTo()
  viewer.show()

def average_pose_for_frag(which_frag: int, frags: list, bs: dict, new_molecules: list, atom_symbols: list,
                          display_flag: bool):
  '''
    Accepts the new_molecules array and combines all poses for the specified fragment

    Args:
      which_frag: index of the fragment to view
      frags: list of fragment dictionaries
      bs: dictionary for the binding site
      new_molecules: list of arrays of coordinates of the binding site and fragment
      atom_symbols: list of atom symbols
    Returns:
      new_coordinates: list of coordinates of the average pose for the fragment
      (also generates the frag_average.xyz file in the frag_files/ folder)
  '''

  total_center = [0.0]*3

  for sheet in new_molecules[which_frag]:
    center = [0.0]*3
    just_frag_array = sheet[bs["size"]:].tolist()
    just_frag_array = list(just_frag_array)

    for row in just_frag_array:
      for i in range(3):
        center[i] += row[i]

    for i in range(3):
      center[i] /= len(just_frag_array)
      total_center[i] += center[i]

  for i in range(3):
    total_center[i] /= len(new_molecules[which_frag])
  
  new_coordinates = get_frag_coordinates(total_center, frags[which_frag])
  new_coordinates = np.array(new_coordinates)
  new_sheet = np.append(new_molecules[which_frag][0][:bs["size"]], new_coordinates, axis=0)

  path = "frag_files/"
  test_files = [f"{frags[which_frag]['name']}_average.xyz"]
  
  all_symbols = atom_symbols + frags[which_frag]["atoms"]

  f = open(path+test_files[0],"w")
  f.write(f"{len(all_symbols)}\n")
  f.write("\n")
  
  for i in range(len(new_sheet)):
      row_string = f"{all_symbols[i]}"
      for coord in new_sheet[i]:
          row_string += f"    {coord}"
      if i != len(new_sheet): 
          f.write(row_string+"\n")
      else:
          f.write(row_string)
  f.close()

  if display_flag: 
    f = open(path+test_files[0],"r")
    lines = f.readlines()
    mol_data = "".join(lines)

    viewer = py3Dmol.view(width=800, height=400)
    viewer.addModel(mol_data, "xyz")  # Add the trajectory frame

    for i in range(bs["size"]):
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5}})

    for i in range(bs["size"],bs['size']+frags[which_frag]["num_atoms"],1): 
      viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5, 'color': 'green'}})
    
    viewer.zoomTo()
    viewer.show()

  return new_coordinates

def combine_average_poses(frags: list, bs: dict, new_molecules: list, atom_symbols: list):
  '''
  '''
  combined_poses = []
  combined_atoms = []

  for idx_frag, frag in enumerate(frags):
      just_frag_array = average_pose_for_frag(idx_frag, frags, bs, new_molecules, atom_symbols, display_flag = False)
      just_frag_array = list(just_frag_array)
      combined_poses = [*combined_poses, *just_frag_array]
      combined_atoms = [*combined_atoms, *frag["atoms"]]

  path = "frag_files/"
  test_files = ["combined_average.xyz"]
  
  all_symbols = combined_atoms

  f = open(path+test_files[0],"w")
  f.write(f"{len(all_symbols)}\n")
  f.write("\n")
  
  for i in range(len(combined_poses)):
      row_string = f"{all_symbols[i]}"
      for coord in combined_poses[i]:
          row_string += f"    {coord}"
      if i != len(combined_poses): 
          f.write(row_string+"\n")
      else:
          f.write(row_string)
  f.close()

  f = open(path+test_files[0],"r")
  lines = f.readlines()
  mol_data = "".join(lines)

  viewer = py3Dmol.view(width=800, height=400)
  viewer.addModel(mol_data, "xyz")  # Add the trajectory frame
  viewer.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
  
  viewer.zoomTo()
  viewer.show()

  
  return combined_poses, combined_atoms