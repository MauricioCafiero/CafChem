import json
import os
import py3Dmol

def make_boltz_files(smiles_list: list, sequence: str, prot_name: str, names = None):
  '''
  accepts a list of SMILES, a protein sequence (single chain for now), the name of 
  the protein, and an optional list of ligand names.
  returns .yaml files for each ligand with the protein, ready for analysis by Boltz.
  
    Args:
      smiles_list: list of SMILES strings
      sequence: protein sequence
      prot_name: name of protein
      names: optional list of ligand names
    Returns:
      names: list of ligand names
  '''

  if names == None:
    names = [f"{prot_name}_{i}" for i in range(len(smiles_list))]

  for smile,name in zip(smiles_list,names):

    f = open("boltz_template.yaml", "r")
    template = f.readlines()
    f.close()
    out_name = name + ".yaml"
    
    g = open(out_name,"w")
    for line in template:
      if 'smiles' in line:
        new_line = line.replace("\n","") + f" '{smile}'\n"
        g.write(new_line)
      elif 'sequence:' in line:
        new_line = line.replace("\n","") + f" {sequence}\n"
        g.write(new_line)
      else:
        g.write(line)
    g.close()

  return names

def cofold(names: list):
  '''
  accepts the names list returned by the previous tool and runs the Boltz analysis on each.
  returns a list of pIC50 values.

    Args:
      names: list of ligand names
    Returns:
      pIC50s: list of pIC50 values
  '''
  pIC50s = []
  for name in names:
    os.system(f'boltz predict "{name}.yaml" --use_msa_server')
    print(f"{name} done")
    path = f"boltz_results_{name}/predictions/{name}/affinity_{name}.json"

    # Open and read the JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    pIC50 = (6 - data['affinity_pred_value'])*1.364
    print(f"pIC50 is: {pIC50}")
    pIC50s.append(pIC50)

    IC50 = 10**(-pIC50)
    print(f"IC50 is: {IC50}")

    binder_decoy = data['affinity_probability_binary']
    print(f"Binder or Decoy: {binder_decoy}")  

  return pIC50s

def get_XYZ_files(name: str, protein: str):
  '''
  Creates three XYZ files: the complex, the protein, and the ligand.

    Args:
      name: name of ligand
      protein: name of protein
    Returns:
      None; creates three XYZ files
  '''
  path = f"boltz_results_{name}/predictions/{name}/{name}_model_0.cif"
  f = open(path,"r")
  lines = f.readlines()
  f.close()

  g = open(f"{name}_{protein}_complex.xyz","w")
  h = open(f"{name}_{protein}_protein.xyz","w")
  l = open(f"{name}_{protein}_ligand.xyz","w")

  atoms_ligand = 0
  atoms_protein = 0
  ligand_string = ""
  protein_string = ""

  for line in lines:
    parts = line.split()
    if "HETATM" in line:
      new_line = f"{parts[2]}    {parts[10]}    {parts[11]}    {parts[12]}\n"
      ligand_string += new_line
      atoms_ligand += 1
    elif "ATOM" in line:
      new_line = f"{parts[2]}    {parts[10]}    {parts[11]}    {parts[12]}\n"
      protein_string += new_line
      atoms_protein += 1

  g.write(f"{atoms_ligand + atoms_protein}\n\n")
  g.write(ligand_string)
  g.write(protein_string)

  h.write(f"{atoms_protein}\n\n")
  h.write(protein_string)

  l.write(f"{atoms_ligand}\n\n")
  l.write(ligand_string)
      
  g.close()
  h.close()
  l.close()
  
def visualize_molecule(xyz_string: str):
  '''
    input an XYZ string to vosualize the molecule in 3D
  '''
  viewer = py3Dmol.view(width=800, height=400)
  viewer.addModel(xyz_string, "xyz")  
  viewer.setStyle({"stick": {}, "sphere": {"radius": 0.5}})
  viewer.zoomTo()
  viewer.show()