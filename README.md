# CafChem
Libraries/modules for the CafChem tools

also see the list of [commonly used code snippets](https://github.com/MauricioCafiero/CafChem/blob/main/Tips_and_Oneliners.md).

## CafChemSubs
- generate analogues of a molecule (from SMILES strings) using generative mask-filling and/or substitutions on phenyl rings.
- Can also calculate some properties (QED, etc) related to drug design.
- visualize molecules. 
## CafChemReDock
- dock molecular SMILES strings in a protein using DockString and save poses.
- Calculate the interaction between a docking pose and a trimmed protein active site using Meta's [UMA MLIP](https://github.com/facebookresearch/fairchem).
- visualize molecules.
## CafChemBML
- read ChEMBL CSV files and clean data.
- featurize data, remove outliers, scale, apply PCA and split into training ad validation sets.
- perform analysis with tree-based methods, linear methods, SVR, and MLP.
## CafChemBoltz
- Input a protein sequence and a list of SMILES strings.
- Co-fold the protein/ligand pairs using [Boltz2](https://github.com/jwohlwend/boltz), extract the structures and predict IC50.
## Example notebooks
- the notebooks folder contains Colab notebooks to demonstrate each CafChem library
