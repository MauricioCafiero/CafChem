# CafChem
Libraries/modules for the CafChem tools for computational chemistry/drug design. Modules include:

- [QM calculations with the UMA MLIP](#cafchemqm_uma) <br>
- [Generative tools for hit expansion](#cafchemsubs) <br>
- [Docking with Autodock Vina, rescoring docking poses with Meta's UMA MLIP](#cafchemredock) <br>
- [Basic machine learning and cleaning ChEMBL CSV files](#cafchembml) <br>
- [Boltz2 for co-folding proteins and ligands](#cafchemboltz) <br>
- [Chemeleon GNN foundation model finetuning](#cafchemeleon) <br>
- [SciKitLearn Classifiers](#cafchemclassifiers) <br>
- [HuggingFace classifier models](#cafchemhfclassifier)<br>
- Solvation (adding explicit waters and optimizing) available in ReDock and QM_UMA <br>

The [notebooks folder](https://github.com/MauricioCafiero/CafChem/tree/main/notebooks) contains Colab notebooks to demonstrate each CafChem library

also see the list of [commonly used Python code snippets](https://github.com/MauricioCafiero/CafChem/blob/main/Tips_and_Oneliners.md).

## Install:
```
git clone https://github.com/MauricioCafiero/CafChem.git

import CafChem.CafChemHFClassifier as cchf
import CafChem.CafChemBoltz as ccb
import CafChem.CafChemQM_UMA as ccqm
import CafChem.CafChemEleon as ccel
import CafChem.CafChemSubs as ccs
import CafChem.CafChemReDock as ccr
import CafChem.CafChemBML as ccml
```

## CafChemQM_UMA
- Uses ASE to implement calculations using Meta's [UMA MLIP](https://github.com/facebookresearch/fairchem).
- perform energy calculations, geometry optimizations, vibrational calculations, and thermodynamics calculations.
- Calculate a reaction Gibbs, Enthalpy and Entropy.
- Perform simple dynamics. (Langevin works, Velocity Verlet seems a bit buggy)

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
## CafChemEleon
- finetune the [Chemeleon](https://github.com/JacksonBurns/chemeleon) foundation model.
- save and load trained models and analyze data.
## CafChemClassifiers
- Create a classifier model using a variety of SciKitLearn models.
- Load a CSV with quantitative data and create classes.
- Tree-based models, Logistic Regression, Support Vector Machines, Ridge, MLP.
- Analyze data with confusion matrices.
## CafChemHFClassifier
- Create a classifier model using HuggingFace.
- Analyze data with confusion matrices.
- Load datasets, add tokens, train, push all to the HuggingFace hub.
