# CafChem - Libraries for computational chemistry/drug design research.

See below for sample notebooks for various computation and medicinal chemistry, machine learning and AI research tools.

## Some basics and background material

- See the [list of commonly used Python techniques](https://github.com/MauricioCafiero/CafChem/blob/main/Tips_and_Oneliners.md).
- See the [sample notebook for PubChemPy usage](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Pubchem_CafChem.ipynb).
- See the [guide to using various python packages on RACC2](https://github.com/MauricioCafiero/CafChem/blob/main/run_python_racc.md).
- See the [primer on regular expressions](https://github.com/MauricioCafiero/CafChem/blob/main/regex.md).

## Generative models for molecules

- [Train a GPT and generate novel molecules](#cafchemgpt) <br>
- [Train an RNN and generate novel molecules](#cafchemrnn) <br>
- [Generative tools for hit expansion](#cafchemsubs) <br>
- [Grow Fragments in a Binding Site](#cafchemfraggrow)<br>

## Machine Learning for chemistry

- [Basic machine learning and cleaning ChEMBL CSV files](#cafchembml) <br>
- [Regression and Classification with dense neural networks using TensorFlow](#cafchemskipdense) <br>
- [MLP for Regression and Classification with PyTorch](#cafchemmlppytorch) <br>
- [SciKitLearn Classifiers](#cafchemclassifiers) <br>
- [HuggingFace classifier models](#cafchemhfclassifier)<br>
- [ChemProp GNN MPNN training and inference](#cafchemprop) <br>
- [Chemeleon GNN foundation model finetuning](#cafchemeleon) <br>
- [Build a dataset with active learning](#cafchemaldatabuild) <br>

## Protein / Ligand interactions

- [Docking with Autodock Vina, rescoring docking poses with Meta's UMA MLIP<sup>1</sup>](#cafchemredock) <br>
- [Boltz2 for co-folding proteins and ligands](#cafchemboltz) <br>

## Medicinal Chemistry

- [Pharmacophore feature testing](#cafchempharm) <br>

## Quantum Chemistry 

- [QM calculations with the UMA MLIP<sup>1</sup>](#cafchemqm_uma) <br>
- [DFT and SAPT calculations using Psi4](#cafchempsi4) <br>

## LLMs for Medchem

- [AI Agent with Medicinal Chemistry Tools](#cafchemagent)
- [Embedding Models for Molecules](#cafchemembed)
- [Inference and finetuning with TxGemma](#cafchemtxgemma) <br>
- [Inference with Ether0](#cafchemether0) <br>

<sup>1</sup>Solvation (adding explicit waters and optimizing) available in ReDock and QM_UMA <br>

## Install:
```
git clone https://github.com/MauricioCafiero/CafChem.git

import CafChem.CafChemGPT as ccgpt
import CafChem.CafChemRNN as ccrnn
import CafChem.CafChemTxGemma as cctxg
import CafChemSkipDense as ccsd
import CafChem.CafChemHFClassifier as cchf
import CafChem.CafChemBoltz as ccb
import CafChem.CafChemQM_UMA as ccqm
import CafChem.CafChemEleon as ccel
import CafChem.CafChemProp as ccp
import CafChem.CafChemSubs as ccs
import CafChem.CafChemReDock as ccr
import CafChem.CafChemBML as ccml
import CafChem.CafChemFragGrow as ccfg
import CafChem.CafChemMLPPyTorch as ccmlp
import CafChem.CafChemPsi4 as ccp4
```
## CafChemGPT
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/GPT_CafChem.ipynb)
- Train a GPT on a SMILES dataset. Use the tools provided to generate novel molecules.
- Using a provided foundation model, finetune with a specific dataset for targeted molecule generation.
- This also uses the CafChemGPTINF module for inference. 

## CafChemRNN
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/RNN_CafChem.ipynb)
- Train an RNN on a SMILES dataset. Use the tools provided to generate novel molecules.
- Using a provided foundation model, finetune with a specific dataset for targeted molecule generation. 

## CafChemSubs
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Hit_Expansion_CafChem.ipynb)
- generate analogues of a molecule (from SMILES strings) using generative mask-filling and/or substitutions on phenyl rings.
- Can also calculate some properties (QED, Lipinski properties) related to drug design.
- Calculate Tanimoto similarities based on Fingerprints between molecules in a list and molecules against a known active.
- visualize molecules.

## CafChemFragGrow
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/FragGrow_CafChem.ipynb)
- Explore a binding site with chemical fragments.
- Various viewing options to probe the nature of the binding site.

## CafChemBML
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/BasicML_CafChem.ipynb)
- read ChEMBL CSV files and clean data.
- featurize data, remove outliers, scale, apply PCA and split into training ad validation sets.
- perform analysis with tree-based methods, linear methods, SVR, and MLP.

## CafChemSkipDense
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Skipdense_CafChem.ipynb)
- Create regression and classification models using skipdense neural networks.
- Train, save, load and evaluate models.

## CafChemMLPPyTorch
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/MLP_with_PyTorch_CafChem.ipynb)
- Featurize a dataset and
- Train an MLP using Pytorch.
- Evaluate, predict with, save and load models.

## CafChemClassifiers
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Classifiers_CafChem.ipynb)
- Create a classifier model using a variety of SciKitLearn models.
- Load a CSV with quantitative data and create classes.
- Tree-based models, Logistic Regression, Support Vector Machines, Ridge, MLP.
- Analyze data with confusion matrices.

## CafChemHFClassifier
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/BertClassifier_CafChem.ipynb)
- Create a classifier model using HuggingFace.
- Analyze data with confusion matrices.
- Load datasets, add tokens, train, push all to the HuggingFace hub.

## CafChemProp
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Chemprop_CafChem.ipynb)
- Train the [Chemprop](https://github.com/chemprop) GNN-based MPNN model.
- save and load trained models and analyze data.
## CafChemEleon
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Chemeleon_CafChem.ipynb)
- finetune the [Chemeleon](https://github.com/JacksonBurns/chemeleon) foundation model.
- save and load trained models and analyze data.

## CafChemALDataBuild
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/AL_DataBuild_CafChem.ipynb)
- Use active learning and a gaussian process regressor to build up a dataset to a desired accuracy.
- export the dataset at the end.

## CafChemReDock
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Rescore_Docking_UMA_CafChem.ipynb)
- dock molecular SMILES strings in a protein using DockString and save poses.
- Calculate the interaction between a docking pose and a trimmed protein active site using Meta's [UMA MLIP](https://github.com/facebookresearch/fairchem).
- visualize molecules.

## CafChemBoltz
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Boltz_CafChem.ipynb)
- Input a protein sequence and a list of SMILES strings.
- Co-fold the protein/ligand pairs using [Boltz2](https://github.com/jwohlwend/boltz), extract the structures and predict IC50.

## CafChemQM_UMA
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/QuantumChem_UMA_CafChem.ipynb)
- Uses ASE to implement calculations using Meta's [UMA MLIP](https://github.com/facebookresearch/fairchem).
- perform energy calculations, geometry optimizations, vibrational calculations, and thermodynamics calculations.
- Calculate a reaction Gibbs, Enthalpy and Entropy.
- Perform simple dynamics. (Langevin works, Velocity Verlet seems a bit buggy)

## CafChemPharm
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Pharmacophores_CafChem.ipynb)
- Generate a defined number of conformers for a list of molecules.
- Test pharmacophore features of a single or multiple conformers against a known active. 

## CafChemPsi4
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Psi4_CafChem.ipynb)
- Use the Psi4 code to run DFT energy and geometry optimization calculations.
- Use SAPT on Psi4 to explore contributions to interaction energies. 

## CafChemAgent
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/ChemAgent_CafChem.ipynb)
- [using Apertus](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/ChemAgent_CafChem_Apertus.ipynb)
- A simple agent to test light-weight HuggingFace models on chemical tool use.

## CafChemEmbed
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Embeddings_CafChem.ipynb)
- Create a contrastive pairs dataset
- Train an embedding model
- Use embeddings for similarity calculations or features for regression

## CafChemTxGemma
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/TxGemma_CafChem.ipynb)
- Inference with TxGemma models.
- These models have been finetuned to answer many types of medicinal chemistry questions.
- Finetune a TxGemma model on your own medchem dataset

## CafChemEther0
- [example notebook](https://github.com/MauricioCafiero/CafChem/blob/main/notebooks/Ether0_CafChem.ipynb)
- Inference with the Ether0 model.
- This model has been finetuned to answer many types of medicinal chemistry questions. (see the notebook for use cases).
