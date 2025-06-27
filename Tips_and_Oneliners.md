# Topics Covered:
- [Using a text classifier from HuggingFace](#using-a-text-classifier-from-huggingface) <br>
- [Quick reading and writing to CSVs](#quick-reading-and-writing-to-csvs) <br>
- [Read and write pickle files](#read-and-write-pickle-files) <br>
- [Getting all of a particular file-type in a directory and making a list](#getting-all-of-a-particular-file-type-in-a-directory-and-making-a-list) <br>
- [Reload library](#reload-library) <br>
- [Pandas frequently used methods](#general-pandas-tips) <br>


## Using a text classifier from HuggingFace
```
from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline("text-classification", model="cafierom/bert-base-cased-DA-ChemTok-ZN1540K-V1-finetuned-HMGCR-IC50s-V1", device=device)

pipe("OC(=O)C[C@H](O)C[C@H](O)\C=C\c1c(C(C)C)nc(N(C)S(=O)(=O)C)nc1c2ccc(F)cc2")
```
then, to implement the pipe:
```
rosuvastatin = pipe("OC(=O)C[C@H](O)C[C@H](O)\C=C\c1c(C(C)C)nc(N(C)S(=O)(=O)C)nc1c2ccc(F)cc2")

rosuvastatin
```
This produces:
```
>> [{'label': '< 50 nM', 'score': 0.9481903910636902}]
```
For trained  classifiers to use, see, for example: [My HuggingFace page](https://huggingface.co/cafierom)

## Quick reading and writing to CSVs
```
import pandas as pd
import numpy as np

df = pd.read_csv("/content/maob_3865_ic50.csv")
df.head()
```
this produces:
![image](https://github.com/user-attachments/assets/9e5818a2-4428-4850-9c19-980faaa748ac)
Convert dataframe columns to lists:
```
ic50s = df["IC50"].to_list()
smiles = df["SMILES"].to_list()
```
Make a set of lists into a new dataframe, and save as CSV:
```
logIC50s = [np.log10(ic50) for ic50 in ic50s]
logic50_dict = {"logIC50": logIC50s, "SMILES": smiles}
logic50_df = pd.DataFrame(logic50_dict)
logic50_df.head()
```
this produces: <br>
![image](https://github.com/user-attachments/assets/41c15286-501e-49be-976a-b224c743471e) <br>
Finally, save a new CSV:
```
logic50_df.to_csv("logic50_df.csv")
```

## Read and write pickle files
The code below opens a file in binary mode for writing, ansd then 'dumps' the contents into it.
This may be used for many types of objects:
- a features array (as shown here)
- a Machine learning model (scikitlearn models, for example)
```
import pickle as pkl

with open(filename+".pkl", "wb") as f:
    pkl.dump(features, f)  
print(f"Features saved to {filename}.pkl")
```
to load a pickled object:
```
with open(filename+".pkl", "rb") as f:
    features = pkl.load(f)
print(f"Features saved to {filename}.pkl")
```
 
## Getting all of a particular file-type in a directory and making a list
```
import os

path = "CSV_files/"
files = os.listdir(path)            
filenames = [file for file in files if (os.path.splitext(file)[1]==".csv")]

print(filenames)
```

## Reload library
```
import importlib
importlib.reload(your_lib)
```
If this library was cloned from Github, be sure to remove the old directory first:
```
rm -r your_lib
```
and then re-clone the library before reloading.

## General Pandas tips
Assuming that pandas is imported!

Create a new dataframe (df_2) as a subset of another dataframe (df_1) which has the columns: "Smiles","IC50","Column 2","Column 3", "Column 4"...:
```
df_2 = df_1[["Smiles","IC50","Column 2"]]
```
Filter a dataframe (df_2) for specific values in a specific column and create a new dataframe (df_3):
```
df_3 = df_2[df_2["IC50"] < 100]
```
Drop a column from a dataframe:
```
df_3.drop(["Column 2"],axis=1,inplace=True)
```
Rename columns:
```
df_3 = df_3.rename(columns={"Smiles":"SMILES","IC50":"IC50 (nM)"})
```
Apply a function to every row in a column (assuming you have a function defined called *smiles_to_canon* and it takes the values in the givn row as arguments):
```
df_3["SMILES"] = df_3["SMILES"].apply(smiles_to_canon)
```
Drop duplicates from a dataframe based on a specific column: 
```
df_3 = df_3.drop_duplicates(subset=["SMILES"])
```
Sort a dataframe by a specific row:
```
df_3.sort_values(by=["IC50 (nM)"],inplace=True)
```

