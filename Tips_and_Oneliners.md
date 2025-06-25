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

## Getting all of a particular file-type in a directory and making a list
```
import os

path = "CSV_files/"
files = os.listdir(path)            
filenames = [file for file in files if (os.path.splitext(file)[1]==".csv")]

print(filenames)
```
