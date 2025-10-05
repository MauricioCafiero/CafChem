# Some basics of Python
See below for examples of basic concepts in Python.

- [Various types of If statements](#if-statements) <br>
- [Basic For Loops](#basic-for-loops) <br>
- [For Loops for Lists/Iterables](#for-loops-for-lists) <br>
- [For Loops for Dictionaries](#for-loops-for-dictionaries) <br>
- [Read in tabular features from a CSV to a Numpy array](#read-in-tabular-features-from-a-csv) <br>
- [Create an RDKit, Scikit-learn and Torch Environment](#set-up-rdkit-environment) <br>

## If statements

### Simple comparisons
```
x = 29.5

if x == 30.0:
  print("x is equal to 30")
else:
  print("x is not equal 30")
```
or with more options:
```
x = 29.5

if x == 30.0:
  print("x is equal to 30")
elif x > 30:
  print("x is greater than 30")
elif x < 30:
  print("x is less than 30")
```
### Using in and not in
Check if a value is in a list.
```
x_list = [30, 67, 90]
x = 29.5

if x in x_list:
  print("x is in the list")
else:
  print("x is not in the list")
```
Check if a string is in another string. (see [below](#for-loops-for-lists) for loops)
```
mol_list = ["chloromethane", "1-fluoroethane", "1,1-difluoroethane"]
frag = "fluoro"

for mol in  mol_list:
  if frag in mol:
    print(f"The fragement {frag} is in the molecule")
  else:
    print(f"The fragement {frag} is not in the molecule")
```
Check if a string is *not* in another string. (see [below](#for-loops-for-lists) for loops)
```
mol_list = ["chloromethane", "1-fluoroethane", "1,1-difluoroethane"]
frag = "chloro"

for mol in  mol_list:
  if frag not in mol:
    print(f"The fragement {frag} is not in the molecule")
  else:
    print(f"The fragement {frag} is in the molecule")
```

## Basic For Loops 

### Basic loop over a range
Goes from 0 to one less than the arguement.
```
for i in range(10):
  value = i*i
  print(f"The square of {i} is {value}")
```

## For Loops for Lists
### Loop over an iterable
This can be a loop over a list, or a Dataframe column.
```
value_list = [0, 1, 2, 3, 4]

for value in value_list:
  val_squared = value*value
  print(f"The square of {value} is {val_squared}")
```
### Loop over an iterable with enumeration
Adds a counting variable to the loop.
```
value_list = [0, 1, 2, 3, 4]

for i, value in enumerate(value_list):
  val_squared = value*value
  print(f"The square of value {i+1} is {val_squared}")
```
### Loop over two iterables with zip
Loops over two iterables of the same length.
```
value_list = [0, 1, 2, 3, 4]
val_squared_list = [0, 1, 4, 9, 16]

for value, square in zip(value_list, val_squared_list):
  print(f"The square of {value} is {square}") 
```
### List comprehension
For loop contained in a list definition:
```
value_list = [0, 1, 2, 3, 4]

val_squared_list = [value*value for value in value_list]
```
or with conditions:
```
value_list = [0, 1, 2, 3, 4]

val_squared_list = [value*value for value in value_list if value % 2 == 0]
```

## For Loops for Dictionaries
### Loop over a dictionary by keys and values
Loop over keys and cooresponding values simultaneously.
```
mol_dict = {"methane": "CH4", "ethane": "C2H6", "ammonia": "NH3"}

for key, value in mol_dict.items():
  print(f"The formula for {key} is {value}")
```
### Loop over a dictionary by keys only
Loop over keys in a dictionary.
```
mol_dict = {"methane": "CH4", "ethane": "C2H6", "ammonia": "NH3"}

print("The molecules available in the list are:")
for mol in mol_dict.keys():
  print(f"{mol}")
```
### Loop over a dictionary by values only
Loop over values in a dictionary.
```
mol_dict = {"methane": "CH4", "ethane": "C2H6", "ammonia": "NH3"}

print("The formulae available in the list are:")
for mol in mol_dict.values():
  print(f"{mol}")
```

## Read in tabular features from a CSV
Say you have a CSV file called features.csv which contains a column called "Targets" (column 11) and several other features in columns 0 through 10:
```
df = pd.read_csv("features.csv")

y = df['Targets'].to_list()
X = df.iloc[:,0:11].values
```
X will be a numpy array with all rows and the specified columns.


## Set-up RDKit environment
Open the Anaconda powershell and go to your desired folder
```
cd my_rdkit_folder
```
Now create the new RDkit environment:
```
conda create -c conda-forge -n rdkit-env rdkit
```
This will take a while, and you will have to indicate a yes at some point
```
Proceed? ([y]/n)?
```
Once the process is complete, activate the environment with:
```
conda activate rdkit-env
```
Once the environment is activated, install SciKit-learn, Torch and Jupyter (do one at a time on the command line):
```
pip install scikit-learn
pip install torch, torchvision
pip install jupyter
```
Now you can start Jupyter Notebook, which will open in your default browser:
```
jupyter notebook
```
When you are done with your session, you can close Jupyter (be sure to use the "close and halt" options from the menu). You can then deactivate the environment:
```
conda deactivate rdkit-env
```
