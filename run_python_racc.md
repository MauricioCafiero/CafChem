# Running Python packages on RACC2

- [Psi4 for quantum chemistry](#running-psi4-on-racc2) <br>
- [UMA MLIP for quantum chemistry](#running-fairchem-uma-mlip-on-racc2) <br>

## Running FairChem UMA MLIP on RACC2
### Load anaconda and create an environment
First ensure you have enough disk space. You can check with the du command and your user-id:
```
[user-id@racc2]$ du -h /home/users/user-id
```
at the end of the scrolling output it will list your used memory. You have 10GB by default. <br>
Now load anaconda:
```
[user-id@racc2]$ module load anaconda
```
You should now get (base) at the begining of your command line:
```
(base)[user-id@racc2]$
```
Now you can create the new environment (called uma-env):
```
conda create -n uma-env python=3.12
```
It will take a few minutes, and at one point you will have to answer yes:
```
Proceed ([y]/n)?
```
Once you have create the environment, enter is with the following command:
```
(base)[user-id@racc2]$ source activate uma-env
```
your command line will the change to reflect the environment:
```
(uma-env)[user-id@racc2]$
```

### Install packages
You will need to install Fairchem and RDKit
```
(uma-env)[user-id@racc2]$ pip install fairchem-core
```
this make take a while! After it is done:
```
(uma-env)[user-id@racc2]$ pip install rdkit
```
Again, this may take a while!

### Set-up your Huggingface token
You can save your huggingface token to your account by running the following:
```
(uma-env)[user-id@racc2]$ hf auth login
```
### Set up scripts
- Find the gpu.slurm file in this repo and place it in your home directory, in a folder called scripts/
- Find the submit_gpu.sh file in this repo and place it in your home directory, in a folder called bin/
(make sure ~/bin is in your path in your .bash_profile)

### Run UMA on a GPU on RACC2
Prepare your python file (test-uma.py) and run it on a GPU:
```
(uma-env)[user-id@racc2]$ submit_gpu.sh test-uma.py
```
Check if it is running using
```
(uma-env)[user-id@racc2]$ squeue
```
If this it is your first time running a particular UMA checkpoint, it will download it first. It will not have to doanload it again. 

## Running Psi4 on RACC2
### Load anaconda and create an environment
First ensure you have enough disk space. You can check with the du command and your user-id:
```
[user-id@racc2]$ du -h /home/users/user-id
```
at the end of the scrolling output it will list your used memory. You have 10GB by default. <br>
Now load anaconda:
```
[user-id@racc2]$ module load anaconda
```
You should now get (base) at the begining of your command line:
```
(base)[user-id@racc2]$
```
Now you can create the new environment (called uma-env):
```
conda create -n p4env psi4 -c conda-forge/label/libint_dev -c conda-forge
```
It will take a few minutes, and at one point you may have to answer yes:
```
Proceed ([y]/n)?
```
Once you have create the environment, enter is with the following command:
```
(base)[user-id@racc2]$ source activate p4env
```
your command line will the change to reflect the environment:
```
(p4env)[user-id@racc2]$
```

### Install packages
You will need to install RDKit
```
(p4env)[user-id@racc2]$ pip install rdkit
```
This may take a few minutes!

### Set up scripts
- Find the template.slurm file in this repo and place it in your home directory, in a folder called scripts/
- Find the submit_psi4.sh file in this repo and place it in your home directory, in a folder called bin/
(make sure ~/bin is in your path in your .bash_profile)
- for this script, your python file which you plan to submit to the queue should have the following lines:
```
cores = 1
p4_mem = 1
```
The script will then substitute the 1's for the amount of memory and cores you specify.

### Run UMA on a GPU on RACC2
Prepare your python file (test_p4.py) and run it:
```
(p4env)[user-id@racc2]$ submit_psi4.sh test-p4.py
```
Check if it is running using
```
(p4env)[user-id@racc2]$ squeue
```
