# Tips for Performant Python (from High Performance Python by Gorelick and Ozsvald)

## Data Structure tips

- When appending to a list of length *n*, a new list is created of length *m > n* and the old list is copied over. Then the new length *m* is full, another new list is created to continue appending. This takes up extra memory and adds time overhead during the allocation and copying processes.
- While tuples cannot be appended to, they can be added together, and the creation of the new tuple takes time, but does not incur the extra memory that appending to a list does.
- Dictionaries and sets have O(1) lookup compared to ~O(n) for a list.
- In general use Numpy arrays as much as possible. These will run efficiently on CPUs.
- Converting data structures to Torch tensors will run efficiently on GPUs.
- A Marisa Trie can store string data more efficiently than other data structures:
```
!pip install marisa-trie
import marisa_trie

trie = marisa_trie.Trie(df["SMILES"].to_list())

#check for membership
'O=C1c2ccccc2-c2nnc(-c3ccc(Cl)cc3)cc21' in trie
>> True

#get id number
trie['O=C1c2ccccc2-c2nnc(-c3ccc(Cl)cc3)cc21']
>> 916

#query value
trie.restore_key(916)
>>'O=C1c2ccccc2-c2nnc(-c3ccc(Cl)cc3)cc21'
```

## Namespace
- When a variable is called, python looks through the local namespace, then global namespace, and then on modules.
- Consider the three examples below:
```
import math
from math import sin

# example 1:
def test(x):
  y = math.sin(x)

#example 2:
def test(x):
  y = sin(x)

#example 3:
def test(x, sin_fcn = math.sin):
  y = sin_fcn(x)
```
The first requires local --> global --> module. The second requires local --> global, and so cutting out a step makes the call faster. The third requires local --> gloabl --> module *only the first time* ans then only local the rest of the time, thus saving time.

## Generators / Iterators
- Instead of using a function to generate many values at once, yield can be used to produce one at a time, reducing memory usage.
```
def make_mol(smiles: list):
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        yield mol

mol = make_mol(smiles)

for m in mol:
  ns = Chem.MolToSmiles(m)
  print(ns)
```
This makes your list of mol objects into an iterable that only uses one unit of memory at a time.

## Memory
- Allocations take a lot of time, so to reduce allocations:
  * Define a variable at the begining and use in-place operations: +=, -=, /= *etc*.
- Test memory usage:
```
!pip install -U memory_profiler
import memory_profiler

print(f"Current RAM: {memory_profiler.memory_usage()[0]:.2f} MiB")

smiles = df["SMILES"].to_list()

print(f"Current RAM: {memory_profiler.memory_usage()[0]:.2f} MiB")
```
 
## Speed
- numexpr can make vector expressions much more efficient:
```
from numexpr import evaluate

v1 = np.array((1.0, 2.0, 3.0))
v2 = np.array((2.3, 3.4, 7.6))
product = np.zeros((3))

evaluate('v1+v2', out=product)
print(product)
```

## Parallel computation
- the multiprocessing module can be used to run on mutiple cores at once.
- Use the Pool method to create workers:
```
from multiprocessing import Pool
import time

def make_mol(smile):
    mols = Chem.MolFromSmiles(smile)
    return mols
        
num_blocks = 4
pool = Pool(processes=num_blocks)

num_per_core = int(len(smiles)/num_blocks)

tic =time.time()
total_mols = pool.map(make_mol,smiles)
total_time = time.time() - tic
print(f"mols made: {len(total_mols)}")
print(f"total time: {total_time:.6f}")

#check the mols
ns = [Chem.MolToSmiles(x) for x in total_mols]
print(ns[:4])
```
This gives:
```
mols made: 950
total time: 0.156927
['O=C1c2ccccc2-c2nnc(-c3ccc(Cl)cc3)cc21', 'CC(C)Nc1cnccc1CN.Cl.Cl', 'Cl.Cl.NCc1ccncc1NC1CC1.O', 'Cl.Cl.NCc1ccncc1NCC1CCCCC1.O']
```
Using 2 cores gives:
```
total time: 0.200889
```
Using 1 core gives:
```
total time: 0.340345
```

