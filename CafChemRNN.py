import tensorflow as tf
import numpy as np
import pandas as pd
import deepchem as dc
import time
import transformers
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.model_selection import train_test_split
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

def test_vocab(filename: str, smiles_column = 'SMILES'):
  '''
    Tests the vocabulary of a new dataset against the foundation model vocabulary.
    Rejects if the new dataset has tokens not in the foundation model vocabulary, or if
    the context window is too large.

      Args:
        filename: name of new dataset
        smiles_column: name of the smiles column
      Returns:
        novel_items: list of tokens not in the foundation model vocabulary
  '''
  df = pd.read_csv(filename)

  Xa = []
  for smiles in df[smiles_column]:
    smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
    smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
    smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
    Xa.append(smiles)

  #===========================================================================================
  #featurize

  tokenizer=dc.feat.SmilesTokenizer(vocab_file="CafChem/data/vocab.txt")
  featname="SMILES Tokenizer"

  fl = list(map(lambda x: tokenizer.encode(x),Xa))

  biggest = 1
  smallest = 200
  for i in range(len(fl)):
      temp = len(fl[i])
      if temp > biggest:
          biggest = temp
      if temp < smallest:
          smallest = temp

  print(biggest, smallest)

  string_length = smallest - 1
  max_length = biggest

  fl2 = list(map(lambda x: tokenizer.add_padding_tokens(x,max_length),fl))

  fl2set=set()
  for sublist in fl2:
    fl2set.update(sublist)
  new_vocab_size = len(fl2set)
  print("New vocabulary size: ",new_vocab_size)

  f = open("CafChem/data/vocab_305K.txt", "r")
  raw_lines = f.readlines()
  f.close()
  VOCAB_SIZE = len(raw_lines)
  print("Vocabulary size for standard dataset: ",VOCAB_SIZE)

  lines = []
  for line in raw_lines:
    lines.append(line.replace("\n",""))

  novel_items = []
  for item in fl2set:
    item = tokenizer.decode([item])
    item = tokenizer.convert_tokens_to_string(item)
    item = item.replace(" ","")

    if item not in lines:
      print(f"{item} not in standard vocabulary")
      novel_items.append(item)

  if(len(novel_items) > 0):
    print("This dataset is not compatible with the Foundation model vocabulary")
  else:
    print("This dataset is compatible with the Foundation model vocabulary")

  if max_length > 166:
    print("This dataset's context window is not compatible with the Foundation model.")
  else:
    print("This dataset's context window is compatible with the Foundation model")

  return novel_items

def make_datasets(filename: str, smiles_column = 'SMILES'):
  '''
    Tokenizes a dataset and returns the input and target arrays.

      Args:
        filename: name of new dataset
        smiles_column: name of the smiles column
      Returns:
        fx: input array
        fy: target array
        VOCAB_SIZE: vocabulary size
        tokenizer: tokenizer object
        max_length: longest SMILES chain
  '''
  df = pd.read_csv(filename)

  Xa = []
  for smiles in df[smiles_column]:
    smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
    smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
    smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
    Xa.append(smiles)

  #===========================================================================================
  #featurize

  tokenizer=dc.feat.SmilesTokenizer(vocab_file="CafChem/data/vocab_305K.txt")
  featname="SMILES Tokenizer"

  fl = list(map(lambda x: tokenizer.encode(x),Xa))

  biggest = 1
  smallest = 200
  for i in range(len(fl)):
      temp = len(fl[i])
      if temp > biggest:
          biggest = temp
      if temp < smallest:
          smallest = temp

  print(biggest, smallest)

  string_length = smallest - 1
  max_length = biggest

  fl2 = list(map(lambda x: tokenizer.add_padding_tokens(x,max_length),fl))

  # fl2set=set()
  # for sublist in fl2:
  #   fl2set.update(sublist)
  # temp_vocab_size = len(fl2set)

  f = open("CafChem/data/vocab_305K.txt", "r")
  lines = f.readlines()
  f.close()
  VOCAB_SIZE = len(lines)
  print("Vocabulary size for this dataset: ",VOCAB_SIZE)

  x = []
  y = []
  i=0
  for string in fl2:
      x.append(string[0:max_length-1]) #string_length
      y.append(string[1:max_length]) #string_length+1

  x = np.array(x)
  y = np.array(y)
  print("Number of features and datapoints, targets: ",x.shape,y.shape)

  #===========================================================================================
  print("featurization done with: ",featname)

  fx = x
  fy = y

  return fx, fy, VOCAB_SIZE, tokenizer, max_length

def strip_smiles(input_string):
  '''
    Cleans un-needed tokens from the SMILES string.

      Args:
        input_string: SMILES string
      Returns:
        output_string: cleaned SMILES string
  '''
  output_string = input_string.replace(" ","").replace("[CLS]","").replace("[SEP]","").replace("[PAD]","")
  output_string = output_string.replace("[Na+].","").replace(".[Na+]","")
  return output_string

def mols_from_smiles(input_smiles_list):
  '''
    Converts a list of SMILES strings to a list of RDKit molecules.

      Args:
        input_smiles_list: list of SMILES strings
      Returns:
        valid_mols: list of RDKit molecules
        valid_smiles: list of SMILES strings
  '''
  valid_mols = []
  valid_smiles = []

  good_count = 0
  for ti, smile in enumerate(input_smiles_list):
    temp_mol = Chem.MolFromSmiles(smile)
    if temp_mol != None:
      valid_mols.append(temp_mol)
      valid_smiles.append(smile)
      good_count += 1
    else:
      print(f"SMILES {ti} was not valid!")

  if len(valid_mols) == len(valid_smiles) == good_count:
    print(f"Generated a total of {good_count} mol objects")
  else:
    print("mismatch!")
  return valid_mols, valid_smiles

def test_gen(model, tokenizer, T_int: float, VOCAB_SIZE: int, rn_seed = 42):
  '''
    use a RNN model to generate novel molecules.

      Args:
        model: the RNN model to use
        tokenizer: tokenizer to use
        T_int: temperature for inference
        VOCAB_SIZE: vocabulary size
        rn_seed: random seed
      Returns:
        img: image of generated molecules
  '''
  tf.random.set_seed(rn_seed)

  test_string = ['C(', 'O=', 'c1', 'NC', 'CO']
  batch_length = len(test_string)
  test_xlist = np.empty([batch_length,3], dtype=int)

  test_tokenized = list(map(lambda x: tokenizer.encode(x),test_string))
  for i in range(batch_length):
      test_xlist[i][:] = test_tokenized[i][:3]
  test_array = np.array(test_xlist)

  proba = np.empty([batch_length,VOCAB_SIZE])
  rescaled_logits = np.empty([batch_length,VOCAB_SIZE])
  preds = np.empty([batch_length])
  gen_molecules = np.empty([batch_length])

  for i in range(1,80,1):
      results = model.predict(test_array)

      if T_int < 0.015:
          print(f"using zero temp generation with {T_int}.")
          for j in range(batch_length):
              preds[j] = tf.argmax(results[j][-1])
              preds = list(map(lambda x: int(x),preds))
      else:
          print(f"using variable temp generation with {T_int}.")
          for j in range(batch_length):
              proba[j] = (results[j][-1:]) ** (1/T_int)
              rescaled_logits[j] = ( proba[j][:] ) / np.sum(proba[j][:])
              preds[j] = np.random.choice(len(rescaled_logits[j][:]),
                                          p=rescaled_logits[j][:])
              preds = list(map(lambda x: int(x),preds))
      test_array = np.c_[test_array,preds]
      print(test_array.shape)

  gen_molecules = list(map(lambda x: tokenizer.decode(x),test_array))
  gen_molecules = list(map(lambda x: tokenizer.convert_tokens_to_string(x),
                            gen_molecules))
  gen_molecules = list(map(lambda x: strip_smiles(x),gen_molecules))

  mols, smiles = mols_from_smiles(gen_molecules)

  img = Draw.MolsToGridImage(mols,molsPerRow=3,legends=smiles)
  return img

def make_rnn(num_layers: int, layer_size: int, max_length: int, vocab_size: int):
  '''
    creates a RNN with a specified number of transformer blocks.

      Args:
        num_layers: number of GRU layers
        layer_size: number of neurons per layer
        max_length: context window
        VOCAB_SIZE: vocabulary size
      Returns:
        rnn: RNN model
  '''
  layers_list = [
    tf.keras.Input(shape=(max_length,)),
    tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=24),
    tf.keras.layers.GRU(layer_size,return_sequences=True),
    tf.keras.layers.Dense(vocab_size,activation="softmax")]
  for _ in range(num_layers-1):
    layers_list.insert(-1,tf.keras.layers.GRU(layer_size,return_sequences=True))
  
  rnn = tf.keras.Sequential(layers_list)
  
  rnn.summary()

  return rnn

def save_rnn(rnn, filename: str):
  '''
    saves a RNN model.

      Args:
        rnn: RNN model
        filename: name of the model
      Returns:
        None; saves model and a list of layer names to files.
  '''
  layer_name_store = []
  for layer in rnn.layers:
    layer.name = layer.name+"_original"
    layer_name_store.append(layer.name)

  print("New layer names:")
  print("===========================================")
  rnn.summary()

  rnn.save_weights(f"{filename}.weights.h5")
  print(f"model saved with name: {filename}.")

  f = open(f"layer_store_{filename}.txt", "w")
  for item in layer_name_store:
      f.write("%s\n" % item)
  f.close()
  print(f"layer names saved in file: layer_store_{filename}.")

def make_finetune_rnn(num_new_layers: int, layer_size: int = 128, freeze_old_layers = True):
  '''
    Creates a finetuning model from a set foundation model. 

      Args:
        num_new_layers: number of new GRU layers to add
        layer_size: how many neurons per layer
        freeze_old_layers: whether to freeze the old layers
      Returns:
        rnn_ft: finetuning model
  '''
  VOCAB_SIZE = 100
  max_length = 166
  rnn_ft = make_rnn(2+num_new_layers, layer_size, max_length, VOCAB_SIZE)

  f = open("CafChem/data/layer_store_RNN_ZN305_50epochs.txt", "r")
  layer_name_store_raw = f.readlines()
  f.close()

  print("Reading in layers:")
  layer_name_store = []
  for line in layer_name_store_raw:
      line = line.replace("\n","")
      layer_name_store.append(line)
      print(line)
  print("===========================================")

  new_layers = num_new_layers + 1
  for i,layer in enumerate(rnn_ft.layers[:-new_layers]):
    layer.name = layer_name_store[i]
    print(f"{layer.name} has been named!")

  for i,layer in enumerate(rnn_ft.layers[-new_layers:-1]):
    layer.name = f"GRU_layer_X_{i+1}"
    print(f"{layer.name} has been named!")

  rnn_ft.layers[-1].name = "dense_X"

  rnn_ft.load_weights("CafChem/data/RNN_ZN305_50epochs.weights.h5", skip_mismatch=True)

  if freeze_old_layers == True:
    for layer in rnn_ft.layers[0:-new_layers]:                 #make old layers freeze and only train new layers
      layer.trainable=False
      print(f"setting layer {layer.name} untrainable.")

    for layer in rnn_ft.layers[-new_layers:]:
      layer.trainable=True
      print(f"setting layer {layer.name} trainable.")

  rnn_ft.summary()
  return rnn_ft

def unfreeze_rnn(rnn_model):
  '''
    unfreezes all layers in a model.

      Args:
        rnn_model: model to unfreeze
      Returns:
        rnn_model: unfrozen model
  '''
  for layer in rnn_model.layers:
    layer.trainable=True
    print(f"setting layer {layer.name} trainable.")

  rnn_model.summary()

  return rnn_model

def load_rnn(filename: str, total_layers: int, layer_size: int, max_length: int, VOCAB_SIZE: int):
  '''
    loads a RNN model.

      Args:
        filename: name of the model (h5 file)
        total_layers: total number of transformer blocks
        max_length: context window
        VOCAB_SIZE: vocabulary size
      Returns:
        rnn_load: loaded RNN model
  '''

  rnn_load = make_rnn(total_layers, layer_size, max_length, VOCAB_SIZE)

  f = open(f"layer_store_{filename}.txt", "r")
  layer_name_store_raw = f.readlines()
  f.close()

  layer_name_store = []
  for line in layer_name_store_raw:
      line = line.replace("\n","")
      layer_name_store.append(line)
      print(line)

  for i,layer in enumerate(rnn_load.layers):
    layer.name = layer_name_store[i]
    print(f"{layer.name} has been named!")

  rnn_load.load_weights(f"{filename}.weights.h5", skip_mismatch=True)
  print(f"model loaded with name: {filename}.")
  rnn_load.summary()

  return rnn_load

def load_foundation():
  '''
    loads the RNN Foundation model.

      Args:
        None
      Returns:
        rnn_load: loaded RNN model
  '''
  VOCAB_SIZE = 100
  max_length = 166
  layer_size = 256
  rnn_load = make_rnn(2, layer_size, max_length, VOCAB_SIZE)

  f = open("CafChem/data/layer_store_RNN_ZN305_50epochs.txt", "r")
  layer_name_store_raw = f.readlines()
  f.close()

  layer_name_store = []
  for line in layer_name_store_raw:
      line = line.replace("\n","")
      layer_name_store.append(line)
      print(line)

  for i,layer in enumerate(rnn_load.layers):
    layer.name = layer_name_store[i]
    print(f"{layer.name} has been named!")

  rnn_load.load_weights("CafChem/data/RNN_ZN305_50epochs.weights.h5", skip_mismatch=True)
  print("Foundation model loaded.")
  rnn_load.summary()

  return rnn_load

def gen_mols(prompts: list, use_ramp: bool, model, tokenizer, TEMP: float, VOCAB_SIZE: int, rn_seed = 42):
  '''
    use an RNN model to generate novel molecules.

      Args:
        prompts: a list of prompts for inference
        use_ramp: Boolean to use temperature ramp during inference
        model: the GPT model to use
        tokenizer: tokenizer to use
        TEMP: temperature for inference
        VOCAB_SIZE: vocabulary size
        rn_seed: random seed
      Returns:
        img: image of generated molecules
  '''
  tf.random.set_seed(rn_seed)

  test_string = prompts
  batch_length = len(test_string)
  prompt_length = len(test_string[0])
  test_xlist = np.empty([batch_length,prompt_length], dtype=int)

  test_tokenized = list(map(lambda x: tokenizer.encode(x),test_string))
  for i in range(batch_length):
      test_xlist[i][:] = test_tokenized[i][:prompt_length]
  test_array = np.array(test_xlist)

  proba = np.empty([batch_length,VOCAB_SIZE])
  rescaled_logits = np.empty([batch_length,VOCAB_SIZE])
  preds = np.empty([batch_length])
  gen_molecules = np.empty([batch_length])

  c_final = 90 - prompt_length
  sig_start = 0.10
  
  for c in range(0,c_final,1):
      
      c_o = int(c_final*sig_start)
      if use_ramp == True:
          T_int = TEMP*(1/(1+np.exp(-(c-c_o))))
      else:
          T_int = TEMP
      
      results = model.predict(test_array)

      if T_int < 0.015:
          print(f"using zero temp generation with {T_int}.")
          for j in range(batch_length):
              preds[j] = tf.argmax(results[j][-1])
              preds = list(map(lambda x: int(x),preds))
      else:
          print(f"using variable temp generation with {T_int}.")
          for j in range(batch_length):
              proba[j] = (results[j][-1:]) ** (1/T_int)
              rescaled_logits[j] = ( proba[j][:] ) / np.sum(proba[j][:])
              preds[j] = np.random.choice(len(rescaled_logits[j][:]),
                                          p=rescaled_logits[j][:])
              preds = list(map(lambda x: int(x),preds))
      test_array = np.c_[test_array,preds]
      print(test_array.shape)

  gen_molecules = list(map(lambda x: tokenizer.decode(x),test_array))
  gen_molecules = list(map(lambda x: tokenizer.convert_tokens_to_string(x),
                            gen_molecules))
  gen_molecules = list(map(lambda x: strip_smiles(x),gen_molecules))

  mols, smiles = mols_from_smiles(gen_molecules)
  
  final_smiles = []
  final_mols = []
  for smile, mol in zip(smiles,mols):
      if smile not in final_smiles:
          final_smiles.append(smile)
          final_mols.append(mol)
  
  print(f"Generated {len(final_smiles)} unique molecules.")

  img = Draw.MolsToGridImage(final_mols,molsPerRow=3,legends=final_smiles)
  return img, final_smiles

def make_prompts(num_prompts: int, prompt_length: int):
  '''
    Makes a set of prompts for inference

      Args:
        num_prompts: now many prompts to make_datasets
        prompt_length: how many tokens in the prompt
      Returns:
        prompts: a list of prompts
  '''
  df = pd.read_csv("CafChem/data/ZN305K_smiles.csv")

  Xa = []
  for smiles in df["SMILES"]:
    smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
    smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
    smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
    Xa.append(smiles)


  raw_prompts = random.choices(Xa,k=num_prompts)
  
  prompts = []
  for smile in raw_prompts:
    prompts.append(smile[:prompt_length])

  return prompts