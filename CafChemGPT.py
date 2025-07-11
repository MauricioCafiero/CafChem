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

def trim_vocab(filename: str, tokens_to_remove: list):
  '''
    trims entries from a SMILES list that contain tokens not found in the 
    Foundation model's vocabulary list. Also trims entries that are longer than the 
    Foundation model's context window. 
    
        Args:
            filename: a CSV file with the dataset to be trimmed
            tokens_to_remove: a set pf tokens to remove,  obtained from the test_vocab tool.
        Returns:
            None: a new CSV file is saved with the trimmed list.
  '''
  
  
df = pd.read_csv(filename)

  Xa = []
  for smiles in df[smiles_column]:
    smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
    smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
    smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
    Xa.append(smiles)
  
  smiles_removed_tokens = []
  for i,smiles in enumerate(Xa):
    bad_list = [True if (token in smiles) else False for token in tokens_to_remove]
    if not any(bad_list):
      smiles_removed_tokens.append(smiles)   

  smiles_no_long = []
  for i,smiles in enumerate(smiles_removed_tokens):
    if len(smiles) <= 166:
      smiles_no_long.append(smiles)
  
  print(f"Removed {len(Xa) - len(smiles_no_long)} entries from the list!")
  
  new_dict = {"SMILES": smiles_no_long}
  new_df = pd.DataFrame(new_dict)
  new_df.to_csv(f"{filename.replace(".csv","")+"_trimmed.csv"}, index=False)
  print("New CSV file written!")
  
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

  return fx, fy, VOCAB_SIZE, tokenizer, max_length #fl2set

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
    use a GPT model to generate novel molecules.

      Args:
        model: the GPT model to use
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
      results, _ = model.predict(test_array)

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

def casual_attention_mask(batch_size,n_dest,n_src,dtype):
  '''
    Make a causal attention mask
  '''
  i = tf.range(n_dest)[:,None]
  j = tf.range(n_src)
  m = i >= j - n_src + n_dest
  mask = tf.cast(m,dtype)
  mask = tf.reshape(mask,[1,n_dest,n_src])
  mult = tf.concat([tf.expand_dims(batch_size,-1),tf.constant([1,1],dtype=tf.int32)],0)
  return tf.tile(mask,mult)

class TransformerBlock(tf.keras.layers.Layer):
  '''
    Transformer block with multi-head attention.
  '''
  def __init__(self,num_heads,key_dim,embed_dim,ff_dim,dropout_rate=0.1):
    super(TransformerBlock,self).__init__()
    self.num_heads = num_heads
    self.key_dim = key_dim
    self.embed_dim = embed_dim
    self.ff_dim = ff_dim
    self.dropout_rate = dropout_rate
    self.attn = tf.keras.layers.MultiHeadAttention(self.num_heads,self.key_dim,
                                                    output_shape=self.embed_dim)
    self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)
    self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=0.000001)
    self.ffn_1 = tf.keras.layers.Dense(self.ff_dim,activation="relu")
    self.ffn_2 = tf.keras.layers.Dense(self.embed_dim)
    self.dropout_2 = tf.keras.layers.Dropout(self.dropout_rate)
    self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=0.000001)

  def call(self,inputs):
    input_shape = tf.shape(inputs)
    batch_size2 = input_shape[0]
    seq_len = input_shape[1]
    casual_mask = casual_attention_mask(batch_size2,seq_len,seq_len,tf.bool)
    attention_output, attention_scores = self.attn(inputs,inputs,
                                                    attention_mask=casual_mask,
                                                    return_attention_scores=True)
    attention_output = self.dropout_1(attention_output)
    out1 = self.ln_1(inputs + attention_output)
    ffn_1 = self.ffn_1(out1)
    ffn_2 = self.ffn_2(ffn_1)
    ffn_output = self.dropout_2(ffn_2)
    return (self.ln_2(out1+ffn_output),attention_scores)

  def get_config(self):
    config = super().get_config()
    config.update({"key_dim": self.key_dim, "embed_dim": self.embed_dim,
                  "num_heads": self.num_heads,"ff_dim": self.ff_dim,
                  "dropout_rate": self.dropout_rate})
    return config

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
  '''
    Embeds tokens and positions.
  '''
  def __init__(self,max_len,vocab_size,embed_dim):
    super(TokenAndPositionEmbedding,self).__init__()
    self.max_len = max_len
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim = embed_dim)
    self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len,output_dim=embed_dim)

  def call(self,x):
    maxlen = tf.shape(x)[-1]
    positions = tf.range(start=0,limit=maxlen,delta=1)
    positions = self.pos_emb(positions)
    x = self.token_emb(x)
    return x + positions

  def get_config(self):
    config = super().get_config()
    config.update({"max_len": self.max_len, "vocab_size": self.vocab_size,
                  "embed_dim": self.embed_dim})
    return config

def make_gpt(num_blocks: int, max_length: int, VOCAB_SIZE: int):
  '''
    creates a GPT with a specified number of transformer blocks.

      Args:
        num_blocks: number of transformer blocks
        max_length: context window
        VOCAB_SIZE: vocabulary size
      Returns:
        gpt: GPT model
  '''
  EMBEDDING_DIM = 256
  N_HEADS = 4
  KEY_DIM = 256
  FEED_FORWARD_DIM = 256

  inputs = tf.keras.layers.Input(shape=(None,),dtype=tf.int32)
  x = TokenAndPositionEmbedding(max_length,VOCAB_SIZE,EMBEDDING_DIM)(inputs)
  for i in range(num_blocks):
    x, attentions_scores = TransformerBlock(N_HEADS,KEY_DIM,EMBEDDING_DIM,FEED_FORWARD_DIM)(x)
  outputs = tf.keras.layers.Dense(VOCAB_SIZE,activation="softmax")(x)

  gpt = tf.keras.models.Model(inputs = inputs, outputs =[outputs, attentions_scores])
  gpt.summary()

  return gpt

def save_gpt(gpt, filename: str):
  '''
    saves a GPT model.

      Args:
        gpt: GPT model
        filename: name of the model
      Returns:
        None; saves model and a list of layer names to files.
  '''
  layer_name_store = []
  for layer in gpt.layers:
    layer.name = layer.name+"_original"
    layer_name_store.append(layer.name)

  print("New layer names:")
  print("===========================================")
  gpt.summary()

  gpt.save_weights(f"{filename}.weights.h5")
  print(f"model saved with name: {filename}.")

  f = open(f"layer_store_{filename}.txt", "w")
  for item in layer_name_store:
      f.write("%s\n" % item)
  f.close()
  print(f"layer names saved in file: layer_store_{filename}.")

def make_finetune_gpt(num_new_blocks: int, freeze_old_layers = True):
  '''
    Creates a finetuning model from a set foundation model. 

      Args:
        num_new_blocks: number of new transformer blocks to add
        freeze_old_layers: whether to freeze the old layers
      Returns:
        gpt_ft: finetuning model
  '''
  VOCAB_SIZE = 100
  max_length = 166
  gpt_ft = make_gpt(2+num_new_blocks, max_length, VOCAB_SIZE)

  f = open("CafChem/data/layer_store_GPT_ZN305_50epochs.txt", "r")
  layer_name_store_raw = f.readlines()
  f.close()

  print("Reading in layers:")
  layer_name_store = []
  for line in layer_name_store_raw:
      line = line.replace("\n","")
      layer_name_store.append(line)
      print(line)
  print("===========================================")

  new_layers = num_new_blocks + 1
  for i,layer in enumerate(gpt_ft.layers[:-new_layers]):
    layer.name = layer_name_store[i]
    print(f"{layer.name} has been named!")

  for i,layer in enumerate(gpt_ft.layers[-new_layers:-1]):
    layer.name = f"transformer_block_X_{i+1}"
    print(f"{layer.name} has been named!")

  gpt_ft.layers[-1].name = "dense_X"

  gpt_ft.load_weights("CafChem/data/GPT_ZN305_50epochs.weights.h5", skip_mismatch=True)

  if freeze_old_layers == True:
    for layer in gpt_ft.layers[0:-new_layers]:                 #make old layers freeze and only train new layers
      layer.trainable=False
      print(f"setting layer {layer.name} untrainable.")

    for layer in gpt_ft.layers[-new_layers:]:
      layer.trainable=True
      print(f"setting layer {layer.name} trainable.")

  gpt_ft.summary()
  return gpt_ft

def unfreeze_gpt(gpt_model):
  '''
    unfreezes all layers in a model.

      Args:
        gpt_model: model to unfreeze
      Returns:
        gpt_model: unfrozen model
  '''
  for layer in gpt_model.layers:
    layer.trainable=True
    print(f"setting layer {layer.name} trainable.")

  gpt_model.summary()

  return gpt_model

def load_gpt(filename: str, total_layers: int, max_length: int, VOCAB_SIZE: int):
  '''
    loads a GPT model.

      Args:
        filename: name of the model (h5 file)
        total_layers: total number of transformer blocks
        max_length: context window
        VOCAB_SIZE: vocabulary size
      Returns:
        gpt_load: loaded GPT model
  '''

  gpt_load = make_gpt(total_layers, max_length, VOCAB_SIZE)

  f = open(f"layer_store_{filename}.txt", "r")
  layer_name_store_raw = f.readlines()
  f.close()

  layer_name_store = []
  for line in layer_name_store_raw:
      line = line.replace("\n","")
      layer_name_store.append(line)
      print(line)

  for i,layer in enumerate(gpt_load.layers):
    layer.name = layer_name_store[i]
    print(f"{layer.name} has been named!")

  gpt_load.load_weights(f"{filename}.weights.h5", skip_mismatch=True)
  print(f"model loaded with name: {filename}.")
  gpt_load.summary()

  return gpt_load

def load_foundation():
  '''
    loads the GPT Foundation model.

      Args:
        None
      Returns:
        gpt_load: loaded GPT model
  '''
  VOCAB_SIZE = 100
  max_length = 166
  gpt_load = make_gpt(2, max_length, VOCAB_SIZE)

  f = open("CafChem/data/layer_store_GPT_ZN305_50epochs.txt", "r")
  layer_name_store_raw = f.readlines()
  f.close()

  layer_name_store = []
  for line in layer_name_store_raw:
      line = line.replace("\n","")
      layer_name_store.append(line)
      print(line)

  for i,layer in enumerate(gpt_load.layers):
    layer.name = layer_name_store[i]
    print(f"{layer.name} has been named!")

  gpt_load.load_weights("CafChem/data/GPT_ZN305_50epochs.weights.h5", skip_mismatch=True)
  print("Foundation model loaded.")
  gpt_load.summary()

  return gpt_load