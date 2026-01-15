import tensorflow as tf
import numpy as np
import pandas as pd
import deepchem as dc
import time
import random
import transformers
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def make_datasets(df, sequence_column = 'Sequence'):
  '''
    Tokenizes a dataset and returns the input and target arrays.

      Args:
        filename: name of new dataset
        sequence_column: name of the sequence column
      Returns:
        fx: input array
        fy: target array
        VOCAB_SIZE: vocabulary size
        tokenizer: tokenizer object
        max_length: longest sequence chain
  '''

  seqs_raw = df[sequence_column].tolist()
  Xa = []
  for seq in seqs_raw:
    prot = ''
    for char in seq:
      prot += char + ' '
    Xa.append(prot)

  #===========================================================================================
  #featurize

  tokenizer= AutoTokenizer.from_pretrained('qilowoq/AbLang_light')
  featname="SMILES Tokenizer"

  #fl = list(map(lambda x: tokenizer.encode(x),Xa))

  biggest = 1
  smallest = 200
  for i in range(len(df['Sequence'])):
      temp = len(df['Sequence'].iloc[i])
      if temp > biggest:
          biggest = temp
      if temp < smallest:
          smallest = temp

  print(biggest, smallest)

  string_length = smallest - 1
  max_length = biggest

  fl2 = list(map(lambda x: tokenizer.encode(x,padding="max_length",
                           max_length=max_length,truncation=True),Xa))

  fl2set=set()
  for sublist in fl2:
    fl2set.update(sublist)
  temp_vocab_size = len(fl2set)

  # f = open("protein_vocab.txt", "r")
  # lines = f.readlines()
  # f.close()
  VOCAB_SIZE = len(tokenizer)
  print("Vocabulary size for this dataset: ",VOCAB_SIZE)

  x = []
  y = []
  i=0
  for string in fl2:
      x.append(string[1:max_length-1]) #string_length
      y.append(string[2:max_length]) #string_length+1

  x = np.array(x)
  y = np.array(y)
  print("Number of features and datapoints, targets: ",x.shape,y.shape)

  #===========================================================================================
  print("featurization done with: ",featname)

  fx = x
  fy = y

  return fx, fy, VOCAB_SIZE, tokenizer, max_length #fl2set

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

def make_gpt(num_blocks: int, max_length: int, VOCAB_SIZE: int,model_dimension: int, aheads: int):
  '''
    creates a GPT with a specified number of transformer blocks.

      Args:
        num_blocks: number of transformer blocks
        max_length: context window
        VOCAB_SIZE: vocabulary size
        model_dimension: model dimension for embedding, key, and feed forward
      Returns:
        gpt: GPT model
  '''
  EMBEDDING_DIM = model_dimension
  N_HEADS = aheads
  KEY_DIM = model_dimension
  FEED_FORWARD_DIM = model_dimension

  inputs = tf.keras.layers.Input(shape=(None,),dtype=tf.int32)
  x = TokenAndPositionEmbedding(max_length,VOCAB_SIZE,EMBEDDING_DIM)(inputs)
  for i in range(num_blocks):
    x, attentions_scores = TransformerBlock(N_HEADS,KEY_DIM,EMBEDDING_DIM,FEED_FORWARD_DIM)(x)
  outputs = tf.keras.layers.Dense(VOCAB_SIZE,activation="softmax")(x)

  gpt = tf.keras.models.Model(inputs = inputs, outputs =[outputs, attentions_scores])
  gpt.summary()

  return gpt

def gen_proteins(prompts: list, use_ramp: bool, model, tokenizer, TEMP: float, VOCAB_SIZE: int,
                 max_gen_length: int, rn_seed = 42):
  '''
    use a GPT model to generate novel proteins.

      Args:
        prompts: a list of prompts for inference
        use_ramp: Boolean to use temperature ramp during inference
        model: the GPT model to use
        tokenizer: tokenizer to use
        TEMP: temperature for inference
        VOCAB_SIZE: vocabulary size
        max_gen_length: maximum sequence length
        rn_seed: random seed
      Returns:
        img: image of generated proteins
  '''
  tf.random.set_seed(rn_seed)

  batch_length = len(prompts)
  prompt_length = len(prompts[0])

  test_string = []
  for seq in prompts:
    prot = ''
    for char in seq:
      prot += char + ' '
    test_string.append(prot)


  test_xlist = np.empty([batch_length,prompt_length+1], dtype=int)

  test_tokenized = list(map(lambda x: tokenizer.encode(x),test_string))
  for i in range(batch_length):
      test_xlist[i][:] = test_tokenized[i][:prompt_length+1]
  test_array = np.array(test_xlist)
  print(test_array)

  proba = np.empty([batch_length,VOCAB_SIZE])
  rescaled_logits = np.empty([batch_length,VOCAB_SIZE])
  preds = np.empty([batch_length])
  gen_proteins = np.empty([batch_length])

  c_final = max_gen_length - prompt_length
  sig_start = 0.10

  for c in range(0,c_final,1):

      c_o = int(c_final*sig_start)
      if use_ramp == True:
          T_int = TEMP*(1/(1+np.exp(-(c-c_o))))
      else:
          T_int = TEMP

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

  gen_proteins = list(map(lambda x: tokenizer.decode(x),test_array))
  #gen_proteins = list(map(lambda x: tokenizer.convert_tokens_to_string(x),
  #                          gen_proteins))

  clean_proteins = []
  for seq in gen_proteins:
    seq = seq.replace('[CLS]','')
    seq = seq.replace('[PAD]','')
    seq = seq.replace('[SEP]','')
    seq = seq.replace(' ','')
    clean_proteins.append(seq)

  final_sequences = []
  for seq in clean_proteins:
      if seq not in final_sequences:
          final_sequences.append(seq)

  print(f"Generated {len(final_sequences)} unique proteins.")

  return final_sequences

def save_gpt(gpt, filename: str, vocab_size: int, max_length: int):
    '''
    Saves a GPT model for protein sequences.
      Args:
        gpt: GPT model
        filename: name of the model
        vocab_size: vocabulary size
        max_length: context window
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
    with open(f"layer_store_{filename}.txt", "w") as f:
        for item in layer_name_store:
            f.write("%s\n" % item)
        f.write("Parameters:\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"max_length: {max_length}\n")
    print(f"model parameters saved in file: layer_store_{filename}.")

def load_gpt(filename: str, total_layers: int, model_dimension: int, aheads: int):
    '''
    Loads a protein GPT model.
      Args:
        filename: name of the model (h5 file)
        total_layers: total number of transformer blocks
        model_dimension: model dimension for embedding, key, and feed forward
        aheads: number of heads
      Returns:
        gpt_load: loaded GPT model
    '''
  
    with open(f"layer_store_{filename}.txt", "r") as f:
        layer_name_store_raw = f.readlines()
    layer_name_store = []
    for i, line in enumerate(layer_name_store_raw):
        if line == "Parameters:\n":
            parameter_start = i
            break
        line = line.replace("\n","")
        layer_name_store.append(line)
        print(line)
    
    vocab_size = int(layer_name_store_raw[parameter_start+1].replace("\n","").split(":")[1])
    max_length = int(layer_name_store_raw[parameter_start+2].replace("\n","").split(":")[1])
    gpt_load = make_gpt(total_layers, max_length, vocab_size, model_dimension, aheads)

    for i,layer in enumerate(gpt_load.layers):
        layer.name = layer_name_store[i]
        print(f"{layer.name} has been named!")
  
    gpt_load.load_weights(f"{filename}.weights.h5", skip_mismatch=True)

    print(f"model loaded with name: {filename}.")
    print(f"vocab_size: {vocab_size}")
    print(f"max_length: {max_length}")

    gpt_load.summary()
    return gpt_load

def make_finetune_gpt(old_gpt_name: str, old_layers: int, model_dimension: int,
                      aheads: int, num_new_blocks: int, freeze_old_layers = True):
    '''
    Creates a finetuning model for protein sequences from a set foundation model.
      Args:
        old_gpt: foundation model
        old_layers: total number of transformer blocks in old gpt
        model_dimension: model dimension for embedding, key, and feed forward
        aheads: number of heads
        num_new_blocks: number of new transformer blocks to add
        freeze_old_layers: whether to freeze the old layers
      Returns:
        gpt_ft: finetuning model
    '''
  
    with open(f"layer_store_{old_gpt_name}.txt", "r") as f:
        layer_name_store_raw = f.readlines()
    print("Reading in layers:")

    layer_name_store = []
    block_count = 0
    for i,line in enumerate(layer_name_store_raw):
        if line == "Parameters:\n":
            parameter_start = i
            break
        if "transformer_block" in line:
            block_count += 1
        line = line.replace("\n","")
        layer_name_store.append(line)
        print(line)
    print("===========================================")
    vocab_size = int(layer_name_store_raw[parameter_start+1].replace("\n","").split(":")[1])
    max_length = int(layer_name_store_raw[parameter_start+2].replace("\n","").split(":")[1])
    print(f"vocab_size: {vocab_size}")
    print(f"max_length: {max_length}")

    gpt_ft = make_gpt(block_count+num_new_blocks, max_length, vocab_size, model_dimension, aheads)

    new_layers = num_new_blocks + 1
    for i,layer in enumerate(gpt_ft.layers[:-new_layers]):
        layer.name = layer_name_store[i]
        print(f"{layer.name} has been named!")
    for i,layer in enumerate(gpt_ft.layers[-new_layers:-1]):
        layer.name = f"transformer_block_X_{i+1}"
        print(f"{layer.name} has been named!")
    gpt_ft.layers[-1].name = "dense_X"
    gpt_ft.load_weights(f"{old_gpt_name}.weights.h5", skip_mismatch=True)

    if freeze_old_layers == True:
        for layer in gpt_ft.layers[0:-new_layers]:
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