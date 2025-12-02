from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM, EsmForSequenceClassification
import torch
from torch import inf
import py3Dmol
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import esm


def get_protein_from_pdb(pdb_id):
  '''
  Get protein structure from PDB
  Input: PDB ID
  Output: PDB file
  '''
  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  r = requests.get(url)
  return r.text

def show_protein(pdb_id):
  '''
  Show protein structure from PDB
  Input: PDB ID
  Output: structure view
  '''
  colors = ['turquoise', 'red', 'yelllow', 'blue', 'green', 'orange', 'purple']
  chains = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

  pdb_str = get_protein_from_pdb(pdb_id)
  pdbview = py3Dmol.view(width=800, height=600)
  pdbview.addModel(pdb_str, 'pdb')
  for i, chain in enumerate(chains):
    try:
      pdbview.setStyle({'chain': chain}, {'cartoon': {'color': colors[i]}})
    except:
      print('No additional chains')
  pdbview.zoomTo()
  return pdbview

def extract_sequence(pdb_id):
  '''
  Extract sequence from PDB file
  Input: PDB ID
  Output: dictionary of chains and sequences in three-letter and one-letter formats
  '''
  pdb_str = get_protein_from_pdb(pdb_id)
  chains = {}

  #print(pdb_str.split('\n')[0])
  for line in pdb_str.split('\n'):
    parts = line.split()
    try:
      if parts[0] == 'SEQRES':
        if parts[2] not in chains:
          chains[parts[2]] = []
        chains[parts[2]].extend(parts[4:])
    except:
      print('Blank line')

    chains_ol = {}
    for chain in chains:
      chains_ol[chain] = three_to_one(chains[chain])

  return chains, chains_ol

def one_to_three(one_seq):
  '''
  Convert one-letter code to three-letter code
  Input: one-letter code
  Output: three-letter code
  '''
  rev_aa_hash = {
      'A': 'ALA',
      'R': 'ARG',
      'N': 'ASN',
      'D': 'ASP',
      'C': 'CYS',
      'Q': 'GLN',
      'E': 'GLU',
      'G': 'GLY',
      'H': 'HIS',
      'I': 'ILE',
      'L': 'LEU',
      'K': 'LYS',
      'M': 'MET',
      'F': 'PHE',
      'P': 'PRO',
      'S': 'SER',
      'T': 'THR',
      'W': 'TRP',
      'Y': 'TYR',
      'V': 'VAL'
  }

  try:
    three_seq = rev_aa_hash[one_seq]
  except:
    three_seq = 'X'

  return three_seq

def three_to_one(three_seq):
  '''
  Convert three-letter code to one-letter code
  Input: three-letter code
  Output: one-letter code
  '''
  aa_hash = {
      'ALA': 'A',
      'ARG': 'R',
      'ASN': 'N',
      'ASP': 'D',
      'CYS': 'C',
      'GLN': 'Q',
      'GLU': 'E',
      'GLY': 'G',
      'HIS': 'H',
      'ILE': 'I',
      'LEU': 'L',
      'LYS': 'K',
      'MET': 'M',
      'PHE': 'F',
      'PRO': 'P',
      'SER': 'S',
      'THR': 'T',
      'TRP': 'W',
      'TYR': 'Y',
      'VAL': 'V'
  }

  one_seq = []
  for residue in three_seq:
    try:
      one_seq.append(aa_hash[residue])
    except:
      one_seq.append('X')

  return one_seq

class gen_mask_fill():
  '''
  Class to generate masks and fill them with ESM predictions
  '''
  def __init__(self, checkpoint: str, seq: list, num_to_mask: int):
    '''
    Constructor for mask filling
    Input:
    - checkpoint: path to ESM model
    - seq: sequence to mask
    - num_to_mask: number of residues to mask
    '''
    self.checkpoint = checkpoint
    self.seq = seq
    self.num_to_mask = num_to_mask

  def start_model(self):
    '''
    Start ESM model and tokenizer
    '''
    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
    self.model = EsmForMaskedLM.from_pretrained((self.checkpoint))

  def mask_tokens(self):
    '''
    Mask tokens in sequence
    Output:
    - seq_ids: sequence of tokens
    - masked_chain: masked sequence
    - masked_chain_ids: masked sequence of tokens
    '''
    self.seq_ids = self.tokenizer(''.join(self.seq))['input_ids']

    masked_chain = []
    self.rdn_ixd = torch.randint(1, len(self.seq)-1, (self.num_to_mask,))
    for i, token in enumerate(self.seq):
      if i in self.rdn_ixd:
        masked_chain.append('<mask>')
      else:
        masked_chain.append(token)

    self.masked_chain = masked_chain
    self.masked_chain_ids = self.tokenizer(''.join(masked_chain))['input_ids']

    return self.seq_ids, self.masked_chain, self.masked_chain_ids

  def unmask(self):
    '''
    Unmask tokens in sequence
    Output:
    - model_preds: predictions from ESM
    '''
    model_out = self.model(**self.tokenizer(text = ''.join(self.masked_chain), return_tensors='pt'))

    model_preds = []
    for row in model_out.logits[0]:
      probs = torch.softmax(row.detach().clone(), dim=0)
      model_preds.append(torch.argmax(probs).detach().clone().item())

    self.model_preds = model_preds
    return self.model_preds

  def new_seq_from_ids(self):
    '''
    Convert tokens to sequence
    Output:
    - new_seq: new sequence
    '''
    raw = self.tokenizer.decode(self.model_preds)
    self.new_seq = raw.replace('<cls>','').replace('<eos>','').replace(' ','')

  def compare_seqs(self):
    '''
    Compare original and new sequences
    Output:
    - chain: original sequence
    - new_seq: new sequence
    '''
    self.new_seq_from_ids()
    self.chain = ''.join(self.seq).replace('<cls>','').replace('<eos>','')
    print(f"Original: {self.chain}")
    print(f"Novel   : {self.new_seq}")

    i = 1
    for char_o, char_n in zip(self.seq,self.new_seq):
      if self.masked_chain_ids[i-1] == 32:
        mask = 'was masked.'
      else:
        mask = 'was not masked.'

      if char_o != char_n:
        print(f"Residue {i} changed {one_to_three(char_o)} --> {one_to_three(char_n)}. This token {mask}")
      i += 1

    return self.chain, self.new_seq
  
  def compare_seqs_naive(self):
    '''
    Compare original and new sequences by % of differences
    Output:
    - chain: original sequence
    - new_seq: new sequence
    '''
    self.new_seq_from_ids()
    self.chain = ''.join(self.seq).replace('<cls>','').replace('<eos>','')
    print(f"Original: {self.chain}")
    print(f"Novel   : {self.new_seq}")

    num_diff = 0
    for char_o, char_n in zip(self.seq,self.new_seq):

      if char_o != char_n:
        num_diff += 1
    
    print(f"Number of differences: {num_diff} out of {len(self.seq)}")
    print(f"Percentage of differences: {num_diff/len(self.seq):.3f}")

    return self.chain, self.new_seq

class embed_proteins():
  '''
  Class to embed proteins using ESM
  '''
  def __init__(self, checkpoint: str, list_seqs: list):
    '''
    Constructor for embedding proteins
    Input:
    - checkpoint: path to ESM model
    - list_seqs: list of sequences to embed
    '''
    self.checkpoint = checkpoint
    self.list_seqs = list_seqs
    self.model_start_flag = 0

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {self.device}")

  def start_model(self):
    '''
    Start ESM model and tokenizer
    '''
    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
    self.model = EsmModel.from_pretrained((self.checkpoint))
    self.model_start_flag = 1
    print(f"Model loaded from {self.checkpoint}")

  def embed_seqs(self):
    '''
    Embed sequences
    Output:
    - embeddings: embeddings of sequences
    '''
    if self.model_start_flag == 0:
      self.start_model()

    model_inputs = self.tokenizer(self.list_seqs, padding=True, return_tensors='pt')
    model_input = {k: v.to(self.device) for k, v in model_inputs.items()}

    self.model.to(self.device)
    self.model.eval()

    with torch.no_grad():
      model_out = self.model(**model_input)
      self.embeddings = model_out.last_hidden_state.mean(dim=1)

    self.embeddings = self.embeddings.detach().cpu().numpy()

    return self.embeddings

  def compare_embeddings(self, a: int, b: int):
    '''
    Compare embeddings of two sequences using cosine similarity
    Input:
    - a: index of first sequence
    - b: index of second sequence
    '''
    v1 = self.embeddings[a].reshape(1,-1)
    v2 = self.embeddings[b].reshape(1,-1)

    ov = cosine_similarity(v1,v2)

    print(f"Overlap between protein {a} and {b}: {ov[0][0]:.5f}")