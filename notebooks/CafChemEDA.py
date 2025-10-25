import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from rdkit import Chem
import deepchem as dc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def featurize(smiles_list: list, y: list,
              ions_to_clean = ["[Na+].", ".[Na+]"], featurizer = "rdkit"):
  '''
  featurize a list of SMILES using RDKit or Mordred, clean counterions, 
  and remove NANs. treats target list as well so values returned match

  Args:
    smiles_list: list of SMILES
    target_list: list of target values
    ions_to_clean: list of ions to remove from SMILES
    featurizer: "rdkit" or "mordred" or "fingerprints"
  Returns:
    X: list of feature vectors
    y: list of target values
    XA: list of SMILES with ions removed
  '''
  Xa = []
  for smile, value in zip(smiles_list, y):
    for ion in ions_to_clean:
      smile = smile.replace(ion,"")
    Xa.append(smile)

  mols = [Chem.MolFromSmiles(smile) for smile in Xa]

  if featurizer == "fingerprints":
    featurizer=dc.feat.CircularFingerprint(size=1024)
    featname="CircularFingerprint"
  elif featurizer == "rdkit":
    featurizer=dc.feat.RDKitDescriptors()
    featname="RDKitDescriptors"
  elif featurizer == "mordred":
    featurizer=dc.feat.MordredDescriptors()
    featname="MordredDescriptors"
  else:
    raise ValueError("featurizer must be 'fingerprints', 'rdkit', or 'mordred'")

  f = featurizer.featurize(mols)

  # y = np.array(y)
  # Xa = np.array(Xa)

  nan_indicies = np.isnan(f)
  bad_rows = []
  for i, row in enumerate(nan_indicies):
      for item in row:
          if item == True:
              if i not in bad_rows:
                  print(f"Row {i} has a NaN.")
                  bad_rows.append(i)

  print(f"Old dimensions are: {f.shape}.")

  for j,i in enumerate(bad_rows):
      k=i-j
      f = np.delete(f,k,axis=0)
      y = np.delete(y,k,axis=0)
      Xa = np.delete(Xa,k,axis=0)
      print(f"Deleting row {k} from arrays.")

  print(f"New dimensions are: {f.shape}")
  if f.shape[0] != len(y) or f.shape[0] != len(Xa):
    raise ValueError("Number of rows in X and y do not match.")

  nan_indicies = np.isnan(f)
  bad_rows = []
  for i, row in enumerate(nan_indicies):
      for item in row:
          if item == True:
              if i not in bad_rows:
                  print(f"Row {i} has a NaN.")
                  bad_rows.append(i)

  return f, y, Xa

def scale_pca(f: np.array, use_scaler = True,
                    use_pca = True, pca_size = 30, seed = 42):
  '''
    receives feature array, target list and smiles list. Can perform scaling and/or 
    pca (as inidicated in the function call). Performs train/test split based on 
    value of splits.  

    Args:
      f: feature array
      use_scaler: boolean for using scaler (optional)
      use_pca: boolean for using pca (optional)
      pca_size: number of components for pca (optional)
      seed: random seed for train/test split (optional)
    Returns:
      f_final: feature array after scaling and/or PCA
  '''

  if use_scaler == True and use_pca == True:
      
    scaler = StandardScaler()
    scalername = "StandardScaler"
    scaler.fit(f)
    f_scaled = scaler.transform(f)

    pca = PCA(n_components=pca_size)
    pca.fit(f_scaled)
    f_final = pca.transform(f_scaled)
    
  elif use_scaler == True and use_pca == False:

    scaler = StandardScaler()
    scalername = "StandardScaler"
    scaler.fit(f)
    f_final = scaler.transform(f)
    pca = None

  elif use_scaler == False and use_pca == True:

    pca = PCA(n_components=pca_size)
    pca.fit(f)
    f_final = pca.transform(f)
    scaler = None
    
  elif use_scaler == False and use_pca == False:

    f_final = f
    pca = None
    scaler = None

  print("Pre-processing done.")

  return f_final

def clean_ions(smiles):
  '''
    Takes a smiles string and cleans the following ions from it: [Na+], [Cl-], [K+],
    [Br-], [I-], [Ca2+].

    Args:
      smiles: smiles string
    Returns:
      smiles: cleaned smiles string
  '''
  smiles = smiles.replace("[Na+].","").replace("[Cl-].","").replace(".[Cl-]","").replace(".[Na+]","")
  smiles = smiles.replace("[K+].","").replace("[Br-].","").replace(".[K+]","").replace(".[Br-]","")
  smiles = smiles.replace("[I-].","").replace(".[I-]","").replace("[Ca2+].","").replace(".[Ca2+]","")
  return smiles

def make_classes(filename: str, target_name: str, num_classes: int):
  '''
    Takes a CSV files with a SMILES column and a target column, divides it into the
    requested number of classes, sets the boundaries for thise classes, and assigns each 
    datapoint to a class. Returns a dataframe with an additional column containing the 
    classes. Also cleans ions from the SMILES strings.

      Args:
        filename: name of the CSV file
        target_name: name of the target column
        num_classes: number of classes to divide the data into
      Returns:
        df: dataframe with the classes
  '''
  df = pd.read_csv(filename)
  df.sort_values(by=[target_name],inplace=True)

  total_samples = len(df)
  samples_per_class = total_samples // num_classes
  print(f"Samples per class: {samples_per_class}, total samples:{total_samples}")


  bottom_range = df[target_name].iloc[0].item()

  range_cutoffs = []
  range_cutoffs.append(bottom_range)

  for i in range(samples_per_class,total_samples-num_classes, samples_per_class):
    range_cutoffs.append(df[target_name].iloc[i].item())
  
  if df[target_name].iloc[-1].item() not in range_cutoffs:
    range_cutoffs.append(df[target_name].iloc[-1].item())

  #print(range_cutoffs)
  class_labels = []
  for i in range(len(range_cutoffs)-1):
    label_string = f"{range_cutoffs[i]} < {range_cutoffs[i+1]}"  
    class_labels.append(label_string)
  
  labels_list = []
  for target in df[target_name]:
    for i in range(len(range_cutoffs)-1):
      if target <= range_cutoffs[i+1]:
        labels_list.append(class_labels[i])
        break

  df["class labels"] = labels_list

  columns = df.columns
  for column in columns:
    if "Smiles" in column or "SMILES" in column or "smiles" in column:
      smiles_name = column
      break
    
  df[smiles_name] = df[smiles_name].apply(clean_ions)

  return df  

def remove_outliers(f: np.array, y: list, use_f = True, use_y = False):
  '''
    Identifies outliers using Elliptic Envelope and removes them from the
    feature matrix, the target list, and the SMILES list.

    Args:
      f: feature matrix
      y: target list
      use_f: Boolean, use features to identify outliers?
      use_y: Boolean, use target values to ientify outliers.
    Returns:
      f: feature matrix without outliers
      y: target list without outliers
      Xa: SMILES list without outliers
  '''
  if use_f == True and use_y == True:
      print("Can only use one criteria at a time! Choosing f.")
      use_y = False
  
  if use_f == False and use_y == False:
      print("Must use one criteria at least! Choosing f.")
      use_f = True
  
  if use_f:
      outlier_detector = EllipticEnvelope(contamination=0.01)
      outlier_detector.fit(f)
      outlier_array = outlier_detector.predict(f)
      indicies = np.where(outlier_array == -1)
      print("===========================================================")
      print(f"Outliers found in the following locations: {indicies} using f.")
  if use_y:
      outlier_detector = EllipticEnvelope(contamination=0.01)
      outlier_detector.fit(f)
      outlier_array = outlier_detector.predict(f)
      indicies = np.where(outlier_array == -1)
      print("===========================================================")
      print(f"Outliers found in the following locations: {indicies} using y.")

  print("Starting outlier removal.")
  for j,i in enumerate(indicies[0]): #indicies[0] for elliptic, indicies for quartile
      f = np.delete(f,i-j,axis=0)
      y = np.delete(y,i-j,axis=0)
      #print(f"Deleting row {i-j} from dataset")
  print(f"New dimensions are: {f.shape}; removed {len(indicies[0])} outliers")

  
  return f, y

def dimreduction(X: list, y_raw:list, decomptype = 'P', whiten=False, perplexity=30):
    '''
    Performs dimensionality reduction on the feature matrix using PCA or t-SNE.
    Args:
      X: feature matrix
      y_raw: target list
      decomptype: type of dimensionality reduction to perform ('P' for PCA, 't' for t-SNE)
      whiten: boolean, whether to whiten the data
      perplexity: perplexity for t-SNE
    Returns:
      X_decomp: feature matrix after dimensionality reduction
    '''
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    num_classes = len(set(y_raw))

    X_scaled, y_raw = remove_outliers(X_scaled, y_raw)

    if type(y_raw[0]) != int:
        unique_classes = set(y_raw)
        class_dict = {}
        for i,y_class in enumerate(unique_classes):
            class_dict[str(y_class)] = i
            
        y = []
        labels = []
        for val in y_raw:
            y.append(class_dict[val])
            labels.append(val)
        labels = list(set(labels))
        print('target converted to ints')
    else:
        y = y_raw
        labels = list(set(y))
    #print(labels)
       
        
    if decomptype == 'P':
        pca = PCA(n_components=2, whiten=whiten)
        pca.fit(X_scaled)
        X_decomp = pca.transform(X_scaled)
    elif decomptype == 't':
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_decomp = tsne.fit_transform(X_scaled)
        print(f"KL divergence: {tsne.kl_divergence_}")

    print(f"Shape is: {X_decomp.shape}")
    print(f"Number of labels: {len(labels)}")

    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    colors = colors[:num_classes]
    cmap = ListedColormap(colors)

    plt.figure(figsize=(8,8))
    scatter = plt.scatter(X_decomp[:,0],X_decomp[:,1], c=y, cmap=cmap)
    plt.legend(*scatter.legend_elements())

    plt.xlabel("1st component")
    plt.ylabel("2nd component")

def create_feature_df(f: list, y_raw:list):
    '''
    Creates a dataframe with the feature matrix and the target list.
    Args:
      f: feature matrix
      y_raw: target list
    Returns:
      feat_df: dataframe with the feature matrix and the target list
    '''
    if type(y_raw[0]) != int:
        unique_classes = set(y_raw)
        class_dict = {}
        for i,y_class in enumerate(unique_classes):
            class_dict[str(y_class)] = i
            
        y = []
        for val in y_raw:
            y.append(class_dict[val])
        print('target converted to ints')
    
    fa = np.asarray(f)
    print(fa.shape)
    #ya = np.asarry(y)
    total_dict = {}
    i=0
    for i in range(fa.shape[1]):
        col_name = f'feature_{i}'
        total_dict[col_name] = fa[:,i]
        i += 1
        
    total_dict['target'] = y
        
    feat_df = pd.DataFrame(total_dict)

    return feat_df