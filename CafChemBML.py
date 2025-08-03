import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolsToGridImage
import matplotlib.pyplot as plt
import re
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import r2_score
import mordred
import deepchem as dc
import pickle
from rdkit.Chem.rdFMCS import FindMCS

def smiles_to_canon(smiles: str):
  '''
    Accepts a SMILES string and converts it to canonical SMILES.
    
    Args:
      smiles: SMILES string
    Returns:
      canon_smiles: canonical SMILES string
  '''
  canon_smiles = Chem.CanonSmiles(smiles)
  return canon_smiles

def process_chembl_csv(filename: str, units = "nM"):
  '''
    Expects a ChEMBL dataframe (';' separator). Units are in nM by default, but can be changed.
    Selects rows with "=" and "IC50" in the "Standard Type" column.
    SMILES are converted to canonical form, and duplicates are removed.
    IC50 values are converted to logIC50 values and included as an additional column.
    
    Args:
      filename: name of the CSV file
      units: units for the "Standard Units" column. Default is nM.
    Returns:
      df: Pandas dataframe with SMILES, IC50 and logIC50 columns.
  '''
  df_primitive = pd.read_csv(filename,sep=";")
  df_relations = df_primitive[["Smiles","Standard Type","Standard Relation","Standard Value","Standard Units"]]
  
  df_type = df_relations[df_relations["Standard Relation"] == "'='"]
  df_units = df_type[df_type["Standard Type"] == "IC50"]
  df = df_units[df_units["Standard Units"] == units]
  df.drop(["Standard Relation","Standard Type", "Standard Units"],axis=1,inplace=True)
  df = df.rename(columns={"Smiles":"SMILES","Standard Value": "IC50"})
  df["SMILES"] = df["SMILES"].apply(smiles_to_canon)
  
  df = df.drop_duplicates(subset=["SMILES"])

  logs = [np.log10(x) for x in df["IC50"]]
  df["logIC50"] = logs

  print("=========================================================================")
  print(f"Dataframe created from {filename}. Number of unique SMILES: {len(df)}")
  print("=========================================================================")

  return df

def list_hist(value_list: list, bins = 100):
  '''
  print a histogram of a list of values

  Args:
    value_list: list of values
    bins: number of bins in the histogram
  Returns:
    None; plots the histogram
  '''
  plt.hist(value_list,bins=100)
  plt.show  

def featurize(smiles_list: list, y: list,
              ions_to_clean = ["[Na+].", ".[Na+]"], featurizer = "rdkit"):
  '''
  featurize a list of SMILES using RDKit or Mordred, clean counterions, 
  and remove NANs. treats target list as well so values returned match

  Args:
    smiles_list: list of SMILES
    target_list: list of target values
    ions_to_clean: list of ions to remove from SMILES
    featurizer: "rdkit" or "mordred"
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

def remove_outliers(f: np.array, y: list, Xa: list, use_f = True, use_y = False):
  '''
    Identifies outliers using Elliptic Envelope and removes them from the
    feature matrix, the target list, and the SMILES list.

    Args:
      f: feature matrix
      y: target list
      Xa: SMILES list
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
      Xa = np.delete(Xa,i-j,axis=0)
      print(f"Deleting row {i-j} from dataset")
  print(f"New dimensions are: {f.shape}; removed {len(indicies[0])} outliers")

  
  return f, y, Xa

def scale_pca_split(f: np.array, y: list, Xa: list, use_scaler = True,
                    use_pca = False, pca_size = 100, seed = 42, splits = 0.9):
  '''
    receives feature array, target list and smiles list. Can perform scaling and/or 
    pca (as inidicated in the function call). Performs train/test split based on 
    value of splits.  

    Args:
      f: feature array
      y: target list
      Xa: smiles list
      use_scaler: boolean for using scaler (optional)
      use_pca: boolean for using pca (optional)
      pca_size: number of components for pca (optional)
      seed: random seed for train/test split (optional)
      splits: fraction of data to use for training (optional)
    Returns:
      x_train: training feature matrix
      x_valid: validation feature matrix
      y_train: training target list
      y_valid: validation target list
      smiles_train: training smiles
      smiles_valid: validation smiles
      pca: fitted pca model
      scaler: fitter scaler model
  '''
  y = np.array(y)
  Xa = np.array(Xa)
  y_smiles = np.stack((y,Xa),axis=1)

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

  X_train, X_valid, ys_train, ys_valid = train_test_split(f_final,y_smiles,train_size=splits, 
                                                              random_state=seed, shuffle=True)
  
  y_train = ys_train[:,0].astype(float)
  y_valid = ys_valid[:,0].astype(float)
  smiles_train = ys_train[:,1]
  smiles_valid = ys_valid[:,1]

  print("Pre-processing done.")

  return X_train, X_valid, y_train, y_valid, smiles_train, smiles_valid, pca, scaler

class tree_regression():
  '''
    class for performing tree regression; all options in constructor.

      Args:
        method: "tree", "forest", or "gradient"
        n_estimators: number of trees in forest
        max_depth: maximum depth of trees
        min_samples_split: minimum number of samples required to split a node
        min_samples_leaf: minimum number of samples required to be in a leaf node
        max_features: number of features to consider when looking for the best split
        learning_rate: learning rate for gradient boosting
      Returns:
        None
  '''
  def __init__(self, method = "forest", n_estimators = 100, max_depth = None,
               min_samples_split = 2, min_samples_leaf = 1, max_features = 1.0,
               learning_rate = 0.1):
    self.method = method
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_leaf = min_samples_leaf
    self.max_features = max_features
    self.learning_rate = learning_rate

    print("Tree regression class initialized.")

  def fit(self, x_train: np.array, x_valid: np.array, y_train: np.array, y_valid: np.array):
    '''
    Fits a tree-based model and calculates the training and validation scores.

    Args:
      x_train: training feature matrix
      x_valid: validation feature matrix
      y_train: training target list
      y_valid: validation target list
    Returns:
      model: fitted model
    '''
    if self.method == "tree":
      model = DecisionTreeRegressor(max_depth=self.max_depth)
      modelname = "DecisionTree"
    elif self.method == "forest":
      model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_features = self.max_features)
      modelname = "RandomForest"
    elif self.method == "gradient":
      model = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate = self.learning_rate,
                                        max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf, 
                                        max_features=self.max_features,
                                        loss = "huber")
      modelname = "GradientBoostingRegressor"
    

    print("model selected: ",modelname)
    tic = time.perf_counter()
    model.fit(x_train, y_train)
    toc = time.perf_counter()
    fittime = (toc-tic)/60

    print(f"fit model in: {fittime} minutes")

    #evaluate models

    train_score = model.score(x_train,y_train)
    print("score for training set: ",train_score)

    valid_score = model.score(x_valid,y_valid)
    print("score for validation set: ",valid_score)

    print("="*70)

    return model

class linear_svr_methods():
  '''
    class for performing linear or support vector regression; all options in constructor.

      Args:
        method: "linear", "ridge", "lasso", or "svr"
        degree: degree of polynomial for SVR
        C: regularization parameter for SVR
        epsilon: epsilon parameter for SVR
        coef0: coef0 parameter for SVR
      Returns:
        None
  '''
  def __init__(self, method = "linear", degree = 3, C = 2.0, epsilon = 0.1, coef0 = 2.0,
               alpha = 1000, max_iter = 10000):
    self.method = method
    self.degree = degree
    self.C = C
    self.epsilon = epsilon
    self.coef0 = coef0
    self.alpha = alpha
    self.max_iter = max_iter
    
    print("Linear SVR class initialized.")

  def fit(self, x_train: np.array, x_valid: np.array, y_train: np.array, y_valid: np.array):
    '''
      Fits a linear or support vector regression model and calculates the training and validation scores.

      Args:
        x_train: training feature matrix
        x_valid: validation feature matrix
        y_train: training target list
        y_valid: validation target list
      Returns:
        model: fitted model
    '''
    if self.method == "linear":
      model = LinearRegression()
      modelname = "LinearRegression"
    elif self.method == "ridge":
      model = Ridge(alpha=self.alpha)
      modelname = "Ridge"
    elif self.method == "lasso":
      model = Lasso(alpha=self.alpha, max_iter = self.max_iter)
      modelname = "Lasso"
    elif self.method == "svr":
      model = SVR(kernel="poly",degree=self.degree, epsilon=self.epsilon,coef0=self.coef0, C=self.C)
      modelname = "Support Vector Regression"

    print("model selected: ",modelname)
    tic = time.perf_counter()
    model.fit(x_train, y_train)
    toc = time.perf_counter()
    fittime = (toc-tic)/60

    print(f"fit model in: {fittime} minutes")

    #evaluate models

    train_score = model.score(x_train,y_train)
    print("score for training set: ",train_score)

    valid_score = model.score(x_valid,y_valid)
    print("score for validation set: ",valid_score)

    print("="*70)

    return model

class mlp_methods():
  '''
    class for performing multi-layer perceptron regression; all options in constructor.

      Args:
        solver: solver for the optimization
        seed: random seed
        hidden_layer_sizes: number of neurons in each hidden layer
        max_iter: maximum number of iterations
        activation: activation function
        alpha: regularization parameter
      Returns:
        None
  '''
  def __init__(self,solver = "lbfgs", seed = 42, hidden_layer_sizes = [20,20], 
               max_iter = 2000, activation = "relu", alpha = 0.1):
    self.solver = solver
    self.seed = seed
    self.hidden_layer_sizes = hidden_layer_sizes
    self.max_iter = max_iter
    self.activation = activation
    self.alpha = alpha

    print("MLP class initialized.")
  
  def fit(self, x_train: np.array, x_valid: np.array, y_train: np.array, y_valid: np.array):
    '''
      Fits a multi-layer perceptron regression model and calculates the training and validation scores.

      Args:
        x_train: training feature matrix
        x_valid: validation feature matrix
        y_train: training target list
        y_valid: validation target list
      Returns:
        model: fitted model
    '''
    model = MLPRegressor(solver=self.solver,random_state=self.seed, 
                         hidden_layer_sizes=self.hidden_layer_sizes, 
                         max_iter=self.max_iter, activation=self.activation,
                         alpha=self.alpha)
    modelname = "MultiLayerPerceptron"
  
    print("model selected: ",modelname)
    tic = time.perf_counter()
    model.fit(x_train, y_train)
    toc = time.perf_counter()
    fittime = (toc-tic)/60

    print(f"fit model in: {fittime} minutes")

    #evaluate models

    train_score = model.score(x_train,y_train)
    print("score for training set: ",train_score)

    valid_score = model.score(x_valid,y_valid)
    print("score for validation set: ",valid_score)

    print("="*70)

    return model

def plot_predictions(model, x_train: np.array, y_train: np.array, 
                     x_valid: np.array, y_valid: np.array):
  '''
    Plots the predictions of a model on the training and validation sets.

    Args:
      model: fitted model
      x_train: training feature matrix
      y_train: training target list
      x_valid: validation feature matrix
      y_valid: validation target list
    Returns:
      None; plots the predictions
  '''
  y_total = np.concatenate((y_train,y_valid))
  y_train_predicted = model.predict(x_train)
  y_valid_predicted = model.predict(x_valid)

  plt.scatter(y_train,y_train_predicted,color="blue",label="ML-train")
  plt.scatter(y_valid,y_valid_predicted,color="green",label="ML-valid")
  plt.plot(y_total,y_total,color="red",label="Best Fit")
  plt.legend()
  plt.xlabel("known")
  plt.ylabel("predicted")
  plt.show

def save_csv(smiles: np.array, y: np.array, preds: np.array, filename: str):
  '''
  Save a CSV file with your cleaned SMILES list, the original targets, and the 
  predicted values.
  Args:
    smiles: list of SMILES
    y: list of original targets
    preds: list of predicted values
    filename: name of the CSV file to save
  Returns:
    df: pandas dataframe with the saved CSV file
  '''
  dict = {"smiles":smiles, "known":y, "predicted":preds}
  df = pd.DataFrame(dict)
  df.to_csv(filename+".csv")
  return df

def save_features(features: np.array, filename: str):
  '''
  Save the feature array to a pickle file

  Args:
    features: feature array
    filename: name of the pickle file to save
  Returns:
    None; saves the feature array to a pickle file
  '''
  with open(filename+".pkl", "wb") as f:
    pickle.dump(features, f)  
  print(f"Features saved to {filename}.pkl")

def kmeans(x_train: np.array, x_valid: np.array, number_groups = 10, seed= 42,
           init = "k-means++"):
  '''
  Find k-means clusters for the given data.

    Args:
        x_train (np.array): Training data.
        x_valid (np.array): Validation data.
        number_groups (int, optional): Number of clusters. Defaults to 10.
        seed (int, optional): Random seed. Defaults to 42.
        init (str, optional): Initialization method. Defaults to "k-means++".

    Returns:
        model: Trained model
        train_labels: Labels for training data
        valid_labels: Labels for validation data
  ''' 
  cluster = KMeans(n_clusters=number_groups, random_state = seed, n_init="auto",
                   init = init)
  model=cluster.fit(x_train)
  train_labels = model.labels_

  valid_labels = model.predict(x_valid)

  return model, train_labels, valid_labels

def predict_with_model(smiles_list: list, model, featurizer = "rdkit", scaler = None, pca = None):
  '''
    receive a list of SMILES, a model and a featurizer name, and possibly a scaler and pca model.
    applies transformations and then makes predictions. 
    
    Args: 
        smiles_list: smiles to predict
        model: fitted ML model
        featurizer: the name of the featurizer used
        scaler (optional): a fitted scaler
        pca (optional): a fitted pca
    Returns:
        predictions
  '''
  mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]

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
  
  if scaler != None and pca != None:
    scaled_f = scaler.transform(f)
    final_f = pca.transform(scaled_f)
  elif scaler != None and pca == None:
    final_f = scaler.transform(f)
  elif scaler == None and pca != None:
    final_f = pca.transform(f)
  elif scaler == None and pca == None:
    final_f = f
    
  predictions = model.predict(final_f)
  
  return predictions

def kmeans_loss(x_train: np.array, x_valid: np.array, y_train: np.array, y_valid: np.array, smiles_valid: list, 
                train_preds: np.array, valid_preds: np.array, number_groups = 10, seed= 42, init = "k-means++"):
  '''
  Find k-means clusters for the given data.

    Args:
        x_train (np.array): Training data.
        x_valid (np.array): Validation data.
        y_train (np.array): Training labels.
        y_valid (np.array): Validation labels.
        smiles_valid (list): Validation SMILES.
        train_preds (np.array): Training predictions.
        valid_preds (np.array): Validation predictions.
        number_groups (int, optional): Number of clusters. Defaults to 10.
        seed (int, optional): Random seed. Defaults to 42.
        init (str, optional): Initialization method. Defaults to "k-means++".

    Returns:
        df_val_list (list): List of dataframes for each validation cluster.
        pics (list): List of images for each cluster.
  ''' 
  cluster = KMeans(n_clusters=number_groups, random_state = seed, n_init="auto",
                   init = init)
  model=cluster.fit(x_train)
  train_labels = model.labels_
  train_loss = np.abs(train_preds - y_train)

  valid_labels = model.predict(x_valid)
  valid_loss = np.abs(valid_preds - y_valid)

  train_dict = {"Train truth": y_train, "Train preds": train_preds,
                "Train loss": train_loss, "Train cluster": train_labels}
  train_df = pd.DataFrame(train_dict)
  train_df.to_csv("train_df.csv")
  print(f"CSV for training set saved!")

  valid_dict = {"Valid SMILES": smiles_valid, "Valid truth": y_valid, "Valid preds": valid_preds,
                "Valid loss": valid_loss, "Valid cluster": valid_labels}
  valid_df = pd.DataFrame(valid_dict)
  valid_df.to_csv("valid_df.csv")
  print(f"CSV for valiation set saved!")

  group_names = []
  df_val_list = []
  df_train_list = []
  for i in range(number_groups):
      group_names.append(f"df_group{i}")

  for i,name in enumerate(group_names):
      name = valid_df[valid_df["Valid cluster"] == i]
      df_val_list.append(name)
  
  for i,name in enumerate(group_names):
      name = train_df[train_df["Train cluster"] == i]
      df_train_list.append(name)

  mean_loss = []
  std_loss = []
  max_loss = []
  number_molecules = []
  mean_train_loss = []
  std_train_loss = []
  max_train_loss = []
  number_train_molecules = []
  mols_total = []
  smiles_total = []
  loss_total = []

  for group, val_dataframe, train_dataframe in zip(group_names,df_val_list,df_train_list):
      df_temp = val_dataframe
      mean_loss.append(df_temp["Valid loss"].mean())
      std_loss.append(df_temp["Valid loss"].std())
      max_loss.append(df_temp["Valid loss"].max())
      number_molecules.append(len(df_temp))

      df_train_temp = train_dataframe
      mean_train_loss.append(df_train_temp["Train loss"].mean())
      std_train_loss.append(df_train_temp["Train loss"].std())
      max_train_loss.append(df_train_temp["Train loss"].max())
      number_train_molecules.append(len(df_train_temp))
      
      df_temp.sort_values(by=["Valid loss"],inplace=True, ascending=False)
     
      mols = [Chem.MolFromSmiles(smile) for smile in df_temp["Valid SMILES"]]
      mols_total.append(mols)
      smiles = [smile for smile in df_temp["Valid SMILES"]]
      losses = [loss for loss in df_temp["Valid loss"]]
      loss_total.append(losses)
      smiles_total.append(smiles)
  
  print(f"Type     Group Number   Number Molecules   Mean Loss   Std Loss   Max Loss")
  for i in range(number_groups):
      print(f"Train    {i:12} {number_train_molecules[i]:18} {mean_train_loss[i]:11.2f} {std_train_loss[i]:10.2f} {max_train_loss[i]:10.2f}")
      print(f"Valid    {i:12} {number_molecules[i]:18} {mean_loss[i]:11.2f} {std_loss[i]:10.2f} {max_loss[i]:10.2f}")
      delta = abs(mean_loss[i] - mean_train_loss[i])
      blank = ""
      print("-"*74)
      print(f"{blank:42}\u0394    {delta:5.2f} ")
      print("")

  pics = []
  for i in range(number_groups):
      legends = [f"SMILES: {smile}, Loss: {loss:.2f}" for loss, smile in zip(loss_total[i], smiles_total[i])]
      try:
        img = MolsToGridImage(mols_total[i], legends = legends, molsPerRow=2, subImgSize=(300,300))
        pics.append(img)
      except:
        print(f"Error with group {i}")
        pics.append(None)
        
  return df_val_list, pics
 
 def find_fragment(mol_list: list, threshold: float):
  '''
    Finds the most common fragment in a list of molecules, based on a particular threshold
    of molecules that must have the fragment, i.e., the fragment only has to appear in 
    threshold % of the list of molecules.

      Args:
        mol_list: mol objects to analyze
        threshold: minimum threshold for the analysis
      Returns:
        mcs.querymol: the found fragment
        frac_with_frag: fraction of molecules that have the fragment
        matching_mols: list of molecules that have the fragment
        matching_smiles: list of SMILES for the molecules that have the fragment
  '''
  params = Chem.rdFMCS.MCSParameters()
  params.Threshold = threshold
  params.BondCompareParameters.CompleteRingsOnly=True
  params.AtomCompareParameters.CompleteRingsOnly=True
  
  mcs = FindMCS(mol_list,params)
  
  mcfrag = mcs.queryMol
  mcsmiles = Chem.MolToSmiles(mcfrag)
  
  matching_mols = []
  matching_smiles = []
  no_match = 0
  for m in mol_list:
    match_found = m.HasSubstructMatch(mcs.queryMol)
    if match_found:
      matching_mols.append(m)
      matching_smiles.append(Chem.MolToSmiles(m))
    else:
      no_match += 1

  print(f"Could not match {no_match} molecules to the fragment")
  print(f"Found {len(mol_list) - no_match} molecules containing the fragment")
  frac_with_frag = (len(mol_list) - no_match) / len(mol_list)
  print(f"Fraction of molecules with fragment: {frac_with_frag:.3f}")
  print(f"Fragment SMILES: {mcsmiles}")
  print("\n\n")

  return mcs.queryMol, frac_with_frag, matching_mols, matching_smiles

def get_common_fragment(smiles_list: list, min_threshold: float, which_result = 0):
  '''
    Finds the most common fragment in a list of smiles, based on a particular threshold
    of molecules that must have the fragment, i.e., the fragment only has to appear in 
    threshold % of the list of molecules.

      Args:
        smiles_list: SMILES to analyze
        min_threshold: minimum threshold to start the analysis
        which_result: 0 --> longest fragement, 1 --> second longest fragment, etc
      Returns:
        img: image of the molecules that match the fragment
        mcsmiles: SMILES for the fragment
        frac_with_frag: fraction of molecules that have the fragment
  '''

  mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
  
  threshold = min_threshold
  fracs = []
  frags = []
  total_matching_mols = []
  total_matching_smiles = []

  while threshold <= 1:
    print(f"Threshold: {threshold:.3f}")
    mcfrag, frac_with_frag, matching_mols, matching_smiles = find_fragment(mol_list, threshold)
    fracs.append(frac_with_frag)
    frags.append(mcfrag)
    total_matching_mols.append(matching_mols)
    total_matching_smiles.append(matching_smiles)
    threshold += 0.05
    if frac_with_frag >= 0.9:
      break

  lengths = []
  for frag in frags:
    smile = Chem.MolToSmiles(frag)
    lengths.append(len(smile))
  
  original_lengths = lengths.copy()
  lengths.sort(reverse=True)
  desired_length = lengths[which_result]
  desired_idx = original_lengths.index(desired_length) 

  mcsmiles = Chem.MolToSmiles(frags[desired_idx])
  mcfrag = frags[desired_idx]
  matching_mols = total_matching_mols[desired_idx]
  matching_smiles = total_matching_smiles[desired_idx]
  frac_with_frag = fracs[desired_idx]

  subst = mcfrag
  AllChem.Compute2DCoords(mcfrag)

  [AllChem.GenerateDepictionMatching2DStructure(m,mcfrag) for m in matching_mols]

  img = MolsToGridImage(matching_mols,
                  highlightAtomLists=[m.GetSubstructMatch(subst) for m in matching_mols],
                  legends = matching_smiles, molsPerRow=5,useSVG=False)
  
  print("Fragment analysis complete.")

  return img, mcsmiles, frac_with_frag