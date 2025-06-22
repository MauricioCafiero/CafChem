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

def remove_outliers(f: np.array, y: list, Xa: list):
  '''
    Identifies outliers using Elliptic Envelope and removes them from the
    feature matrix, the target list, and the SMILES list.

    Args:
      f: feature matrix
      y: target list
      Xa: SMILES list
    Returns:
      f: feature matrix without outliers
      y: target list without outliers
      Xa: SMILES list without outliers
  '''
  outlier_detector = EllipticEnvelope(contamination=0.01)
  outlier_detector.fit(f)
  outlier_array = outlier_detector.predict(f)
  indicies = np.where(outlier_array == -1)
  print("===========================================================")
  print(f"Outliers found in the following locations: {indicies}")

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

  elif use_scaler == False and use_pca == True:

    pca = PCA(n_components=pca_size)
    pca.fit(f)
    f_final = pca.transform(f)

  seed = 102

  X_train, X_valid, ys_train, ys_valid = train_test_split(f_final,y_smiles,train_size=splits, 
                                                              random_state=seed, shuffle=True)
  
  y_train = ys_train[:,0].astype(float)
  y_valid = ys_valid[:,0].astype(float)
  smiles_train = ys_train[:,1]
  smiles_valid = ys_valid[:,1]

  print("Pre-processing done.")

  return X_train, X_valid, y_train, y_valid, smiles_train, smiles_valid

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
      model = SVR(kernel="poly",degree=self.degree, epsilon=self.epsilon,coef0=self.coef0)
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