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
  
  y_train = ys_train[:,0]
  y_valid = ys_valid[:,0]
  smiles_train = ys_train[:,1]
  smiles_valid = ys_valid[:,1]

  print("Pre-processing done.")

  return X_train, X_valid, y_train, y_valid, smiles_train, smiles_valid, pca, scaler

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

class LogReg_methods():
  '''
    class for performing logistic regression (classification); all options in constructor.

      Args:
        penalty: "l1", "l2", or "elasticnet"
        C: regularization parameter
        l1_ratio: ratio of L1 to L2 regularization (only needed for elasticnet)
      Returns:
        None
  '''
  def __init__(self, penalty = "l2", C = 1.0, l1_ratio = 0.5):
    self.penalty = penalty
    self.C = C
    self.l1_ratio = l1_ratio

    print("Logistic Regression (classification) class initialized.")
  
  def fit(self, x_train: np.array, x_valid: np.array, y_train: np.array, y_valid: np.array):
    '''
      Fits a logistic regression model and calculates the training and validation scores.

      Args:
        x_train: training feature matrix
        x_valid: validation feature matrix
        y_train: training target list
        y_valid: validation target list
      Returns:
        model: fitted model
    '''
    model = LogisticRegression(penalty = self.penalty, C = self.C, max_iter = 10000, 
                               l1_ratio=self.l1_ratio, solver = 'saga')
    modelname = "LogisticRegression"
  
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

class tree_classification():
  '''
    class for performing tree classification; all options in constructor.

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

    print("Tree classification class initialized.")

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
      model = DecisionTreeClassifier(max_depth=self.max_depth)
      modelname = "DecisionTree"
    elif self.method == "forest":
      model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_features = self.max_features)
      modelname = "RandomForest"
    elif self.method == "gradient":
      model = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate = self.learning_rate,
                                        max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf, 
                                        max_features=self.max_features,
                                        loss = "huber")
      modelname = "GradientBoostingClassifier"
    

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

class ridge_svm_methods():
  '''
    class for performing ridge or support vector machines classification; all options in constructor.

      Args:
        method:"ridge" or "svm"
        degree: degree of polynomial for SVM
        C: regularization parameter for SVM
        epsilon: epsilon parameter for SVM
        coef0: coef0 parameter for SVM
      Returns:
        None
  '''
  def __init__(self, method = "svm", degree = 3, C = 2.0, coef0 = 2.0,
               alpha = 1000, max_iter = 10000):
    self.method = method
    self.degree = degree
    self.C = C
    self.coef0 = coef0
    self.alpha = alpha
    self.max_iter = max_iter
    
    print("Ridge SVM class initialized.")

  def fit(self, x_train: np.array, x_valid: np.array, y_train: np.array, y_valid: np.array):
    '''
      Fits a ridge or SVM classifier and calculates the training and validation scores.

      Args:
        x_train: training feature matrix
        x_valid: validation feature matrix
        y_train: training target list
        y_valid: validation target list
      Returns:
        model: fitted model
    '''
    if self.method == "ridge":
      model = RidgeClassifier(alpha=self.alpha)
      modelname = "Ridge"
    elif self.method == "svm":
      model = SVC(kernel="poly",degree=self.degree, coef0=self.coef0, C=self.C)
      modelname = "Support Vector Machines"

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
    class for performing multi-layer perceptron classification; all options in constructor.

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
  def __init__(self,solver = "adam", seed = 42, hidden_layer_sizes = [20,20], 
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
      Fits a multi-layer perceptron classification model and calculates the training and validation scores.

      Args:
        x_train: training feature matrix
        x_valid: validation feature matrix
        y_train: training target list
        y_valid: validation target list
      Returns:
        model: fitted model
    '''
    model = MLPClassifier(solver=self.solver,random_state=self.seed, 
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

class evaluate():
  '''
    evaluates the testing set with a confusion matrix.
  
  '''
  def __init__(self, model, labels: list, test_values, truths):
    '''
        Constructor
            Args:
                model: fitted model
                labels: class labels
                test_values: testing values, to predict
                truths: ground truth
            returns:
                None
    '''
    self.model = model
    self.labels = labels
    self.test_values = test_values 
    self.truths = truths
 
  def plot_confusion_matrix(self, y_preds, y_true, labels):
    '''
        actually plots the confusion matirx.
        
        Args:
            y_preds: predicted values
            y_true: ground truth
            labels: class labels
        returns:
                None; plots confusion matrix.
    '''
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

  def confusion(self):
    '''
        uses the trainer to make predictions on the dataset and plots the confusion matrix
    '''
    preds_output = self.model.predict(self.test_values)
    y_preds = preds_output
    y_true = self.truths
    self.plot_confusion_matrix(y_preds, y_true, labels=self.labels)

def analyze_predictions(predictions: list, truths: list):
  '''
    Takes a list of predictions and comares it against a list of truths. Shows
    which are correct/incorrect and prints the % correct.

    Args:
      predictions: list of predictions
      truths: list of truths
    Returns:
      None
  '''
  num_correct = 0
  for pred, exp in zip(predictions, truths):
    parts = pred.split()
    ll = parts[0]
    ul = parts[2]
    if exp >= float(ll) and exp <= float(ul):
      print(f"Predicted: {pred:20}, Truth: {exp:10.2f}, prediction is correct")
      num_correct += 1
    else:
      print(f"Predicted: {pred:20}, Truth: {exp:10.2f}, prediction is incorrect")

  print(f"Percentage of correct predictions: {(100*num_correct/len(preds)):.3f}")