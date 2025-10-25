import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from rdkit import Chem
import deepchem as dc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

xgb.set_config(verbosity=2)

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

def scale_pca_split(f: np.array, y_raw: list, Xa: list, use_scaler = True,
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

  if type(y_raw[0]) != int:
    unique_classes = set(y_raw)
    class_dict = {}
    for i,y_class in enumerate(unique_classes):
        class_dict[str(y_class)] = i
        
    y = []
    for val in y_raw:
        y.append(class_dict[val])
    print('target converted to ints')
  else:
    y = y_raw

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
  
  #separate target values and SMILES strings and change y values to ints    
  y_train = ys_train[:,0].astype(int)
  y_valid = ys_valid[:,0].astype(int)   
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

class Boost_methods():
  '''
    class for performing gradient boosting regression (classification); all options in constructor.
  '''
  def __init__(self, model_type = "XGBoost", classifier_flag = False, n_estimators = 500, learning_rate = 0.05, max_depth = 50,
                num_leaves = 31, feature_fraction = 0.8, min_data_in_leaf = 20, iterations = 100, depth = 6):
    '''
    Constructor
      Args:
        model_type: type of model to use ('XGBoost', 'LightGBM', 'CatBoost')
        classifier_flag: boolean for classification (optional)
        n_estimators: number of estimators
        learning_rate: learning rate
        max_depth: max depth
        num_leaves: number of leaves in the tree (LightGBM only)
        feature_fraction: feature fraction (LightGBM only)
        min_data_in_leaf: minimum data in leaf (LightGBM only)
        iterations: number of iterations (CatBoost only)
        depth: depth of the tree (CatBoost only)
      Returns:
        None
    '''
    self.model_type = model_type
    self.classifier_flag = classifier_flag
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.num_leaves = num_leaves
    self.feature_fraction = feature_fraction
    self.min_data_in_leaf = min_data_in_leaf
    self.iterations = iterations
    self.depth = depth

    print("Gradient Boosting Regression (classification) class initialized.")
  
  def fit(self, x_train: np.array, x_valid: np.array, y_train: np.array, y_valid: np.array):
    '''
      Fits a gradient boosting regression model and calculates the training and validation scores.

      Args:
        x_train: training feature matrix
        x_valid: validation feature matrix
        y_train: training target list
        y_valid: validation target list
      Returns:
        model: fitted model
    '''
    if self.model_type == "XGBoost":
      if self.classifier_flag:
        model = XGBClassifier(n_estimators = self.n_estimators, learning_rate = self.learning_rate, max_depth = self.max_depth)
        modelname = "XGBoost Classifier"
      else:
        model = XGBRegressor(n_estimators = self.n_estimators, learning_rate = self.learning_rate, max_depth = self.max_depth)
        modelname = "XGBoost Regressor"
    elif self.model_type == "LightGBM":
      if self.classifier_flag:
        model = LGBMClassifier(metric='logloss', max_depth = self.max_depth, verbose = -1, num_leaves = self.num_leaves, 
        feature_fraction = self.feature_fraction, min_data_in_leaf = self.min_data_in_leaf)
        modelname = "LightGBM Classifier"
      else:
        model = LGBMRegressor(metric='rmse', max_depth = self.max_depth, verbose = -1, num_leaves = self.num_leaves, 
        feature_fraction = self.feature_fraction, min_data_in_leaf = self.min_data_in_leaf)
        modelname = "LightGBM Regressor"
    elif self.model_type == "CatBoost":
      if self.classifier_flag:
        model = CatBoostClassifier(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth, 
        verbose=False)
        modelname = "CatBoost Classifier"
      else:
        model = CatBoostRegressor(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth, 
        verbose=False)
        modelname = "CatBoost Regressor"
    else:
      raise ValueError("Model type must be 'XGBoost', 'LightGBM', or 'CatBoost'")
  
    print("model selected: ",modelname)
    tic = time.perf_counter()
    model.fit(x_train, y_train)
    toc = time.perf_counter()
    fittime = (toc-tic)/60

    print(f"fit model in: {fittime} minutes")

    #evaluate models

    train_score = model.score(x_train,y_train)
    print(f"score for training set: {train_score:.3f}")

    valid_score = model.score(x_valid,y_valid)
    print(f"score for validation set: {valid_score:.3f}")

    print("="*70)

    return model, train_score, valid_score

  def grid_search(self, model_type: str, X: list, y: list, param_grid: list, cv = 5):
    '''
      Performs a grid search on the model.
      Args:
        model_type: type of model to use ('XGBoost', 'LightGBM', 'CatBoost')
        X: feature matrix
        y: target list
        param_grid: parameter grid
        cv: number of cross-validation folds
      Returns:
        best_model: best model
      '''
    if self.model_type == "XGBoost":
      model = XGBRegressor()
      modelname = "XGBoost"
    elif self.model_type == "LightGBM":
      model = LGBMRegressor(metric='rmse', max_depth = -1, verbose = -1)
      modelname = "LightGBM"
    elif self.model_type == "CatBoost":
      model = CatBoostRegressor(verbose=False)
      modelname = "CatBoost"
    else:
      raise ValueError("Model type must be 'XGBoost', 'LightGBM', or 'CatBoost'")

    print("Performing grid search on: ",modelname)

    search_f = GridSearchCV(estimator = model,
                       param_grid = param_grid,
                       n_jobs=-1,
                       cv=5)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    search_f.fit(X_scaled, y)
    print(f" Best parameters: {search_f.best_params_} and best score: {search_f.best_score_:.3f}.")
      
    return search_f.best_params_, search_f.best_score_

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

  print(f"Percentage of correct predictions: {(100*num_correct/len(predictions)):.3f}")
