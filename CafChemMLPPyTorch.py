import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import mordred
import deepchem as dc

def featurize(smiles_list: list, y: list,
              ions_to_clean = ["[Na+].", ".[Na+]"], featurizer = "rdkit", classifier_flag = False):
  '''
  featurize a list of SMILES using RDKit or Mordred, clean counterions, 
  and remove NANs. treats target list as well so values returned match

  Args:
    smiles_list: list of SMILES
    target_list: list of target values
    ions_to_clean: list of ions to remove from SMILES
    featurizer: "rdkit" or "mordred"
    classifier_flag: boolean to use classifier
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
  
  if classifier_flag == True:
    num_cats = len(set(y))
    y_cats = np.zeros((len(y),num_cats))
    for i, val in enumerate(y):
      y_cats[i,int(val)] = 1

    return f, y_cats, Xa
  
  else:
    return f, y, Xa

class MLP_Model(nn.Module):
  '''
  Multilayer Perceptron Model using PyTorch. Activation is ReLU6
    Args:
      neurons: number of neurons in each hidden layer
      input_dims: number of input dimensions
      num_hidden_layers: number of hidden layers
      classifier_flag: Boolean to perform classification
  '''
  def __init__(self, neurons: int, input_dims: int, num_hidden_layers: int, 
               classifier_flag=False, num_classes = 1):
    super(MLP_Model, self).__init__()
    self.neurons = neurons
    self.input_dims = input_dims
    self.num_hidden_layers = num_hidden_layers
    self.classifier_flag = classifier_flag
    self.num_classes = num_classes
    self.batchnorm = nn.BatchNorm1d(self.input_dims)
    self.linear_input = nn.Sequential(
        nn.Linear(self.input_dims, self.neurons),
        nn.ReLU6())
    self.linear_relu6 = nn.Sequential(
        nn.Linear(self.neurons, self.neurons),
        nn.ReLU6())
    self.linear_output = nn.Linear(self.neurons, 1)
    self.linear_class_out = nn.Linear(self.neurons, self.num_classes)
    self.classifier_output = nn.LogSoftmax()
    
    f = open("MLP_model_params.txt","w")
    f.write(f"neurons: {self.neurons}\n")
    f.write(f"input_dims: {self.input_dims}\n")
    f.write(f"num_hidden_layers: {self.num_hidden_layers}\n")
    f.write(f"classifier_flag: {self.classifier_flag}\n")
    f.write(f"num_classes: {self.num_classes}")
    f.close()

  def forward(self, x):
    '''
      Passes the input through a batch normalization layer, an input layer,
      a number of hidden layers, and an output layer.

        Args:
          x: input tensor
        Returns:
          output: output tensor
    '''
    x = self.batchnorm(x)
    x = self.linear_input(x)
    for i in range(self.num_hidden_layers):
      x = self.linear_relu6(x)
    
    if self.classifier_flag == False:
      output = self.linear_output(x)
    else:
      x = self.linear_class_out(x)
      output = self.classifier_output
      
    return output

def train(dataloader, model, loss_fn, optimizer, classifier_flag=False, num_classes = 1):
  '''
    Trains the model for one epoch.

    Args:
      dataloader: dataloader for the training data
      model: model to train
      loss_fn: loss function to use
      optimizer: optimizer to use
      classifier_flag: Boolean to perform classification
    Returns:
      model: trained model
  '''
  if classifier_flag == False:
    num_classes = 1
  
  size = len(dataloader.dataset)
  model.train()

  total_loss = 0

  for batch, (X, y) in enumerate(dataloader):
    optimizer.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, y.view(-1,num_classes))
    total_loss += loss

    loss.backward()
    optimizer.step()

    if batch % 2 == 0:
      current = batch * len(X)
      avg_loss = total_loss / (batch + 1)
      print(f"Batch: {batch}, Loss: {avg_loss:.7f} [{current:>5d}/{size:>5d}]")

  return model

def evaluate_regression(X_train, y_train, X_test, y_test, model):
  '''
    Evaluates the model on the training and test data.

    Args:
      X_train: training data
      y_train: training truths
      X_test: test data
      y_test: test truths
      model: model to evaluate
    Returns:
      train_r2: R^2 score of the training data
      test_r2: R^2 score of the test data
      Plots the training and test data against the model's predictions.
  '''
  X_train = torch.tensor(X_train, dtype=torch.float32)
  X_test = torch.tensor(X_test, dtype=torch.float32)

  model.eval()
  train_pred = model(X_train)
  test_pred = model(X_test)
  y_total = np.concatenate((y_train, y_test), axis=0)

  train_r2 = r2_score(y_train, train_pred.detach().numpy())
  test_r2 = r2_score(y_test, test_pred.detach().numpy())
  print(f"Train R2 Score: {train_r2}")
  print(f"Test R2 Score: {test_r2}")

  plt.scatter(y_train,train_pred.detach().numpy(),color="blue",label="ML-train")
  plt.scatter(y_test,test_pred.detach().numpy(),color="green",label="ML-valid")
  plt.plot(y_total,y_total,color="red",label="Best Fit")
  plt.legend()
  plt.xlabel("known")
  plt.ylabel("predicted")
  plt.show

  return train_r2, test_r2

def predict_single_value(smiles_to_predict: str, model, featurizer = "rdkit", 
                         scaler = None, truth = None):
  '''
    Predicts the value of a single SMILES string. Featurizes the SMILES, applies
    the scaler if provided, and passes the feature vector through the model.

    Args:
      smiles_to_predict: SMILES string to predict
      model: model to use for prediction
      featurizer: "fingerprints", "rdkit", or "mordred"
      scaler: scaler to use for scaling the feature vector
      truth (optional): truth value to compare the prediction to
    Returns:
      prediction: predicted value
  '''
  print(f"Predicting value for {smiles_to_predict}")
  mol = Chem.MolFromSmiles(smiles_to_predict)

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

  f = featurizer.featurize([mol])

  if scaler is not None:
    temp_vec = scaler.transform(f)
  
  #temp_vec = np.array(temp_vec).reshape(1,-1)
  temp_tensor = torch.tensor(temp_vec, dtype=torch.float32)

  model.eval()

  with torch.no_grad():
    prediction = model(temp_tensor).item()

  if truth is not None:
    plot_text = f"Prediction: {prediction}, Truth: {truth}"
  else:
    plot_text = f"Prediction: {prediction}"

  print(plot_text)
  return prediction

class prep_data():
  '''
  Data class to prepare raw data for model
  '''
  def __init__(self, batch_size: int, shuffle = True):
    '''
        Sets up data prep parameters.
        
        Args:
            batch_size: batch size for training / data loader
            shuffle: Boolean to shuffle training set
    '''
    self.batch_size = batch_size
    self.shuffle = shuffle
    
    print("prep data class initialized!")
  
  def scale_split(self, X, y, test_size=0.2, random_state=32):
    '''
        apply standard scaler and split the dataset
        
        Args:
            X: features array
            y: target array
            test_size = fraction to use for test set
            random_state = random number initializer
        Returns:
            X_train: training data
            y_train: training truths
            X_test: test data
            y_test: test truths     
    '''
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, 
                                       test_size=test_size, random_state=random_state)
    
    return self.X_train, self.X_test, self.y_train, self.y_test, scaler
    
  def create_data_loader(self):
    '''
      Creates a data loader for the training and test data.

      Args: 
        None
      Returns:
        train_dataset: training dataset
        test_dataset: test dataset
        train_loader: training data loader
        test_loader: test data loader
    '''
    self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
    self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
    self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
    self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
    self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
    self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    train_dataset = TensorDataset(self.X_train, self.y_train)
    test_dataset = TensorDataset(self.X_test, self.y_test)

    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader

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
  target_list = []
  for target in df[target_name]:
    for i in range(len(range_cutoffs)-1):
      if target <= range_cutoffs[i+1]:
        labels_list.append(class_labels[i])
        target_list.append(i)
        break

  df["class labels"] = labels_list
  df["target"] = target_list

  columns = df.columns
  for column in columns:
    if "Smiles" in column or "SMILES" in column or "smiles" in column:
      smiles_name = column
      break
    
  df[smiles_name] = df[smiles_name].apply(clean_ions)

  return df, class_labels
 
def load_model():
   '''
    Loads a PyTorch model from a text file containing the model parameters
    and a .pt file containing the model weights.

    Args:
      None
    Returns:  
      model: The loaded PyTorch model.
   '''
   f = open("MLP_model_params.txt","r")
   lines = f.readlines()
   f.close()
   
   neurons = lines[0].split()[1]
   input_dims = lines[1].split()[1]
   num_hidden_layers = lines[2].split()[1]
   classifier_raw = lines[3].split(":")[1].strip().replace("\n","")
   if classifier_raw == "True":
     classifier_flag = True
   elif classifier_raw == "False":
     classifier_flag = False
   else:
     raise ValueError("classifier_flag must be True or False")
   num_classes = lines[4].split()[1]

   model = MLP_Model(neurons=int(neurons), input_dims=int(input_dims), num_hidden_layers=int(num_hidden_layers),
                     classifier_flag = classifier_flag, num_classes = int(num_classes))
   model.load_state_dict(torch.load("saved_model.pt",weights_only=True))

   return model