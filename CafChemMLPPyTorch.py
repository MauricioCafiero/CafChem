import numpy as np
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

class MLP_Model(nn.Module):
  '''
  Multilayer Perceptron Model using PyTorch. Activation is ReLU6
    Args:
      neurons: number of neurons in each hidden layer
      input_dims: number of input dimensions
      num_hidden_layers: number of hidden layers
  '''
  def __init__(self, neurons: int, input_dims: int, num_hidden_layers: int):
    super(MLP_Model, self).__init__()
    self.neurons = neurons
    self.input_dims = input_dims
    self.num_hidden_layers = num_hidden_layers
    self.batchnorm = nn.BatchNorm1d(self.input_dims)
    self.linear_input = nn.Sequential(
        nn.Linear(self.input_dims, self.neurons),
        nn.ReLU6())
    self.linear_relu6 = nn.Sequential(
        nn.Linear(self.neurons, self.neurons),
        nn.ReLU6())
    self.linear_output = nn.Linear(self.neurons, 1)

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
    output = self.linear_output(x)
    return output

def train(dataloader, model, loss_fn, optimizer):
  '''
    Trains the model for one epoch.

    Args:
      dataloader: dataloader for the training data
      model: model to train
      loss_fn: loss function to use
      optimizer: optimizer to use
    Returns:
      model: trained model
  '''
  size = len(dataloader.dataset)
  model.train()

  total_loss = 0

  for batch, (X, y) in enumerate(dataloader):
    optimizer.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, y.view(-1,1))
    total_loss += loss

    loss.backward()
    optimizer.step()

    if batch % 2 == 0:
      current = batch * len(X)
      avg_loss = total_loss / (batch + 1)
      print(f"Batch: {batch}, Loss: {avg_loss:.7f} [{current:>5d}/{size:>5d}]")

  return model

def evaluate_training(X_train, y_train, X_test, y_test, model):
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

def create_data_loader(X_train, y_train, X_test, y_test, batch_size: int, shuffle = True):
  '''
    Creates a data loader for the training and test data.

    Args:
      X_train: training data
      y_train: training truths
      X_test: test data
      y_test: test truths
      batch_size: batch size for the data loader
      shuffle: whether to shuffle the training data
    Returns:
      train_dataset: training dataset
      test_dataset: test dataset
      train_loader: training data loader
      test_loader: test data loader
  '''
  y_train = list(y_train)
  y_test = list(y_test)
  X_train = torch.tensor(X_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)
  X_test = torch.tensor(X_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  train_dataset = TensorDataset(X_train, y_train)
  test_dataset = TensorDataset(X_test, y_test)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_dataset, test_dataset, train_loader, test_loader