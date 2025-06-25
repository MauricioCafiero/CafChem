import torch
from pathlib import Path
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from chemprop import data, models, featurizers, nn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from chemprop.models.model import MPNN

def read_data(filename: str, smiles_column: str, target_columns: list[str], transform_flag: bool):
  '''
  Reads data from a CSV file and returns SMILES strings and target values.

    Args:
      filename (str): The path to the CSV file.
      smiles_column (str): The name of the column containing SMILES strings.
      target_columns (list[str]): A list of names of the columns containing target values.
      transform_flag (bool): A flag indicating whether to apply a log transformation to the target values.

    Returns:
      smis: A list of SMILES strings.
      ys: An array of target values.
  '''
  df_input = pd.read_csv(filename)
  smis = df_input.loc[:, smiles_column].values
  ys_raw = df_input.loc[:, target_columns].values

  if transform_flag:
    ys = np.log10(ys_raw)
  else:
    ys = ys_raw

  return smis, ys

def make_model(smis: list, ys: np.array, splits_tuple: tuple):
  '''
    defines the model to use for finetuning; creates three datasets: training, 
    validation, and testing, and their associated dataloaders.

    Args:
      smis (list): A list of SMILES strings.
      ys (np.array): An array of target values.
      splits_tuple: the fractions for training, validation and testing
    Returns:
      mpmm: the model
      train_loader: the training dataloader
      val_loader: the validation dataloader
      test_loader: the testing dataloader
      train_dset: the training dataset
      val_dset: the validation dataset
      test_dset: the testing dataset
  '''
  featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
  agg = nn.MeanAggregation()
  chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
  mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
  mp.load_state_dict(chemeleon_mp['state_dict'])

  chemprop_dir = Path.cwd().parent
  num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
  all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
  mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits
  train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", splits_tuple)  # unpack the tuple into three separate lists
  train_data, val_data, test_data = data.split_data_by_indices(
      all_data, train_indices, val_indices, test_indices
  )
  train_dset = data.MoleculeDataset(train_data[0], featurizer)

  scaler = train_dset.normalize_targets()
  val_dset = data.MoleculeDataset(val_data[0], featurizer)
  val_dset.normalize_targets(scaler)
  test_dset = data.MoleculeDataset(test_data[0], featurizer)
  train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
  val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
  test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)
  output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
  ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=mp.output_dim)
  metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
  mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)
  
  return mpnn, train_loader, val_loader, test_loader, train_dset, val_dset, test_dset

def transformed_lists(test_preds: list, test_dset):
  '''
    Produces four lists: original y values and predicted y values, each
    untransformed (raw) and transformed (log10). Calculates the R2 score for each type.

      Args:
        test_preds (list): A list of predicted values.
        test_dset: The test dataset.

      Returns:
        trans_ys: A list of transformed y values.
        untrans_ys: A list of untransformed y values.
        trans_preds: A list of transformed predicted values.
        untrans_preds: A list of untransformed predicted values.
  '''
  trans_ys = []
  untrans_ys = []
  for i in range(len(test_dset)):
    trans_ys.append(test_dset[i].y[0].item())
    untrans_ys.append(10 ** test_dset[i].y[0].item())

  trans_preds = []
  for i in range(len(test_preds)):
    trans_preds.append(np.log10(test_preds[i]).item())

  untrans_preds = test_preds

  untrans_r2 = r2_score(untrans_ys,untrans_preds)
  trans_r2 = r2_score(trans_ys,trans_preds)
  print(f"regular scale R2 = {untrans_r2:.2f}, log scale R2 = {trans_r2:.2f}")

  return trans_ys, untrans_ys, trans_preds, untrans_preds

def values_differences(ys: list, preds: list):
  '''
  prints the ground truth and predicted values, as well as the difference between them.
  reports the average difference.

    Args:
      ys (list): A list of ground truth values.
      preds (list): A list of predicted values.
    Returns:
      None.
  '''
  total_diff = 0
  for y, p in zip(ys, preds):
    diff = abs(y-p)
    print(f"y = {y:10.2f}, pred = {p:10.2f}, difference: {diff:>.2f}")
    total_diff += diff
  total_diff /= len(trans_ys)
  print(f"average difference: {total_diff}")

def plot_test(ys: list, preds: list):
  '''
    Plots the ground truth values against the predicted values.

    Args:
      ys (list): A list of ground truth values.
      preds (list): A list of predicted values.
    Returns:
      None.
  '''
  plt.scatter(ys, preds)
  plt.xlabel("Experimental")
  plt.ylabel("Predicted")
  plt.show()

def save_model(model, path: str):
  '''
  Saves a chemeleon model to a specified path.

    Args:
      model: The chemeleon model to be saved.
      path (str): The path where the model will be saved.
    Returns:
      None.
  '''
  model_dict = {"hyper_parameters": model.hparams, "state_dict": model.state_dict()}
  torch.save(model_dict, path)
  print(f"{model} model saved to {path}")

def load_model(path: str):
  '''
  Loads a chemeleon model from a specified path.

    Args:
      path (str): The path from which the model will be loaded.
    Returns:
      None.
  '''
  loaded_model = MPNN.load_from_file(path)
  print(f"{loaded_model} model loaded from {path}")
  return loaded_model
