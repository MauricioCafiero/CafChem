from pathlib import Path
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from chemprop import data, featurizers, models, nn
from chemprop.models.model import MPNN

chemprop_dir = Path.cwd().parent

class chemprop_data():
  '''
    class for reading in data, splitting, featurizing and creating dataloaders
  '''
  def __init__(self, num_workers=0, split_type = "random", train_split = 0.8,
               valid_split = 0.1, test_split = 0.1):
    '''
     initializes the data class

      Args:
        num_workers: number of workers for data loading
        split_type: type of split to use
        train_split: percentage of data to use for training
        valid_split: percentage of data to use for validation
        test_split: percentage of data to use for testing
    '''
    self.num_workers = num_workers
    self.split_type = split_type
    self.train_split = train_split
    self.valid_split = valid_split
    self.test_split = test_split

    print("Class chemprop_data initialized")

  def read_prep_data(self, df_input, smiles_column: str, target_columns: list, log_flag = False):
    '''
      reads in a dataframe, a SMILES column name, and target column names. Also reads a flag to apply
      a log to the target values.

        Args:
          df_input: input dataframe
          smiles_column: the name of the SMILES column
          target_columns: the names of the target columns
          log_flag: a flag to apply a log to the target values


          Returns:
          train_data: training data
          val_data: validation data
          test_data: testing data
    '''
    df_input = df_input

    smis = df_input.loc[:, smiles_column].values
    ys_prim = df_input.loc[:, target_columns].values
    if log_flag:
      ys = [np.log(y) for y in ys_prim]
    else:
      ys = ys_prim

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    mols = [d.mol for d in all_data]

    train_indices, val_indices, test_indices = data.make_split_indices(mols, self.split_type,
                                              (self.train_split, self.valid_split, self.test_split))
    self.train_data, self.val_data, self.test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices)

    print("Data read and split.")

  def featurize_dataloaders(self):
    '''
      featurizes the data and creates dataloaders for training, validation, and testing.
        Args:
          None
        Returns:
          scaler: a standard scaler for the targets
          train_loader: training dataloader
          val_loader: validation dataloader
          test_loader: testing dataloader
    '''
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    self.train_dset = data.MoleculeDataset(self.train_data[0], featurizer)
    scaler = self.train_dset.normalize_targets()

    self.val_dset = data.MoleculeDataset(self.val_data[0], featurizer)
    self.val_dset.normalize_targets(scaler)

    self.test_dset = data.MoleculeDataset(self.test_data[0], featurizer)

    train_loader = data.build_dataloader(self.train_dset, num_workers=self.num_workers)
    val_loader = data.build_dataloader(self.val_dset, num_workers=self.num_workers, shuffle=False)
    test_loader = data.build_dataloader(self.test_dset, num_workers=self.num_workers, shuffle=False)

    print("Data featurized.")

    return scaler, train_loader, val_loader, test_loader

  def get_full_dsets(self):
    '''
      returns the full training, validation, and testing datasets, expanded from the batches

      Args:
        None

      Returns:
        full_train: full training dataset
        full_val: full validation dataset
        full_test: full testing dataset
    '''

    full_test = []
    for i in range(len(self.test_dset)):
      full_test.append(self.test_dset[i].y.item())

    full_val = []
    for i in range(len(self.val_dset)):
      full_val.append(self.val_dset[i].y.item())

    full_train = []
    for i in range(len(self.train_dset)):
      full_train.append(self.train_dset[i].y.item())

    return full_train, full_val, full_test

  def make_new_dataloader(self, new_df, smiles_column, target_columns = None, log_flag = False):
    '''
      creates a new dataloader from a new dataframe

      Args:
        new_df: new dataframe
        smiles_column: the name of the SMILES column
        target_columns: (optiona) the names of the target columns
        log_flag: a flag to apply a log to the target values

      Returns:
        new_loader: new dataloader
    '''
    smis = new_df.loc[:, smiles_column].values

    if target_columns == None:
      ys = None
    else:
      ys_prim = new_df.loc[:, target_columns].values
      if log_flag:
        ys = [np.log(y) for y in ys_prim]
      else:
        ys = ys_prim

    new_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    new_dset = data.MoleculeDataset(new_data, featurizer=featurizer)
    new_loader = data.build_dataloader(new_dset, shuffle=False)

    return new_loader

class chemprop_model():
  '''
    class for constructing a model
  '''
  def __init__(self, mess_pass = "bond", aggre = "mean", batch_norm = True):
    '''
      initializes the Chemprop model class. This is a GNN based MPNN model.

      Args:
        mess_pass: type of message passing to use
        aggre: type of aggregation to use
        batch_norm: a flag to use batch normalization
      Returns:
        None
    '''
    self.mess_pass = mess_pass
    self.aggre = aggre
    self.batch_norm = batch_norm

    if self.mess_pass == "bond":
      self.mp = nn.BondMessagePassing()
    elif self.mess_pass == "atom":
      self.mp = nn.AtomMessagePassing()

    if self.aggre == "mean":
      self.agg = nn.MeanAggregation()
    elif self.aggre == "sum":
      self.agg = nn.SumAggregation()
    elif self.aggre == "norm":
      self.agg = nn.NormAggregation()

    print("Class chemprop_model initialized")

  def construct_model(self, scaler, model_type = 'regression', metrics = ["mae"]):
    '''
      puts together the model

      Args:
        model_type: type of model to construct
        metrics: list of metrics to use. can include: mse, rmse, mae, accuracy, binary-mcc,
                 multiclass-mcc, r2

      Returns:
        model: constructed model
    '''
    metrics_hash = {"mse": nn.metrics.MSE(), "rmse": nn.metrics.RMSE(), "mae": nn.metrics.MAE(),
                    "accuracy": nn.metrics.BinaryAccuracy(), "binary-mcc": nn.metrics.BinaryMCCMetric(),
                    "multiclass-mcc": nn.metrics.MulticlassMCCMetric(), "r2": nn.metrics.R2Score()}

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    if model_type == 'regression':
      self.ffn = nn.RegressionFFN(output_transform=output_transform)
    elif model_type == 'binary classification':
      self.ffn = nn.BinaryClassificationFFN(output_transform=output_transform)
    elif model_type == 'multiclass classification':
      self.ffn = nn.MulticlassClassificationFFN(output_transform=output_transform)

    metric_list = []
    for metric in metrics:
      if metric in metrics_hash.keys():
        metric_list.append(metrics_hash[metric])

    self.mpnn = models.MPNN(self.mp, self.agg, self.ffn, self.batch_norm, metric_list)
    self.mpnn

    return self.mpnn

  def train_model(self, train_loader, val_loader, epochs = 100, devices = 1):
    '''
      trains the model

      Args:
        train_loader: training dataloader
        val_loader: validation dataloader
        epochs: number of epochs to train for
        devices: number of devices to use

      Returns:
        trainer: trained model
    '''
    checkpointing = ModelCheckpoint("checkpoints", "best-{epoch}-{val_loss:.2f}",
                                    "val_loss", mode="min",save_last=True)


    self.trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=devices,
        max_epochs=epochs,
        callbacks=[checkpointing])

    self.trainer.fit(self.mpnn, train_loader, val_loader)

    return self.trainer

  def test_model(self, test_loader):
    '''
      tests the model

      Args:
        test_loader: testing dataloader

      Returns:
        results: results of testing
    '''
    results = self.trainer.test(dataloaders=test_loader)

    return results

  def get_test_preds(self, trial_loader):
    '''
      gets the predictions for the test set/ run Inference

      Args:
        trial_loader: testing dataloader

      Returns:
        trial_preds: predictions for the test set
    '''
    with torch.inference_mode():
      inftrainer = pl.Trainer(
          logger=None,
          enable_progress_bar=True,
          accelerator="auto",
          devices=1)

    raw_preds = inftrainer.predict(self.mpnn, trial_loader)

    trial_preds = []
    for sublist in raw_preds:
      for val in sublist:
        trial_preds.append(val[0].item())

    return trial_preds

  def r2_scores(self, full_test, test_loader):
    '''
      calculates the R2 score for the test set

      Args:
        full_test: full testing dataset
        test_loader: testing dataloader

      Returns:
        test_r2: R2 score for the test set
        also plots the data
    '''
    test_preds = self.get_test_preds(test_loader)

    test_r2 = r2_score(full_test, test_preds)

    print(f"Test R2 = {test_r2:.2f}")

    plt.scatter(full_test,test_preds)
    plt.xlabel("Experimental")
    plt.ylabel("Predicted")
    plt.show()

    return test_r2

  def save_model(self, saved_model_path):
    '''
      saves the model to a file

      Args:
        saved_model_path: path to save the model to

      Returns:
        None
    '''
    saved_model = Path(saved_model_path)

    model_dict = {"hyper_parameters": self.mpnn.hparams, "state_dict": self.mpnn.state_dict()}
    torch.save(model_dict, saved_model)

    print(f"Model saved to {saved_model_path}")

  def load_model(self, saved_model_path):
    '''
      loads the model from a file

      Args:
        saved_model_path: path to load the model from

      Returns:
        model: loaded model
    '''
    saved_model = Path(saved_model_path)
    model = MPNN.load_from_file(saved_model)

    self.mpnn = model

    return self.mpnn

# k_splits = KFold(n_splits=5)
# k_train_indices, k_val_indices, k_test_indices = [], [], []
# for fold in k_splits.split(mols):
#     k_train_indices.append(fold[0])
#     k_val_indices.append([])
#     k_test_indices.append(fold[1])
# k_train_data, _, k_test_data = data.split_data_by_indices(
#     all_data, k_train_indices, None, k_test_indices)