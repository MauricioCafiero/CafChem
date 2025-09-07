import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
import datasets
from datasets import load_dataset,Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

os.environ['WANDB_DISABLED'] = 'true'

class embed_training_data():
  '''
    Class to create datasets for embedding models
  '''
  def __init__(self, num_classes: int):
    '''
      Create a set of contrastive pairsm either from class data or target data.

      Args:
        num_classes: number of classes in the dataset
    '''
    self.num_classes = num_classes

  def read_with_classes(self, file_path: str, SMILES_column: str, classes_column: str,
                        classes_to_contrast: list[int], test_size = 0.15):
    '''
      Takes a CSV files with a SMILES column and a target column, create contrastive pairs, 
      return a dataframe and a split dataset. Cleans ions from the SMILES strings.

        Args:
          filen_path: name of the CSV file
          SMILES_column: name of the SMILES column
          classes_column: name of the classes column
          classes_to_contrast: which classes should have a 2 (dis-similar).
          test_size: size of the test set
        Returns:
          pairs_df: dataframe with the contrastive pairs
          pairs_ds: split dataset
    '''
    df = pd.read_csv(file_path)
    df[SMILES_column] = df[SMILES_column].apply(clean_ions)
    class_labels = df[classes_column].unique()
    
    class_holder_list = []
    for i in range(self.num_classes):
      df_temp = df[df[classes_column] == class_labels[1]]
      list_temp = df_temp[SMILES_column].tolist()   
      class_holder_list.append(list_temp)
      print(f"Length of class {class_labels[i]}: {len(list_temp)}")
    
    pairs = []

    f = open(f'{file_path}_pairs.csv', 'w')
    f.write("premise,hypothesis,label\n")

    for class_idx, classes in enumerate(classes_to_contrast):
      for i in range(len(class_holder_list[classes])):
        for j in range(i+1,len(class_holder_list[classes]),1):
          f.write(f"{class_holder_list[classes][i]},{class_holder_list[classes][j]}, 0\n")
      
      if classes < self.num_classes-1:
        for contrast_idx in classes_to_contrast[class_idx+1:]:
          for i in range(len(class_holder_list[classes])):
            for j in range(len(class_holder_list[contrast_idx])):
              f.write(f"{class_holder_list[classes][i]},{class_holder_list[contrast_idx][j]},2\n")

    f.close()
    pairs_df = pd.read_csv(f'{file_path}_pairs.csv')

    pairs_ds = load_dataset('csv', data_files=f'{file_path}_pairs.csv')
    pairs_ds = pairs_ds["train"].train_test_split(test_size=test_size)

    return pairs_df, pairs_ds

  def read_make_classes(self, file_path: str, target_name: str, 
                        classes_to_contrast: list[int], test_size = 0.15):
    '''
      Takes a CSV files with a SMILES column and a target column, divides it into the
      requested number of classes, sets the boundaries for thise classes, and assigns each 
      datapoint to a class. Also cleans ions from the SMILES strings. Calls 
      read_with_classes to create the contrastive pairs.

        Args:
          file_path: name of the CSV file
          target_name: name of the target column
          classes_to_contrast: which classes should have a 2 (dis-similar).
          test_size: size of the test set
        Returns:
          pairs_df: dataframe with the contrastive pairs
          pairs_ds: split dataset
    '''
    df = pd.read_csv(file_path)
    df.sort_values(by=[target_name],inplace=True)

    total_samples = len(df)
    samples_per_class = total_samples // self.num_classes
    print(f"Samples per class: {samples_per_class}, total samples:{total_samples}")


    bottom_range = df[target_name].iloc[0].item()

    range_cutoffs = []
    range_cutoffs.append(bottom_range)

    for i in range(samples_per_class,total_samples-self.num_classes, samples_per_class):
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
    df.to_csv(f"{file_path}_classes.csv", index = False)

    pairs_df, pairs_ds = self.read_with_classes(f"{file_path}_classes.csv", smiles_name, 
                                                "class labels", classes_to_contrast, test_size)

    return pairs_df, pairs_ds

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

class embedding_model():
  '''
    Class to train an embedding model.
  '''
  def __init__(self, training_dataset, validation_dataset,
               num_labels: int, num_epochs = 5, batch_size = 64, weight_decay = 0.01,               
               base_model_name = 'bert-base-cased', trained_suffix = 'finetuned-contrastive'):
    '''
      Accepts a contrastive pairs dataset and trains an embedding model.

        Args:
          training_dataset: contrastive pairs dataset
          validation_dataset: contrastive pairs dataset
          num_labels: number of classes in the dataset
          num_epochs: number of epochs to train the model
          batch_size: batch size for training
          weight_decay: weight decay for training
          base_model_name: base model to use for training
          trained_suffix: suffix to add to the model name
        Returns:
          None
    '''
    self.base_model_name = base_model_name
    self.trained_suffix = trained_suffix
    self.num_labels = num_labels
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.weight_decay = weight_decay
    self.training_dataset = training_dataset
    self.validation_dataset = validation_dataset

  def set_up_model(self):
    '''
      Sets up the model for training.

        Args:
          None
        Returns:
          None  
    '''
    model_name = self.base_model_name
    self.embedding_model = SentenceTransformer(model_name)

    train_loss = losses.SoftmaxLoss(model = self.embedding_model, 
                                    sentence_embedding_dimension = self.embedding_model.get_sentence_embedding_dimension(), 
                                    num_labels = self.num_labels)

    args = SentenceTransformerTrainingArguments(
        output_dir= f"{model_name}-{self.trained_suffix}",
        num_train_epochs=self.num_epochs,
        per_device_train_batch_size=self.batch_size,
        per_device_eval_batch_size=self.batch_size,
        warmup_steps=100,
        fp16=True,
        weight_decay=self.weight_decay,
        logging_steps=100,
        eval_steps = 100,
        report_to=None
        #logging_dir=f"{model_name}-finetuned-contrastive/logs",
    )

    trainer = SentenceTransformerTrainer(
        model = self.embedding_model,
        loss = train_loss,
        args = args,
        train_dataset = self.training_dataset,
        eval_dataset= self.validation_dataset)
    
    self.trainer = trainer

    print("Model set-up complete.")
  
  def train_model(self):
    '''
      Trains the model.

        Args:
          None
        Returns:
          None
    '''
    self.trainer.train()
    print("Model training complete.")

    return self.embedding_model
  
  def push_to_hub(self, repo_name: str):
    '''
      Pushes the model to the Huggingface Hub.

        Args:
          repo_name: name of the repository
        Returns:
          None
    '''
    self.embedding_model.push_to_hub(repo_name)
    print("Model saved to Huggingface Hub.")
  
  def load_model_from_hub(self, repo_name: str):
    '''
      Loads the model from the Huggingface Hub.

        Args:
          repo_name: name of the repository
        Returns:
          None
    '''
    self.embedding_model = SentenceTransformer(repo_name)
    print("Model loaded from Huggingface Hub.")

    return self.embedding_model
  
  def encode(self, smiles_list: list[str]):
    '''
      Encodes a list of smiles strings.

        Args:
          smiles_list: list of smiles strings
        Returns:
          embeddings: list of embeddings
    '''
    embeddings = self.embedding_model.encode(smiles_list)
    return embeddings
  
  def similarity_scalar(self, test_smiles: str, ref_smiles: str):
    '''
      Calculates the similarity between two smiles strings.

        Args:
          test_smiles: smiles string
          ref_smiles: smiles string
        Returns:
          similarity: similarity between the two smiles strings
    '''
    test_embedding = self.embedding_model.encode([test_smiles])
    ref_embedding = self.embedding_model.encode([ref_smiles])
    similarity = self.embedding_model.similarity(test_embedding, ref_embedding)
    return similarity
  
  def similarity_list(self, test_smiles: list[str], ref_smiles: str):
    '''
      Calculates the similarity between a list of smiles strings and a single smiles string.

        Args:
          test_smiles: list of smiles strings
          ref_smiles: smiles string
        Returns:
          similarities: list of similarities
    '''
    test_embeddings = self.embedding_model.encode(test_smiles)
    ref_embedding = self.embedding_model.encode([ref_smiles])
    similarities = self.embedding_model.similarity(test_embeddings, ref_embedding)
    return similarities
  
  def similarity_matrix(self, test_smiles: list[str]):
    '''
      Calculates the similarity between a list of smiles strings with itself.

        Args:
          test_smiles: list of smiles strings
        Returns:
          similarities: list of similarities
    '''
    test_embeddings = self.embedding_model.encode(test_smiles)
    print(test_embeddings.shape)
    similarities = self.embedding_model.similarity(test_embeddings, test_embeddings)
    return similarities

  def embed_and_featurize(self, smiles_list: list[str]):
    '''
      Encodes a list of smiles strings and returns a dataframe with the embeddings.

        Args:
          smiles_list: list of smiles strings
        Returns:
          df: dataframe with the embeddings
          embeddings: array of embeddings
    '''
    embeddings = self.embedding_model.encode(smiles_list)

    cols_list = [f"col-{i}" for i in range(len(embeddings[0]))]
    df = pd.DataFrame(embeddings, columns=cols_list)
    df.insert(0, "smiles", smiles_list)

    return df, embeddings

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