import tensorflow as tf
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
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import r2_score
import mordred
import deepchem as dc
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def scale_pca_split(f: np.array, y: np.array, Xa: list, use_scaler = True,
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
  if type(y) == list:
    y = np.array(y).reshape(-1,1)
  elif type(y) == np.ndarray:
    y = np.array(y)
  
  Xa = np.array(Xa).reshape(-1,1)
  y_smiles = np.concatenate((y,Xa),axis=1)

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
  
  y_train = ys_train[:,:-1].astype(float)
  y_valid = ys_valid[:,:-1].astype(float)
  smiles_train = ys_train[:,-1]
  smiles_valid = ys_valid[:,-1]

  print("Pre-processing done.")

  return X_train, X_valid, y_train, y_valid, smiles_train, smiles_valid, pca, scaler

class SkipDenseBlock(tf.keras.layers.Layer):
  '''
    Defines the SkipDense block for the Neural Network 
  '''
  def __init__(self, N_LAYERS,N_NEURONS,ACTIVATION,L2REG,SKIP=False,**kwargs):
    '''
      Sets up SkipDense parameters

        Args:
          N_LAYERS: number of layers per block
          N_NEURONS: number of neurons per layer
          ACTIVATION: activation functions
          L2REG: L regulation coefficient
          SKIP: boolean for using skip connections
        Returns:
          None
    '''
    super().__init__(**kwargs)
    self.SKIP = SKIP
    self.hidden = [tf.keras.layers.Dense(N_NEURONS, activation=ACTIVATION,
                                        kernel_regularizer=tf.keras.regularizers.l2(L2REG))
                  for _ in range(N_LAYERS)]
    self.cat = tf.keras.layers.Concatenate()

  def call(self,inputs):
    '''
      Call functions for the SkipDense block

        Args:
          inputs: input to the block
        Returns:
          SDB_out: output of the block
    '''
    Z=inputs
    for layer in self.hidden:
      Z = layer(Z)
    if self.SKIP == True:
      SDB_out = self.cat([inputs,Z])
    if self.SKIP == False:
      SDB_out = Z
    return SDB_out

class skipdense_model():
  '''
    Defines the skipdense model.
  '''
  def __init__(self, ntlu = 100, act= "relu", lrate = 0.002, alpha = 0.001, layers = 4, 
               epochs = 50, batch_size = 64, num_blocks = 2, wide = False, skip = [False, False],
               classifier_flag = False, num_classes = None):
    '''
      Reads in parameters for the model.

        Args:
          ntlu: number of neurons per layer
          act: activation function
          lrate: learning rate
          alpha: L2 regularization coefficient
          layers: number of layers per block
          epochs: number of epochs
          batch_size: batch size
          num_blocks: number of blocks
          wide: boolean for doing a parallel stack
          skip: boolean for using skip connections
          classifier_flag: boolean for using classifier
          num_classes: number of classes
        Returns:
          None
    '''
    self.ntlu = ntlu
    self.act = act
    self.lrate = lrate
    self.alpha = alpha
    self.layers = layers
    self.epochs = epochs
    self.batch_size = batch_size
    self.num_blocks = num_blocks
    self.wide = wide
    self.skip = skip
    self.classifier_flag = classifier_flag
    self.num_classes = num_classes
    self.cat = tf.keras.layers.Concatenate()

    print("skipdense model initialized!")
  
  def exp_decay(self):
    '''
      Sets up exponential decay for the learning rate.
    '''
    def exp_decay_fcn(epoch):
        return self.lrate*0.1**(epoch/self.epochs)
    return exp_decay_fcn

  def build_model(self, X_train):
    '''
      Builds the model.

        Args:
          X_train: training feature matrix
        Returns:
          model: built model
    '''
  
    input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
    norm_layer = tf.keras.layers.Normalization()(input_)
    
    # single pass through a stack of skip-dense blocks
    x = SkipDenseBlock(N_LAYERS = self.layers, N_NEURONS = self.ntlu, ACTIVATION = self.act, 
                       L2REG = self.alpha, SKIP = self.skip[0])(norm_layer) #(input_)
    
    for i in range(self.num_blocks - 1):
      x = SkipDenseBlock(N_LAYERS = self.layers, N_NEURONS = self.ntlu, ACTIVATION = self.act, 
                         L2REG = self.alpha, SKIP = self.skip[i])(x)
    
    # possible second stack of skip-dense blocks
    if self.wide:
      y = SkipDenseBlock(N_LAYERS = self.layers, N_NEURONS = self.ntlu, ACTIVATION = self.act, 
                         L2REG = self.alpha, SKIP = self.skip[0])(norm_layer) #(input_)
    
      for i in range(self.num_blocks - 1):
        y = SkipDenseBlock(N_LAYERS = self.layers, N_NEURONS = self.ntlu, ACTIVATION = self.act, 
                           L2REG = self.alpha, SKIP = self.skip[i])(y)
                         
      combined = self.cat([x,y])
      x = combined
    
    if self.classifier_flag == True:
      output_ = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
    else:
      output_ = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=[input_],outputs=[output_])

    model.summary()

    return model
  
  def train_model(self, model, X_train, y_train, X_valid, y_valid, optim = "Adam", rn_seed = 42):
    '''
      Trains the model.

        Args:
          model: built model
          X_train: training feature matrix
          y_train: training target list
          X_valid: validation feature matrix
          y_valid: validation target list
          optim: optimizer; options are "Adam" and "SGD"
          rn_seed: random seed
        Returns:
          model: trained model
          train_df: dataframe with training history
    '''
    tf.random.set_seed(rn_seed)
    exp_decay_fcn = self.exp_decay()
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exp_decay_fcn)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(r"Checks/skip.weights.h5",save_weights_only=True)
  
    if optim == "Adam":
      optimizer = tf.keras.optimizers.Adam(learning_rate=self.lrate)
    elif optim == "SGD":  
      optimizer = tf.keras.optimizers.SGD(learning_rate=self.lrate, momentum=0.9)   #clipvalue=1.0,

    if self.classifier_flag == True:
      model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"]) 
    else:
      model.compile(loss="mae",optimizer=optimizer) 

    history = model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=self.epochs,verbose=2,
                        batch_size=self.batch_size, callbacks=[checkpoint_cb, lr_scheduler])

    train_df = pd.DataFrame(history.history)
    max_y = train_df["loss"].max()
    train_df.plot(figsize=(6,4),xlim=[1,self.epochs],ylim=[0,max_y*1.1],xlabel="Epoch",
                                        style=["r--","r--.","b-","b-*"])

    picname = "skipdense.jpg"
    plt.savefig(picname)
    plt.show

    return model, train_df
  
  def eval_model(self, model, X_train, y_train, X_valid, y_valid):
    '''
      Evaluates the model.

        Args:
          model: trained model
          X_train: training feature matrix
          y_train: training target list
          X_valid: validation feature matrix
          y_valid: validation target list
        Returns:
          None; prints the R2 scores for the training and validation sets., diplays and 
          saves the evaluation plot.
    '''
    fnpredy = model.predict(X_train)
    vpredy = model.predict(X_valid)

    favey = np.average(y_train)
    vavey = np.average(y_valid)
    
    fsst, vsst = 0.0, 0.0
    for x in y_train:
        fsst = fsst + (x-favey)**2
    for x in y_valid:
        vsst = vsst + (x-vavey)**2

    fssr, vssr = 0.0, 0.0
    for i in range(len(fnpredy)):
        fssr = fssr +(y_train[i]-fnpredy[i])**2
    for i in range(len(vpredy)):
        vssr = vssr +(y_valid[i]-vpredy[i])**2
    fscore = 1-(fssr/fsst)
    vscore = 1-(vssr/vsst)

    print(f"Training set score is {fscore}, validation set score is {vscore}")

    plt.scatter(y_train,fnpredy,color="blue",label="ML-t")
   
    plt.scatter(y_valid,vpredy,color="green",label="ML-v")

    y_total = [*y_train,*y_valid]
    plt.plot(y_total,y_total,color="red",label="Best Fit")
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.show
    picname = "skipdense_eval.jpg"
    plt.savefig(picname)

  def save_model(self, model, model_name: str):
    '''
      Saves the model with the given model_name

        Args:
          model: model to save
          model_name: name for files
        Returns:
          None; saves the model and the parameters to the Checks/ folder
    '''
    f = open(f"Checks/{model_name}_params.txt", "w")
    f.write(f"ntlu: {self.ntlu}\n")
    f.write(f"act: {self.act}\n")
    f.write(f"lrate: {self.lrate}\n")
    f.write(f"alpha: {self.alpha}\n")
    f.write(f"layers: {self.layers}\n")
    f.write(f"epochs: {self.epochs}\n")
    f.write(f"batch_size: {self.batch_size}\n")
    f.write(f"num_blocks: {self.num_blocks}\n")
    f.write(f"skip: {self.skip}\n")
    f.write(f"classifier_flag: {self.classifier_flag}\n")
    f.write(f"num_classes: {self.num_classes}\n")
    f.close()

    os.system(f'mv Checks/skip.weights.h5 Checks/{model_name}.weights.h5')

    print("Paramters and weights saved to the Checks/ folder!")

def load_model(model_name: str, X_train):
  '''
    Loads a saved model. Requires the weights and parameters files

      Args:
        model_name: name of the model
        X_train: training feature matrix
      Returns:
        new_model: model class
        model: loaded model
  '''
  f = open(f"{model_name}_params.txt", "r")
  params = f.readlines()
  f.close()
  ntlu = int(params[0].split(":")[1])
  act = params[1].split(":")[1].strip().replace("\n","")
  lrate = float(params[2].split(":")[1])
  alpha = float(params[3].split(":")[1])
  layers = int(params[4].split(":")[1])
  epochs = int(params[5].split(":")[1])
  batch_size = int(params[6].split(":")[1])
  num_blocks = int(params[7].split(":")[1])
  bool_line = params[8].split(":")[1].strip().strip("[]")
  bools = bool_line.split(",")
  skip = []
  for bool in bools:
    if bool == "True":
      skip.append(True)
    elif bool == "False":
      skip.append(False)
    else:
      pass
  classifier_raw = params[9].split(":")[1].strip().replace("\n","")
  if classifier_raw == "True":
    classifier_flag = True
  elif classifier_raw == "False":
    classifier_flag = False
  else:
    raise ValueError("classifier_flag must be True or False")
  try:
    num_classes = int(params[10].split(":")[1])
  except:
    num_classes = None

  new_model = skipdense_model(ntlu=ntlu, act=act, lrate=lrate, alpha=alpha, layers=layers,
                              epochs=epochs, batch_size=batch_size, num_blocks=num_blocks,
                              skip=skip)
  model = new_model.build_model(X_train)
  model.load_weights(f"{model_name}.weights.h5")

  return new_model, model

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
    preds_output = np.argmax(preds_output,axis=1)
    y_preds = preds_output
    y_true = self.truths
    y_true = np.argmax(y_true,axis=1)
    self.plot_confusion_matrix(y_preds, y_true, labels=self.labels)