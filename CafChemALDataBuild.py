import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.models import ActiveLearner

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def GP_regression_std(regressor, X_in):
    _, std = regressor.predict(X_in, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X_in[query_idx]
    
class build_data():
    '''
    Class to build the minimum dataset which, when fit with a Gaussian Processes model,
    will give an R2 score for a vaidation set above a set threshold (0.65 default).
    '''
    def __init__(self, path_to_input: str, target_column: str, do_log = False, splits = 0.80, 
                 fraction_initial = 0.3, fraction_queries = 0.05, random_seed = 42):
        '''
        Initialize class

            Args:
                path_to_input: the input datafile
                target_column: which column to set as a target
                splits: what fraction to use for training data
                random seed: seed for splitting
                fraction_initial: what fraction of the training data to use for the initial learning set
                fraction_queries: what fraction of the training data to use for the queries
            Returns:
                None
        '''
        self.path_to_input = path_to_input
        self.target_column = target_column
        self.splits = splits
        self.random_seed = random_seed
        self.do_log = do_log
        self.fraction_initial = fraction_initial
        self.fraction_queries = fraction_queries

    def process_initial_data(self):
        '''
            Reads in the CSV file, detects the SMILES column, and featurizes the molecules using
            RDKit descriptors (217). Scales data with standard scaler, and splits into 
            training and validation sets. Finally, takes half of the training and sets it aside 
            to initiate the active learner.

                Args:
                    None
                Returns:
                    None, creates full trainig datasets (_raw) and learner datasets. Also creates
                    the validation sets.
        '''
        df = pd.read_csv(self.path_to_input)
        columns = df.columns
        for column in columns:
            if ("smiles" in column) or ("SMILES" in column) or ("Smiles" in column):
                self.smiles_column = column
                break

        
        smiles_raw = df[self.smiles_column].to_list()
        target = df[self.target_column].to_list()
        
        X_raw = []
        y = []
        smiles = []
        print_flag = False
        for i,smile in enumerate(smiles_raw):
            add_flag = True
            try:
              smile = smile.replace('[Na+].','').replace('.[Na+]','')
              mol = Chem.MolFromSmiles(smile)
              dictionary_descriptors = Chem.Descriptors.CalcMolDescriptors(mol)
              temp_vec = []
              for key in dictionary_descriptors:
                temp_vec.append(dictionary_descriptors[key])
              for val in temp_vec:
                  if 'nan' in str(val).lower():
                      add_flag = False
                      break
              if add_flag == True:
                  #print('adding')
                  X_raw.append(temp_vec)
                  y.append(target[i])
                  smiles.append(smile)
                  if print_flag == True:
                    print(f"{len(temp_vec)} descriptors calculated for: {smile}")
                    print("--------------------------------------------------------")
              else:
                  print(f"Could not featurize molecule {i}")
            except:
              print(f"Could not featurize molecule {i}")
        
        print(f"Total number of molecules: {len(X_raw)}")
        print(f"Total number of descriptors per molecule: {len(X_raw[0])}")

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        print('Data scaled')

        if self.do_log:
            y = [np.log10(val) for val in y]
        
        X = np.array(X)
        y = np.array(y)
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

        smiles = np.array(smiles)
        y_smiles = np.stack((y,smiles),axis=1)
        
        self.X_train_raw, self.X_valid, y_train_smi, y_valid_smi = train_test_split(X,y_smiles,train_size=self.splits, 
                                                                      random_state=self.random_seed, shuffle=True)
        
        self.y_train_raw = y_train_smi[:,0].astype(float)
        self.y_valid = y_valid_smi[:,0].astype(float)
        self.smiles_train_raw = y_train_smi[:,1]
        self.smiles_valid = y_valid_smi[:,1]
        
        print('Data split!')
        print(f'Length of training set: {self.X_train_raw.shape[0]}')

        self.n_initial = int(self.fraction_initial*len(self.X_train_raw))
        self.initial_idx = np.random.choice(range(len(self.X_train_raw)),size = self.n_initial, replace=False)
        self.original_size = len(self.X_train_raw)
        
        X_train = []
        y_train = []
        smiles_train = []
        for j in self.initial_idx:
            X_train.append(self.X_train_raw[j])
            y_train.append(self.y_train_raw[j])
            smiles_train.append(self.smiles_train_raw[j])

        for j in np.flip(np.sort(self.initial_idx)):
            self.X_train_raw = np.delete(self.X_train_raw, j, axis=0)
            self.y_train_raw = np.delete(self.y_train_raw, j, axis=0)
            self.smiles_train_raw = np.delete(self.smiles_train_raw, j, axis=0)

        self.smiles_train_dyn = list(smiles_train)
        self.X_train_dyn = np.array(X_train)
        self.y_train_dyn = np.array(y_train)
        
        self.smiles_train = list(smiles_train)
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        if self.X_train.shape[0] == self.y_train.shape[0]:
            print(f'Length of initial learning set: {self.X_train.shape[0]}')    

    def make_learner(self):
        '''
            Sets up the gaussian processes regressor as the active learner. Sets some variables
            for the learning loop.

                Args:
                    None
                Returns:
                    None; creates:
                        R2: holds validation R2 values at each cycle
                        points used: keeps track of what datapoints from X_train_raw have been used
                        data_memory: keeps track of any data points at each cycle
                        MAE_history: keeps track of validation MAE at each cycle
                        n_queries: how many queries per cycle; currently 10% of X_train_raw
                        
        '''
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-5,1e5)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5,1e+5))

        self.regressor = ActiveLearner(
            estimator = GaussianProcessRegressor(kernel=kernel),
            query_strategy=GP_regression_std,
            X_training=self.X_train, y_training=self.y_train)
        
        print('Active Learner initiated!')

        self.R2 = []
        self.points_used = list(self.initial_idx)
        self.data_memory = []
        self.MAE_history = []
        self.Train_score_history = []
        self.n_queries = int(self.fraction_queries*len(self.X_train_raw))
        
        print('initial data holders set!')

    def learning_loop(self, stop_criteria = 0.65):
        '''
            Loop for learning. Performs a cycle of learning and then evaluates the R2 for the 
            validation set. If it is above the threshold, it exits. Reports stats at each cycle

            Args:
                None
            Returns:
                None; at the end produces 'points_used' which has all points used to learn
        '''
        self.stop_criteria = stop_criteria
        
        
        while len(self.points_used) < (self.original_size - self.n_queries):
        
            temp_used = []
            for idx in range(self.n_queries):
                query_idx, query_inst = self.regressor.query(self.X_train_raw)
                #print(f'At step{idx} the query number is: {query_idx}')
                self.regressor.teach(self.X_train_raw[query_idx].reshape(1,-1), self.y_train_raw[query_idx].reshape(1,))
                temp_used.append(query_idx)

                self.X_train_dyn = np.concatenate((self.X_train_dyn, self.X_train_raw[query_idx,:].reshape(1,-1)), axis=0)
                self.y_train_dyn = np.concatenate((self.y_train_dyn, self.y_train_raw[query_idx].reshape(1)), axis=0)
                self.smiles_train_dyn.append(self.smiles_train_raw[query_idx])

                self.X_train_raw = np.delete(self.X_train_raw, query_idx, axis=0)
                self.y_train_raw = np.delete(self.y_train_raw, query_idx, axis=0)
                self.smiles_train_raw = np.delete(self.smiles_train_raw, query_idx, axis=0)
                
                
                if idx%5 ==0:
                    print(f"completed step {idx}")
            
            self.points_used.extend(temp_used)
            
            print(f"Active learning with {self.n_queries} datapoints complete")
            if len(self.data_memory) > 0:
                print(f'Total of {len(self.points_used) - self.data_memory[-1]} added to previous {self.data_memory[-1]} data points')
            else:
                print(f'Total of {len(self.points_used) - len(self.initial_idx)} added to previous {len(self.initial_idx)} data points')
            
            y_pred, y_std = self.regressor.predict(self.X_valid, return_std=True)
            y_pred, y_std = y_pred.ravel(), y_std.ravel()
            
            print("============================================")
            ave_diff = 0
            for exp, pred in zip(self.y_valid, y_pred):
                ave_diff += (exp-pred)
                #print(f'Experimental: {exp:10.3f}, Predicted: {pred:10.3f}')
            ave_diff /= len(self.y_valid)
            ave_diff = abs(ave_diff)
            print(f'MAE: {ave_diff:19.3f}')
            rs_score = r2_score(self.y_valid, y_pred)
            print(f'Current Validation set R2 score = {rs_score:10.3f}')
            
            r2_train, _ = self.train_score()
            print("============================================")
            
            self.R2.append(rs_score)
            self.Train_score_history.append(r2_train)
            self.MAE_history.append(ave_diff)
            self.data_memory.append(len(self.points_used))
            
            print('R2 score history:')
            for i, score in enumerate(self.R2):
                print(f"Train Score = {self.Train_score_history[i]:7.3f}, Val score = {score:7.3f}, Val MAE = {self.MAE_history[i]:7.3f}, total datapoints = {self.data_memory[i]:7}")

            print("============================================")
            
            if self.R2[-1] > self.stop_criteria:
                break

        self.finish_learning()

    def train_score(self):
        '''
        Calculates the R2 score for the training set.

            Args:
                None
            Returns:
                r2_train: the R2 score for the training set
                train_pred: the predicted values for the training set
        '''
        train_pred = self.regressor.predict(self.X_train_dyn, return_std=False)
        r2_train = r2_score(self.y_train_dyn,train_pred)
        print(f"Current training set R2 score: {r2_train:.3f}")

        return r2_train, train_pred

    def finish_learning(self):
        '''
        Saves the final dataset to a CSV file.

            Args:
                None
            Returns:
                None
        '''

        r2_train, train_pred = self.train_score()

        y_pred, y_std = self.regressor.predict(self.X_valid, return_std=True)
        y_pred, y_std = y_pred.ravel(), y_std.ravel()
    
        plt.scatter(self.y_train_dyn,train_pred,color="blue",label="ML-train")
        plt.scatter(self.y_valid,y_pred,color="green",label="ML-valid")
        plt.legend()
        plt.xlabel("known")
        plt.ylabel("predicted")
        plt.show

        if len(self.X_train_dyn) == self.data_memory[-1]:
            print(f'Built dataset with {self.data_memory[-1]} data points')
            built_dict = {'SMILES': self.smiles_train_dyn, 'target': self.y_train_dyn}
            df = pd.DataFrame.from_dict(built_dict)
            df.to_csv('input_file_ALBuilt.csv', index=False)
            print('Saved to CSV file.')

