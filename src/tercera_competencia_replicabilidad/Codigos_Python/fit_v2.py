

import pandas as pd
import numpy as np
#import seaborn as sns
#pip install polars
#from umap import UMAP
#import matplotlib.pyplot as plt
#from sklearn.cluster import DBSCAN
#from sklearn.ensemble import  RandomForestClassifier
#from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
from multiprocessing import Pool
#import dask.dataframe as dd
from boruta import BorutaPy
import time

import pandas as pd
import numpy as np
#import seaborn as sns
import pytz
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import lightgbm as lgb

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour

from time import time
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import polars as pl

import os
import random
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import time
import optuna
from optuna.study import StudyDirection

from sklearn.model_selection import StratifiedKFold, cross_val_predict


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

              
#import matplotlib.pyplot as plt


def lgb_gan_eval_LGBM(y_pred, y_true, ganancia_acierto, costo_estimulo):
    if type(y_true ) == type( pd.DataFrame() ): 
        y_true= y_true.values.reshape(-1)
    if type(y_pred ) == type( pd.DataFrame() ): 
        y_pred = y_pred.values.reshape(-1)    
    
    assert len(y_pred.shape) == len(y_true.shape)==1
    assert y_pred.shape == y_true.shape
   
    ganancia = np.where(y_true == 1, ganancia_acierto, -costo_estimulo)
   
    sorted_indices = np.argsort(y_pred)[::-1]
    ganancia = ganancia[sorted_indices]
    
    
    ganancia = np.cumsum(ganancia)
        
    max_ganancia = np.max(ganancia)
    max_index = np.argmax(ganancia)+1
    print( 'envios = ', max_index)
    return max_ganancia

def lgb_gan_eval_envios(y_pred, y_true, ganancia_acierto, costo_estimulo):
    if type(y_true ) == type( pd.DataFrame() ): 
        y_true= y_true.values.reshape(-1)
    if type(y_pred ) == type( pd.DataFrame() ): 
        y_pred = y_pred.values.reshape(-1)    
    
    assert len(y_pred.shape) == len(y_true.shape)==1
    assert y_pred.shape == y_true.shape
   
    ganancia = np.where(y_true == 1, ganancia_acierto, -costo_estimulo)
   
    sorted_indices = np.argsort(y_pred)[::-1]
    ganancia = ganancia[sorted_indices]    
    
    ganancia = np.cumsum(ganancia)
        
    max_ganancia = np.max(ganancia)
    max_index = np.argmax(ganancia)+1
    print( 'envios = ', max_index)
    th= y_pred[sorted_indices[max_index] ]
    return max_ganancia, max_index, th #ganancia, envios  y th


def generate_random_list_numpy(seed, size, min_val, max_val ):
    """Generate random integers using NumPy"""
    np.random.seed(seed)
    return np.random.randint(min_val, max_val , size=size).tolist()

def find_threshold_by_cumsum(res_mean, target_negative_count):        
    sorted_preds = np.sort(res_mean)     
    #sorted_preds[-target_negative_count]
    #cumulative_negatives = np.arange(1, len(sorted_preds) + 1)
    #threshold_index = np.argmin(np.abs(cumulative_negatives - target_negative_count))      
    return sorted_preds[target_negative_count]
def calculate_false_negatives_auc(res_mean, y_train, max_threshold):
    thresholds = np.linspace(0, max_threshold, 300)
    false_neg_counts = []
    for threshold in thresholds:
        # Make predictions based on threshold
        y_pred_threshold = (res_mean >= threshold).astype(int)
        # Calculate false positives
        false_neg = ((y_pred_threshold == 0) & (y_train.to_pandas().to_numpy().reshape(-1) == 1)).sum()
        false_neg_counts.append(false_neg)        
  
    auc_false_neg = np.trapz(false_neg_counts, thresholds)    
    
    thresholds = np.linspace(0, max_threshold, 300)
    droped_neg_counts = []
    for threshold in thresholds:
        # Make predictions based on threshold
        y_pred_threshold = (res_mean <= threshold).astype(int)
        # Calculate false positives
        true_neg = ((y_pred_threshold == 0) & (y_train.to_pandas().to_numpy().reshape(-1) == 0)).sum()
        droped_neg_counts.append(true_neg)        
  
    auc_true_neg = np.trapz(droped_neg_counts, thresholds)    
    return auc_false_neg, auc_true_neg #,, thresholds, false_neg_counts

"""
trial_number = 1
cant_exp= 3
iter_limit= 400
n_folds= 3
mode_cv=True
mode_production= True
trains= [202104]
suggest_params=None
seed = trial_number
data_classification(data_x, mode_cv, mode_production ,trains, suggest_params,iter_limit, cant_exp, n_folds, seed)

"""


    

def generate_random_list_numpy(seed, size, min_val, max_val ):
    """Generate random integers using NumPy"""
    np.random.seed(seed)
    return np.random.randint(min_val, max_val , size=size).tolist()
 

def create_LGBM_dataset_th(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params, mode_stacking, mode_FM, th_data, neg_bagging_fraction):
    
    assert mes_test not in trains
    data_x = data_x.with_columns( pl.lit(1.0).alias('clase_peso')  )
    data_x = data_x.with_columns(
        pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(clase_peso_lgbm+0.00002)
        #.when(pl.col('clase_ternaria') == 'BAJA+1').then(clase_peso_lgbm+0.00001)
        .otherwise(pl.col('clase_peso'))  # Keep the original value if no condition matches
        .alias('clase_peso')
    )    
   
    if type(final_selection) == list:
        data_x_selected= data_x[final_selection]
        df_test = data_x_selected.filter(pl.col('foto_mes') == mes_test)
    else:
        data_x_selected= data_x    
        df_test = data_x_selected.filter(pl.col('foto_mes') == mes_test)
    
    #wd= data_x_selected.filter(data_x_selected["res_mean"] > th)
    #wd = data_x_selected.filter( data_x_selected["clase_ternaria"].is_in(['BAJA+2', 'BAJA+1']) )
    
    
    wd = data_x_selected.filter(
    (data_x_selected["future_predictions"] > th_data) | 
    (data_x_selected["clase_ternaria"].is_in(['BAJA+2', 'BAJA+1'])) )
    
    #resample = data_x_selected.filter(data_x_selected["future_predictions"] <= th)
    #resample = data_x_selected.filter[(data_x_selected["future_predictions"] <= th) & 
    #                       (data_x_selected["clase_ternaria"] == 'CONTINUA')]
    resample = data_x_selected.filter(  (pl.col("future_predictions").is_null()) | (pl.col("future_predictions") <= th_data) | (pl.col("clase_ternaria") == 'CONTINUA' ) )

    pesos = np.exp(resample['future_predictions']+0.01)
    resample= resample.to_pandas()
    chosen_rows = resample.sample(n=int(len(resample)* neg_bagging_fraction ), weights= pesos, random_state= params['seed'])
    
    #chosen_rows = resample.sample(n=int(len(resample)* neg_bagging_fraction),  seed= params['seed'])
    
    wd =wd.to_pandas()
    data_x_selected = pd.concat([wd, chosen_rows])
    #data_x_selected = wd.vstack(chosen_rows)
    data_x_selected = pl.from_pandas( data_x_selected     ) 
    
    df_train_3 = data_x_selected.filter(pl.col('foto_mes').is_in(trains))
    print( set(df_train_3['foto_mes'].unique()))
    print( set(df_test['foto_mes'].unique()))
    print( set(df_train_3['foto_mes'].unique()).intersection(set(df_test['foto_mes'].unique())))
    assert set(df_train_3['foto_mes'].unique()).intersection(set(df_test['foto_mes'].unique())) == set() 
       
    y_train = df_train_3.select(
        pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)
        .otherwise(0)
        .alias("y_train")
    )        
    w_train = df_train_3['clase_peso']       
    if not mode_stacking:
        X_train = df_train_3.drop(['future_predictions'])
    else:
        X_train = df_train_3
    if not mode_FM:           
          X_train = df_train_3.drop(['foto_mes'])    
        
    X_train = X_train.drop(['clase_ternaria','clase_peso'])
    
    train_data = lgb.Dataset( X_train.to_pandas(),
                         label=y_train.to_pandas(),
                         weight=w_train.to_pandas(), params=params)    
    print( X_train.shape)                         
    y_test = df_test.select(
        pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)
        .otherwise(0)
        .alias("y_test")
    )
    w_test = df_test['clase_peso']
    
    if not mode_stacking:
        X_test = df_test.drop(['future_predictions'])
    else:
        X_test = df_test
    if mode_FM:            
        #X_test= add_moth_encode( X_test)
        pass
    else:
        X_test = df_test.drop(['foto_mes'])    
    X_test = X_test.drop(['clase_ternaria','clase_peso'])
    
    #return train_data, X_test.to_pandas(), y_test.to_pandas() , X_train.to_pandas(), y_train.to_pandas() 
    if np.sum( y_test.to_numpy().reshape(-1) ) ==0:
        print( 'TEST IN FUTURE')
        return X_train, y_train,  w_train, X_test.to_pandas(), None ,params
    return X_train, y_train,  w_train, X_test.to_pandas(), y_test.to_pandas() ,params

    


import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def create_LGBM_dataset_basic2(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params, neg_bagging_fraction):
    #print( 'WARNING MINIMIZING FALSE NEGATIVES')  
    #print( 'WARNING MINIMIZING FALSE NEGATIVES')  
    assert mes_test not in trains
    data_x = data_x.with_columns( pl.lit(1.0).alias('clase_peso')  )
    data_x = data_x.with_columns(
        pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(clase_peso_lgbm+0.00002)
        #.when(pl.col('clase_ternaria') == 'BAJA+1').then(clase_peso_lgbm+0.00001)
        .otherwise(pl.col('clase_peso'))  # Keep the original value if no condition matches
        .alias('clase_peso')
    )    
  
    if type(final_selection) == list:
        data_x_selected= data_x[final_selection]
        df_test = data_x_selected.filter(pl.col('foto_mes') == mes_test)
    else:
        data_x_selected= data_x    
        df_test = data_x_selected.filter(pl.col('foto_mes') == mes_test)
    
    #wd= data_x_selected.filter(data_x_selected["res_mean"] > th)
    wd = data_x_selected.filter(    
    (data_x_selected["clase_ternaria"].is_in(['BAJA+2', 'BAJA+1'])) )
    
    
    #resample = data_x_selected[ (data_x_selected["clase_ternaria"] == 'CONTINUA')]
    resample = data_x_selected.filter(pl.col("clase_ternaria") == "CONTINUA")
    #pesos = np.exp(resample['res_mean']+0.01)
    chosen_rows = resample.sample(n=int(len(resample)* neg_bagging_fraction ),  seed= params['seed'])
    
    
    data_x_selected = pl.concat([wd, chosen_rows])
    
    
    df_train_3 = data_x_selected.filter(pl.col('foto_mes').is_in(trains))
    print( set(df_train_3['foto_mes'].unique()))
    print( set(df_test['foto_mes'].unique()))
    print( set(df_train_3['foto_mes'].unique()).intersection(set(df_test['foto_mes'].unique())))
    assert set(df_train_3['foto_mes'].unique()).intersection(set(df_test['foto_mes'].unique())) == set() 
       
    y_train = df_train_3.select(
        pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)        
        .otherwise(0)
        .alias("y_train")
    )        
    w_train = df_train_3['clase_peso']         
   
    X_train = df_train_3
  
          
    X_train = X_train.drop(['clase_ternaria', 'clase_peso'])
    """
    train_data = lgb.Dataset( X_train.to_pandas(),
                         label=y_train.to_pandas(),
                         weight=w_train.to_pandas(), params=params)    """
    print( X_train.shape)                         
    y_test = df_test.select(
        pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)
        .otherwise(0)
        .alias("y_test")
    )
    w_test = df_test['clase_peso']      
   
    X_test = df_test   
    X_test = X_test.drop(['clase_ternaria', 'clase_peso'])
    
    assert np.sum(y_train.to_numpy().reshape(-1) ) !=0
    if np.sum( y_test.to_numpy().reshape(-1) ) ==0:
        print( 'TEST IN FUTURE')
        return X_train, y_train,  w_train, X_test.to_pandas(), None ,params
    return X_train, y_train,  w_train, X_test.to_pandas(), y_test.to_pandas() ,params
   


def create_LGBM_dataset(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params, mode_stacking, mode_FM, th):
    
    assert mes_test not in trains
    data_x = data_x.with_columns( pl.lit(1.0).alias('clase_peso')  )
    data_x = data_x.with_columns(
        pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(clase_peso_lgbm+0.00002)
        #.when(pl.col('clase_ternaria') == 'BAJA+1').then(clase_peso_lgbm+0.00001)
        .otherwise(pl.col('clase_peso'))  # Keep the original value if no condition matches
        .alias('clase_peso')
    )    
    if  mode_FM:
        data_x = add_moth_encode( data_x) 
    if type(final_selection) == list:
        data_x_selected= data_x[final_selection]
        df_test = data_x_selected.filter(pl.col('foto_mes') == mes_test)
    else:
        data_x_selected= data_x    
        df_test = data_x_selected.filter(pl.col('foto_mes') == mes_test)
    
    #wd= data_x_selected.filter(data_x_selected["res_mean"] > th)
    wd = data_x_selected.filter(
    (data_x_selected["res_mean"] > th) | 
    (data_x_selected["clase_ternaria"].is_in(['BAJA+2', 'BAJA+1'])) )
    
    #resample = data_x_selected.filter(data_x_selected["res_mean"] <= th)
    resample = data_x_selected[(data_x_selected["res_mean"] <= th) & 
                           (data_x_selected["clase_ternaria"] == 'CONTINUA')]
    pesos = np.exp(resample['res_mean']+0.01)
    chosen_rows = resample.sample(n=int(len(resample)*params['neg_bagging_fraction']), weights= pesos, random_state= params['seed'])
    params['neg_bagging_fraction'] =1
    
    data_x_selected = pd.concat([wd, chosen_rows])
        
    
    df_train_3 = data_x_selected.filter(pl.col('foto_mes').is_in(trains))
    print( set(df_train_3['foto_mes'].unique()))
    print( set(df_test['foto_mes'].unique()))
    print( set(df_train_3['foto_mes'].unique()).intersection(set(df_test['foto_mes'].unique())))
    assert set(df_train_3['foto_mes'].unique()).intersection(set(df_test['foto_mes'].unique())) == set() 
       
    y_train = df_train_3.select(
        pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)
        .otherwise(0)
        .alias("y_train")
    )        
    w_train = df_train_3['clase_peso']       
    if not mode_stacking:
        X_train = df_train_3.drop(['res_mean'])
    else:
        X_train = df_train_3
    if not mode_FM:           
          X_train = df_train_3.drop(['foto_mes'])    
        
    X_train = X_train.drop(['clase_ternaria','clase_peso'])
    
    train_data = lgb.Dataset( X_train.to_pandas(),
                         label=y_train.to_pandas(),
                         weight=w_train.to_pandas(), params=params)    
    print( X_train.shape)                         
    y_test = df_test.select(
        pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)
        .otherwise(0)
        .alias("y_test")
    )
    w_test = df_test['clase_peso']
    
    if not mode_stacking:
        X_test = df_test.drop(['res_mean'])
    else:
        X_test = df_test
    if mode_FM:            
        #X_test= add_moth_encode( X_test)
        pass
    else:
        X_test = df_test.drop(['foto_mes'])    
    X_test = X_test.drop(['clase_ternaria','clase_peso'])
    
    return train_data, X_test.to_pandas(), y_test.to_pandas() , X_train.to_pandas(), y_train.to_pandas() 

    





def modify_params(params):
    # List of parameters to modify
    param_keys = ['learning_rate', 'feature_fraction', 'num_leaves', 'num_iterations', 'neg_bagging_fraction', 'bagging_fraction', 'min_data_in_leaf']
    #param_keys = [ 'feature_fraction', 'num_leaves', 'bagging_fraction']
    
    # Modify each parameter
    for key in param_keys:
        if key in params:
            if key in ['num_leaves', 'num_iterations', 'min_data_in_leaf']:  # Handle integers
                # Generate a random percentage change using normal distribution with mean 0 and std dev 0.01 (1%)
                sd= params[key] *.1
                change = random.gauss(0, sd)
                #change = random.gauss(0, 0.01)
                #change = random.gauss(0, 0.03)
                #change = random.gauss(0, 0.002)
                # Apply the percentage change and round to the nearest integer
                new_value = int(round(params[key] +  change ) )
                
                # Ensure the new value is positive and non-zero (for num_leaves and num_iterations)
                new_value = max(1, new_value)  # Ensure it's at least 1 (can't be 0)
                
            else:  # Handle floats
                # Generate a random percentage change using normal distribution with mean 0 and std dev 0.01 (1%)
                sd= params[key] *.1
                change = random.gauss(0, sd)
                #change = random.gauss(0, 0.005)
                #change = random.gauss(0, 0.015)
                #change = random.gauss(0, 0.001)
                # Apply the percentage change
                new_value = params[key]  + change
                
                # Ensure that the new value is positive (if it's a non-negative parameter)
                if new_value < 0:
                    new_value = 0.0001  # Prevent zero or negative values for certain params
            
            # Update the parameter in the dictionary
            params[key] = new_value
            
    return params




class CustomLightGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, params, n_models,mode, seed, min_data_in_leaf):
        self.params = params
        self.n_models = n_models
        self.models = []
        self.mode = mode
        self.seed = seed
        self.min_data_in_leaf = min_data_in_leaf

    def fit(self, X, y):
        self.models = []
        w_params= self.params
        w_params['min_data_in_leaf']  = int( len(X)  * self.min_data_in_leaf)
        
        print(self.params['min_data_in_leaf'])
        w_params['data_random_seed'] =  self.seed
        random_numbers= generate_random_list_numpy(self.seed, size=self.n_models, min_val=1, max_val=32767)
                  
        train_data = lgb.Dataset(X, label=y)
        
        for i in range(self.n_models):
            
            w_params['seed'] = random_numbers[i]
            w_params['bagging_seed'] = random_numbers[i]
            w_params['feature_fraction_seed'] = random_numbers[i]+1
            #w_params_run = modify_params( w_params.copy() )
            w_params_run = w_params.copy() 
            #print (w_params_run)
            model = lgb.train(
                 w_params_run ,
                train_data,
                feval=lgb_gan_eval_LGBM,
                #verbose_eval=False
            )
            #model = lgb.LGBMClassifier().fit(X,y )
            self.models.append(model)
        return self

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])  # Shape: (n_models, n_samples)
        
        if self.mode== 'none':
            sorted_preds = preds.T #np.sort(preds.T, axis=1)  # Sort each row (for each sample)
        if self.mode== 'sort':
            sorted_preds = np.sort(preds.T, axis=1)  # Sort each row (for each sample)
        if self.mode== 'max':
            sorted_preds = np.max(preds.T, axis=1)  # Sort each row (for each sample)    
        if self.mode== 'mean':
            sorted_preds = np.mean(preds.T, axis=1)  # Sort each row (for each sample)          
        if self.mode== 'mean5':
            sorted_preds = np.mean( np.sort(preds.T, axis=1)[:,-5:], axis=1)  # Sort each row (for each sample)    
        if self.mode== 'min':
            sorted_preds = np.min(preds.T, axis=1)  # Sort each row (for each sample)          
              
        return sorted_preds

    


def data_classification(data_x, suggest_params,   seed):   
    ganancia_acierto = 273000
    costo_estimulo =7000
    params = {
          "boosting_type": "gbdt",  # Can also be 'dart', but not tested with random_forest
          "objective": "binary",    # Binary classification
          "metric": "custom",       # Custom evaluation metric
          "first_metric_only": True,
          "boost_from_average": True,
          "feature_pre_filter": False,
          "force_row_wise": True,   # To reduce warnings
          "verbosity": -100,        # Set verbosity level to reduce output
          "max_depth": -1,          # No limit on depth
          "min_gain_to_split": 0.0, # Minimum gain to split a node
          "min_sum_hessian_in_leaf": 0.001, # Minimum sum of Hessian in leaf
          "lambda_l1": 0.0,         # L1 regularization
          "lambda_l2": 0.0,         # L2 regularization
          "max_bin": 31,            # Maximum number of bins
          #"num_iterations": 9999,   # Large number, controlled by early stopping
          #"bagging_fraction": 1.0,  # Fraction of data used for bagging
          #"pos_bagging_fraction": 1.0,  # Fraction of positive data used for bagging
          #"neg_bagging_fraction": 1.0,  # Fraction of negative data used for bagging
          "is_unbalance": False,    # Do not balance the classes
          "scale_pos_weight": 1,  # Weighting for positive class # 0.1
          #"scale_neg_weight": 1.0,  # Weighting for positive class # 1.0
          #"drop_rate": 0.1,         # Drop rate for DART (if used)
          #"max_drop": 50,           # Maximum number of drops for DART
          #"skip_drop": 0.5,         # Probability of skipping a drop for DART
          "extra_trees": False,     # Disable extra trees
          #'learning_rate': 0.5710265563272063, 'feature_fraction': 0.6678659849023909, 'num_leaves': 8, 'num_iterations': 25, 'neg_bagging_fraction': 0.02248066909316943, 'bagging_fraction': 0.38622712865058906
      }  
    if suggest_params != None:
        params.update(suggest_params)
    else:
        default= { 'min_data_in_leaf' : 0.00782738, 'learning_rate': 0.35, 'feature_fraction':0.767196, 'num_leaves': 36, 'num_iterations': 13, 'neg_bagging_fraction': 0.127811, 'bagging_fraction': 0.151587}
        default= {'learning_rate': 0.30626395740470547, 'feature_fraction': 0.3175923049296587, 'num_leaves': 457, 'num_iterations': 23, 'min_data_in_leaf': 0.006645547568462023, 'neg_bagging_fraction': 0.3505625529064554, 'bagging_fraction': 0.14821234319475232}
        default= {'learning_rate': 0.25918844068753766, 'feature_fraction': 0.6374747270740887, 'num_leaves': 460, 'num_iterations': 26, 'min_data_in_leaf': 0.00645591178248033, 'neg_bagging_fraction': 0.2900935243870532, 'bagging_fraction': 0.5099784979530179}
        default ={'learning_rate': 0.2875202173592346, 'feature_fraction': 0.4081533179654711, 'num_leaves': 383, 'num_iterations': 10, 'min_data_in_leaf': 0.004543722323727136, 'neg_bagging_fraction': 0.9755680360170047, 'bagging_fraction': 0.34254665393636974}
        params.update(default)      
    
    min_data_in_leaf =params['min_data_in_leaf']
    neg_bagging_fraction =params['neg_bagging_fraction']
    del params['min_data_in_leaf']    
    del params['neg_bagging_fraction']
            
    cant_semillas_ensamble = 250
    cant_exp = 1
    final_selection= None        
    clase_peso_lgbm = 1
    mode= 'mean'   
       
    futures= []
    performance = {}
    
    final_train= list( data_x['foto_mes'].unique() )
    for mes_future in final_train[2:]:
        if suggest_params != None:
            if mes_future!= final_train[-3]:
                continue
        i_mes = final_train.index(mes_future)
        trains= final_train[: i_mes-1]
        print(trains )
        print(mes_future )        
           
        params['data_random_seed'] =  seed          
        params['seed'] =  seed      
        mes_test = mes_future
        #data_x = pl.from_pandas(data)
        X_train, y_train,  w_train, X_test, y_test, params = create_LGBM_dataset_basic2 (final_selection, trains, mes_test, data_x, clase_peso_lgbm,params, neg_bagging_fraction)
        th_data= 0.901
        mode_stacking= True
        mode_FM = True
        #X_train, y_train,  w_train, X_test, y_test, params  = create_LGBM_dataset_th(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params, mode_stacking, mode_FM, th_data, neg_bagging_fraction)
        X_train, y_train =  X_train.to_pandas(), y_train.to_pandas()
        
        custom_lgbm = CustomLightGBM(params, cant_semillas_ensamble, mode, seed, min_data_in_leaf)
        custom_lgbm.fit(X_train.to_numpy(), y_train.to_numpy().reshape(-1))
        y_pred_fut_avg= custom_lgbm.predict(X_test.to_numpy())
        #solo para envios
        #y_pred_cv_nn = cross_val_predict(nn_model, y_custom_lgbm, y_train.values.ravel(), cv=3, method='predict_proba')   
        if suggest_params != None:         
            return lgb_gan_eval_LGBM(y_pred_fut_avg, y_test.values.ravel(), ganancia_acierto, costo_estimulo)
        if type(y_test)!= type(None):
            
            
            ganancia, envios, th = lgb_gan_eval_envios(y_pred_fut_avg, y_test.values.ravel(), ganancia_acierto, costo_estimulo)
            performance[mes_future] = (ganancia, envios, th)
            print(  lgb_gan_eval_envios(y_pred_fut_avg, y_test.values.ravel(), ganancia_acierto, costo_estimulo))                             
            
        
        
        
                
        #filtered_data = data_x.filter(pl.col('foto_mes') == mes_future).select('clase_ternaria')           
        #gan.append ( lgb_gan_eval(y_pred_future.reshape(-1), y_test.to_pandas().values.reshape(-1), ganancia_acierto, costo_estimulo) )
        futures.append(y_pred_fut_avg)
    
    futures_array = np.hstack(futures). reshape(-1) 
    data_x_pd = data_x.to_pandas()
    data_x_pd['future_predictions']  = np.nan
    data_x_pd.loc[data_x_pd['foto_mes'].isin( final_train[2:]), 'future_predictions' ]= futures_array
    return      pl.from_pandas(data_x_pd) 
   
    data_x_pd.to_csv( '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_03_test.csv')
    data_x =pl.from_pandas(data_x_pd) 
    data_x_b =  data_x
    data_x = add_lags_future_predictions(data_x, 3)
    
def add_lags_future_predictions(data, n_lags):
    numeric_cols_lagueables = ['future_predictions']
    
    data = data.sort(["numero_de_cliente", "foto_mes"])
    
    for col in numeric_cols_lagueables:
        for i in range(1, n_lags + 1):  # Include n_lags
            data = data.with_columns(
                pl.col(col).shift(i).over("numero_de_cliente").alias(f"{col}_lag{i}")
            )
    return data



def objective_data_classification(trial,data_w):
   
    params= {}
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.25, 1)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.05, 0.9)
    params['num_leaves'] = trial.suggest_int("num_leaves", 8, 500)
    params['num_iterations'] = trial.suggest_int("num_iterations", 1, 50)
    params['min_data_in_leaf'] = trial.suggest_float("min_data_in_leaf",  1.5E-05, 0.008)  # Example of leaf size        
    params['neg_bagging_fraction'] =  trial.suggest_float("neg_bagging_fraction", 0.02, 0.05)  
    params['pos_bagging_fraction'] = 1
    params['bagging_fraction'] = trial.suggest_float("bagging_fraction", 0.05, 1)
    #print(f"data_w: {data_w}")
    
    seed = trial.number   

    
    
   
    start= time.time()                      
    suggest_params = params
    ganancia= data_classification(data_w, suggest_params,  seed)
    elapsed_time=  time.time() - start 
   
    return  ganancia ,elapsed_time #- len(feature_selection )*penalty, time

def optuna_objective_data_classification():
    
 
    desktop_folder = '/home/reinaldomedina_robledo/Desktop/'
    desktop_folder = '/home/reinaldo_medina_gcp/Desktop/'
 
    
    path_data_x = desktop_folder+ 'data_x_final.parquet'
    exp_folder = '/home/reinaldo_medina_gcp/buckets/b2/comp_3_exp/escopeta_3_1/'
    nombre_exp_study = '1st_study_1.joblib'
    
    nombre_exp_study = '1st_study_250.joblib'
    study= joblib.load(exp_folder+nombre_exp_study )
    if os.path.exists(exp_folder+nombre_exp_study):
        study= joblib.load(exp_folder+nombre_exp_study )
    else: 
        #data_w= pl.read_parquet( '/home/reinaldomedina_robledo/Documents/competencia_01_mini.parquet')
        data_w= pl.read_parquet( path_data_x)        
       # data_w= data_x
        #study = optuna.create_study(direction="maximize")
        #study = optuna.create_study(direction="minimize")
        #study = optuna.create_study( directions=[StudyDirection.MINIMIZE,StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] ) #, timeout=60*60*2  )
        study = optuna.create_study( directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] ) #, timeout=60*60*2  )
        #study = optuna.create_study( directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] ) #, timeout=60*60*2  )

    for i in range(0, 4000):
        #study.optimize(objective, n_trials=1)  # You can specify the number of trials
        study.optimize(lambda trial: objective_data_classification(trial, data_w), n_trials=1, n_jobs=-1)
        #sdf =study.trials_dataframe()
        #study.trials[10]
        #study.optimize(objective_data_classification, n_trials=2, n_jobs=-1)
        joblib.dump( study, exp_folder+ nombre_exp_study)   
        
    """ 3 finished with values: [63.21171213214609, 30514.22332104275, 1037.7145915031433] and parameters: {'learning_rate': 0.8207775171619601, 'feature_fraction': 0.23640639561437643, 'num_leaves': 402, 'num_iterations': 29, 'min_data_in_leaf': 0.005473464714277627, 'neg_bagging_fraction': 0.1931051708608652, 'bagging_fraction': 0.193374290064174}.
    5 finished with values: [39.999552374643855, 24161.56068348571, 965.3486881256104] and parameters: {'learning_rate': 0.5750344086417372, 'feature_fraction': 0.1356758876690755, 'num_leaves': 111, 'num_iterations': 33, 'min_data_in_leaf': 0.005414715141237464, 'neg_bagging_fraction': 0.17228894720624818, 'bagging_fraction': 0.45431199818328394}.
    """



"""
sdf= study.trials_dataframe()

import matplotlib.pyplot as plt
plt.scatter( sdf['number'], sdf['values_0'])

plt.scatter( sdf['number'], np.sort( sdf['values_0']) )
"""



def calculate_treshold_cant_envios(y_test_true, y_pred_lgm, y_future, X_future):
    piso_envios = 4000
    techo_envios = 25000
    ganancia = np.where(y_test == 1, ganancia_acierto, 0) - np.where(y_test == 0, costo_estimulo, 0)    
    idx = np.argsort(y_pred_lgm)[::-1]    
    ganancia = ganancia[idx]
    y_pred_lgm = y_pred_lgm[idx]    
    ganancia_cum = np.cumsum(ganancia)
    max_ganancia_index = np.argmax(ganancia_cum)
    max_ganancia_value = ganancia_cum[max_ganancia_index]
    optimal_threshold = y_pred_lgm[max_ganancia_index]
    #np.sum(y_pred_lgm>=optimal_threshold) 
    
    ganancia_max = ganancia_cum.max()
    gan_max_idx = np.where(ganancia_cum == ganancia_max)[0][0]
   
    result= {}
    for i in range(piso_envios,techo_envios,500):
        idx = np.argsort(y_future)[::-1]    
        wt= y_future[idx[i]]
        w_final = np.where(y_future >= wt, 1, 0)
        w_final_df = X_future[['numero_de_cliente']].copy()  # Use .copy() to avoid SettingWithCopyWarning
        w_final_df['Predicted'] = w_final  # Add the predictions
        result[i]= w_final_df
    result['thresh_test']= optimal_threshold
    result['envios_test']= gan_max_idx
    return result

ds()

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters for LightGBM
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "custom",
    "first_metric_only": True,
    "boost_from_average": True,
    "feature_pre_filter": False,
    "force_row_wise": True,
    "verbosity": -100,
    "max_depth": -1,
    "min_gain_to_split": 0.0,
    "min_sum_hessian_in_leaf": 0.001,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "max_bin": 31,
    "num_iterations": 25,
    "is_unbalance": False,
    "scale_pos_weight": 1,
    "num_leaves": 8,
    "neg_bagging_fraction": 0.022,
    "bagging_fraction": 0.386,
}

# Base model
custom_lgbm = CustomLightGBM(params, n_models, mode)

# Stacking classifier
stack_clf = StackingClassifier(
    estimators=[('custom_lgbm', custom_lgbm)],
    final_estimator=LogisticRegression(),
    stack_method='predict'  # Use sorted predictions as input for meta-model
)

# Fit and evaluate
stack_clf.fit(X_train, y_train)
print(f"Accuracy: {stack_clf.score(X_test, y_test):.4f}")
