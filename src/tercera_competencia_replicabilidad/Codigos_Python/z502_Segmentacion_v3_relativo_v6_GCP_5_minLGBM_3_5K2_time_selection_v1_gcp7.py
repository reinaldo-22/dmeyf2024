#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:24:19 2024

@author: reinaldo
"""


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
              
import matplotlib.pyplot as plt

class MockTrial:
    def __init__(self, params, number=0):
        self.params = params
        self.number = number     
    def suggest_float(self, name, low, high):
        return self.params.get(name, low) 
    def suggest_int(self, name, low, high):
        return self.params.get(name, low)     
    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])  

import os
from kaggle.api.kaggle_api_extended import KaggleApi

def to_kaggle_file (n_envios, y_future, exp_folder , X_future, trial_number) :
    idx = np.argsort(y_future)[::-1]    
    wt= y_future[idx[n_envios]]
    w_final = np.where(y_future >= wt, 1, 0)
    w_final_df = X_future[['numero_de_cliente']].copy()  # Use .copy() to avoid SettingWithCopyWarning
    w_final_df['Predicted'] = w_final  # Add the predictions
    submission_path = exp_folder + 'trial_' + str(trial_number)+'_env_'+str(n_envios)+'.csv'
    w_final_df.to_csv( submission_path, index=False)
    assert len( w_final)==165442
    return submission_path



def get_kaggle_score( submission_path, competition_name  ):
    api = KaggleApi()
    api.authenticate()
    for i in range (0,5):
        try:
            #api.competitions_submissions_submit(blob_file_tokens, submission_description='tst api', id )
            api.competition_submit(submission_path, message='submission X', competition=competition_name)
            
            #api.competitions_submissions_list(competition_name)
            time.sleep(20)
            for i in range (0,5):
                try: 
                    submissions = api.competition_submissions(competition=competition_name)
                    latest_submission = submissions[0] 
                    print("Submission ID:", latest_submission.ref)
                    print("Score:", latest_submission.publicScore)
                    print("Submission Status:", latest_submission.status)
                    return float(latest_submission.publicScore)
                except Exception as e:
                    time.sleep(20)
        except Exception as e:
            time.sleep(60)
            pass
    return '5 tryes at kaggle'

def add_lags_diff(data, lag_flag, n_lags,  delta_lag_flag ):
    campitos = ["numero_de_cliente", "foto_mes", "clase_ternaria"]
    numeric_cols_lagueables = [
        col for col in data.columns if col not in campitos and data.schema[col] in (pl.Int32, pl.Int64, pl.Float32, pl.Float64)
    ]
    
    # Sort data by 'numero_de_cliente' and 'foto_mes'
    data = data.sort(["numero_de_cliente", "foto_mes"])
    
    
    for col in numeric_cols_lagueables:
        for i in range(1, n_lags):
            data = data.with_columns(
                pl.col(col).shift(1).over("numero_de_cliente").alias(f"{col}_lag{i}")
            )
      
    if delta_lag_flag:
        for col in numeric_cols_lagueables:
            data = data.with_columns(
                (pl.col(col) - pl.col(f"{col}_lag1")).alias(f"{col}_delta1")
            )
    if not lag_flag:
        lag_columns = [f"{col}_lag1" for col in numeric_cols_lagueables]
        data = data.drop(*lag_columns)
    return data

def standardize_columns(data): 
    numeric_cols = [
        col for col in data.columns
        if data.schema[col] in (pl.Int32, pl.Int64, pl.Float32, pl.Float64)
        and col not in ["numero_de_cliente", "foto_mes"]
    ]    
    
    standardized_cols = [
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_std")
        for col in numeric_cols
    ]    
   
    data = data.with_columns(standardized_cols)
    return data

def percentage_binning(data):
    numeric_cols = [
        col for col in data.columns
        if data.schema[col] in (pl.Int32, pl.Int64, pl.Float32, pl.Float64)
        and col not in ["numero_de_cliente", "foto_mes"]
    ]    
    
    ranked_cols = [
        (pl.col(col).rank() / pl.count()).alias(f"{col}_rank")
        for col in numeric_cols
    ]
    
    data = data.with_columns(ranked_cols)
    return data
def convert_to_int_float32(df_train):
    for col in df_train.select_dtypes(include=['float64']).columns:
        df_train[col] = df_train[col].astype('float32')
    
    for col in df_train.select_dtypes(include=['int64']).columns:
        df_train[col] = df_train[col].astype('int32')
    return df_train

def convert_to_int_float32_polars(df_train):
    # Select float64 columns and cast them to float32
    float32_cols = [pl.col(col).cast(pl.Float32) for col in df_train.columns if df_train.schema[col] == pl.Float64]
    
    # Select int64 columns and cast them to int32
    int32_cols = [pl.col(col).cast(pl.Int32) for col in df_train.columns if df_train.schema[col] == pl.Int64]
    
    # Apply conversions and return modified DataFrame
    return df_train.with_columns(float32_cols + int32_cols)
def convert_to_int_float32_polars(df_train):
    # Define the maximum and minimum values for float32
    max_float32 = np.float32(3.4028235e+38)
    min_float32 = np.float32(-3.4028235e+38)
    
    # Select float64 columns and cast them to float32, while handling infinities
    float32_cols = [
        pl.when(pl.col(col).is_infinite())  # When column has infinity values
        .then(pl.when(pl.col(col) == np.inf)  # If positive infinity
              .then(pl.lit(max_float32))  # Replace  with max float32
              .otherwise(pl.lit(min_float32)))  # If negative infinity, replace with min float32
        .otherwise(pl.col(col))  # Otherwise keep the column as it is
        .cast(pl.Float32)
        .alias(col)
        for col in df_train.columns if df_train.schema[col] == pl.Float64
    ]
    
    # Select int64 columns and cast them to int32, replacing infinities with a max or min integer (if desired)
    int32_cols = [
        pl.when(pl.col(col).is_infinite())  # When column has infinity values
        .then(pl.lit(np.nan))  # Replace infinity with NaN (or any other placeholder)
        .otherwise(pl.col(col))  # Otherwise keep the column as it is
        .cast(pl.Int32)
        .alias(col)
        for col in df_train.columns if df_train.schema[col] == pl.Int64
    ]
    
    # Apply conversions and return modified DataFrame
    return df_train.with_columns(float32_cols + int32_cols)

def replace_infinities_with_limits(df_train):
    import numpy as np
    import polars as pl

    # Define replacement values for infinities
    max_limit = int(3.4e30)
    min_limit = int(-3.4e30)

    # Only apply replacement to numeric columns (float and int)
    return df_train.with_columns([
        pl.when(pl.col(col).is_infinite())  # Check for infinities
        .then(pl.when(pl.col(col) == np.inf).then(max_limit).otherwise(min_limit))  # Replace with limits
        .otherwise(pl.col(col))  # Keep the original value otherwise
        .alias(col)
        for col in df_train.columns if df_train.schema[col] in [pl.Float64,pl.Float32, pl.Int64, pl.Int32]  # Only numeric columns
    ])
def convert_to_int_float32_polars2(df_train):
    import numpy as np
    import polars as pl

    # Define the maximum and minimum representable values for float32
    max_float32 = np.float32(3.4028235e+30)
    min_float32 = np.float32(-3.4028235e+30)

    # Convert float64 columns to float32 while handling overflow
    float32_cols = [
        pl.when(pl.col(col) > max_float32)  # If value exceeds max float32
        .then(pl.lit(max_float32))  # Set to max float32
        .when(pl.col(col) < min_float32)  # If value is less than min float32
        .then(pl.lit(min_float32))  # Set to min float32
        .otherwise(pl.col(col))  # Otherwise, keep the original value
        .cast(pl.Float32)
        .alias(col)
        for col in df_train.columns if df_train.schema[col] == pl.Float64
    ]

    # Convert int64 columns to int32 (replace infinities or large values with placeholders if needed)
    int32_cols = [
        pl.when(pl.col(col).is_infinite())  # Handle infinity if present
        .then(pl.lit(np.nan))  # Replace infinity with NaN (or set a specific max/min int32 if preferred)
        .otherwise(pl.col(col))  # Otherwise, keep the column as it is
        .cast(pl.Int32)
        .alias(col)
        for col in df_train.columns if df_train.schema[col] == pl.Int64
    ]

    # Apply conversions and return the modified DataFrame
    return df_train.with_columns(float32_cols + int32_cols)

def time_features( df_train):
    w_df_train = df_train.sort_values(by=['numero_de_cliente', 'foto_mes'])    
    threshold = 0.1 * len(w_df_train)    
   
    for feature in w_df_train.columns:
        print(feature)
        if feature  in ['numero_de_cliente', 'foto_mes','clase_ternaria']:
            continue
        if not((w_df_train[feature].ne(0).sum() + w_df_train[feature].notna().sum()) > threshold):
            continue
        w_df_train[ feature+'_rolling_6'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.rolling(window=6).mean()).astype('float32')
        w_df_train[ feature+'_rolling_3'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.rolling(window=3).mean()).astype('float32')
        w_df_train[ feature+'_roll_6_diff_1'] = w_df_train.groupby('numero_de_cliente')[feature+'_rolling_3'].fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
        w_df_train[ feature+'_roll_3_diff_1'] = w_df_train.groupby('numero_de_cliente')[feature+'_rolling_3'].fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
        #w_df_train[ feature+'_diff_1'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
    return w_df_train



def time_features(df_train: pl.DataFrame) -> pl.DataFrame:
    # Sort dataframe
    w_df_train = df_train.sort(["numero_de_cliente", "foto_mes"])
    
    threshold = 0.1 * len(w_df_train)
    exclude_cols = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']
    
    # Get columns to process
    features = [
        col for col in w_df_train.columns 
        if col not in exclude_cols
    ]
    
    # Filter features based on non-zero and non-null threshold
    valid_features = []
    for feature in features:
        non_zero = w_df_train.select(
            pl.col(feature).ne(0).sum() + 
            pl.col(feature).is_not_null().sum()
        ).item()
        if non_zero > threshold:
            valid_features.append(feature)
    
    # Create expressions for rolling operations
    expressions = []
    for feature in valid_features:
        print(feature)
        
        # Fill nulls with 0 for calculations
        feat_col = pl.col(feature).fill_null(0)
        
        expressions.extend([
            # Rolling means
            feat_col.rolling_mean(window_size=6)
                .over("numero_de_cliente")
                .cast(pl.Float32)
                .alias(f"{feature}_rolling_6"),
                
            feat_col.rolling_mean(window_size=3)
                .over("numero_de_cliente")
                .cast(pl.Float32)
                .alias(f"{feature}_rolling_3"),
            
            # Rolling diffs
            feat_col.rolling_mean(window_size=6)
                .over("numero_de_cliente")
                .diff()
                .cast(pl.Float32)
                .alias(f"{feature}_roll_3_diff_6"),
                
            feat_col.rolling_mean(window_size=3)
                .over("numero_de_cliente")
                .diff()
                .cast(pl.Float32)
                .alias(f"{feature}_roll_2_diff_3"),
                
        ])
    """
    # Regular diff
    feat_col.diff()
    .over("numero_de_cliente")
    .cast(pl.Float32)
    .alias(f"{feature}_diff_1")"""
    # Apply all transformations at once
    return w_df_train.with_columns(expressions)
def bins_least_importatn( data,least_15_features, N_bins ):
    new_features=[]
    for feature in least_15_features:
         
         min_val = data.select(pl.col(feature).min()).item()
         max_val = data.select(pl.col(feature).max()).item()
         
         # Define bin edges
         bins = np.linspace(min_val, max_val, N_bins + 1)
         
         # Create conditional binning expression
         bin_expr = pl.col(feature)
         for i in range(N_bins):
             bin_expr = pl.when((pl.col(feature) >= bins[i]) & (pl.col(feature) < bins[i + 1])).then(i).otherwise(bin_expr)
         # Handle max edge inclusively
         bin_expr = pl.when(pl.col(feature) == max_val).then(N_bins - 1).otherwise(bin_expr)
    
         new_features.append(feature+'_binned')
         data = data.with_columns(bin_expr.alias(f"{feature}_binned"))
     
    data=  data.with_columns(sum=pl.sum_horizontal(new_features))    
    return data

def enhanced_feature_binning(data, features_list, N_bins=5): # agrupamiento por Claude Sonnet
    """
    Enhanced feature binning with logical grouping of related banking features
    """
    # Define feature groups based on banking domain knowledge
    feature_groups = {
        'credit_card_usage': [
            'Master_mconsumospesos', 'Visa_mconsumospesos',
            'Master_cconsumos', 'Visa_cconsumos',
            'Master_mconsumosdolares', 'Visa_mconsumosdolares',
            'Master_mlimitecompra', 'Visa_mlimitecompra'
        ],
        'payment_behavior': [
            'Master_mpagominimo', 'Visa_mpagominimo',
            'Master_mpagospesos', 'Visa_mpagospesos',
            'Master_mpagosdolares', 'Visa_mpagosdolares',
            'Master_mpagado', 'Visa_mpagado'
        ],
        'account_balances': [
            'Master_msaldopesos', 'Visa_msaldopesos',
            'Master_msaldodolares', 'Visa_msaldodolares',
            'Master_msaldototal', 'Visa_msaldototal',
            'mcaja_ahorro_dolares', 'mcaja_ahorro_adicional'
        ],
        'digital_engagement': [
            'internet', 'tmobile_app', 'thomebanking',
            'cmobile_app_trx', 'mpagomiscuentas', 'cpagomiscuentas'
        ],
        'service_usage': [
            'tcallcenter', 'ccallcenter_transacciones',
            'ctarjeta_debito_transacciones', 'catm_trx',
            'mautoservicio', 'matm', 'matm_other'
        ],
        'automatic_payments': [
            'mcuenta_debitos_automaticos', 'ccuenta_debitos_automaticos',
            'mttarjeta_visa_debitos_automaticos', 'mttarjeta_master_debitos_automaticos',
            'ctarjeta_visa_debitos_automaticos', 'ctarjeta_master_debitos_automaticos'
        ],
        'transfers': [
            'mtransferencias_recibidas', 'mtransferencias_emitidas',
            'ctransferencias_recibidas', 'ctransferencias_emitidas'
        ],
        'commissions': [
            'mcomisiones_mantenimiento', 'mcomisiones_otras', 'mcomisiones',
            'ccomisiones_otras', 'ccomisiones_mantenimiento'
        ]
    }
    
    new_features = []
    
    # Process each feature group
    for group_name, group_features in feature_groups.items():
        # Filter features that exist in our dataset
        existing_features = [f for f in group_features if f in features_list]
        if not existing_features:
            continue
            
        # Bin each feature in the group
        group_binned_features = []
        for feature in existing_features:
            min_val = data.select(pl.col(feature).min()).item()
            max_val = data.select(pl.col(feature).max()).item()
            
            # Define bin edges
            bins = np.linspace(min_val, max_val, N_bins + 1)
            
            # Create conditional binning expression
            bin_expr = pl.col(feature)
            for i in range(N_bins):
                bin_expr = pl.when(
                    (pl.col(feature) >= bins[i]) & (pl.col(feature) < bins[i + 1])
                ).then(i).otherwise(bin_expr)
            
            # Handle max edge inclusively
            bin_expr = pl.when(pl.col(feature) == max_val).then(N_bins - 1).otherwise(bin_expr)
            
            binned_feature_name = f"{feature}_binned"
            group_binned_features.append(binned_feature_name)
            data = data.with_columns(bin_expr.alias(binned_feature_name))
        
        # _d group aggregate feature
        if group_binned_features:
            group_feature_name = f"{group_name}_agg"
            data = data.with_columns(
                pl.sum_horizontal(group_binned_features).alias(group_feature_name)
            )
            new_features.append(group_feature_name)
            
    # Additional time-based features
    time_features = [f for f in features_list if any(x in f.lower() for x in ['fvencimiento', 'fultimo_cierre', 'finiciomora'])]
    if time_features:
        for feature in time_features:
            min_val = data.select(pl.col(feature).min()).item()
            max_val = data.select(pl.col(feature).max()).item()
            bins = np.linspace(min_val, max_val, N_bins + 1)
            
            bin_expr = pl.col(feature)
            for i in range(N_bins):
                bin_expr = pl.when(
                    (pl.col(feature) >= bins[i]) & (pl.col(feature) < bins[i + 1])
                ).then(i).otherwise(bin_expr)
            bin_expr = pl.when(pl.col(feature) == max_val).then(N_bins - 1).otherwise(bin_expr)
            
            data = data.with_columns(bin_expr.alias(f"{feature}_binned"))
            
    return data, new_features

def lgb_gan_eval_data(y_pred, data):
    #weight = data.get_weight()
    weight = data.weight.copy()
    max_weight=np.max(data.weight.copy())
    ganancia = np.where(weight == max_weight, ganancia_acierto, 0) - np.where(weight < max_weight, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    return 'gan_eval', np.max(ganancia) , True

"""
def lgb_gan_eval(y_pred, y_true, ganancia_acierto, costo_estimulo):
    #weight = data.get_weight()
    #weight = data.weight.copy()
    ganancia = np.where(y_pred == 1, ganancia_acierto, 0) - np.where(y_true ==0 , costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    return 'gan_eval', np.max(ganancia) , True
"""

def lgb_gan_eval(y_pred, y_true, ganancia_acierto, costo_estimulo):
   
    ganancia = np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true != 1 , costo_estimulo, 0)
    
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    return 'gan_eval', np.max(ganancia), True


def lgb_gan_eval(y_pred, y_true, ganancia_acierto, costo_estimulo):
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
    return 'gan_eval', max_ganancia, True


def add_features_manual( data):
    data = data.with_columns((pl.col('mcuenta_corriente') + pl.col('mcaja_ahorro')).alias('m_suma_CA_CC'))
    data = data.with_columns((pl.col('Master_mconsumospesos') + pl.col('Visa_mconsumospesos')).alias('Tarjetas_consumos_pesos'))
    data = data.with_columns((pl.col('Master_mconsumosdolares') + pl.col('Visa_msaldodolares')).alias('Tarjetas_consumos_colares'))
    data = data.with_columns((pl.col('mcuentas_saldo')/ pl.col('cliente_edad')).alias('saldo/edad'))
    return data

def atributos_presentes(data, columns):
    """Helper function to check if all columns are present in the dataset."""
    return all(col in data.columns for col in columns)

def add_features_manual(data):
    # Create initial features as in the original function
    data = data.with_columns((pl.col('mcuenta_corriente') + pl.col('mcaja_ahorro')).alias('m_suma_CA_CC'))
    data = data.with_columns((pl.col('Master_mconsumospesos') + pl.col('Visa_mconsumospesos')).alias('Tarjetas_consumos_pesos'))
    data = data.with_columns((pl.col('Master_mconsumosdolares') + pl.col('Visa_msaldodolares')).alias('Tarjetas_consumos_dolares'))
    data = data.with_columns((pl.col('mcuentas_saldo') / pl.col('cliente_edad')).alias('saldo_edad'))

    # Additional transformations based on the R code
    if atributos_presentes(data, ["foto_mes"]):
        data = data.with_columns((pl.col("foto_mes") % 100).alias("kmes"))

    if atributos_presentes(data, ["ctrx_quarter"]):
        data = data.with_columns(pl.col("ctrx_quarter").alias("ctrx_quarter_normalizado"))
        if atributos_presentes(data, ["cliente_antiguedad"]):
            data = data.with_columns(
                pl.when(pl.col("cliente_antiguedad") == 1).then(pl.col("ctrx_quarter") * 5)
                .when(pl.col("cliente_antiguedad") == 2).then(pl.col("ctrx_quarter") * 2)
                .when(pl.col("cliente_antiguedad") == 3).then(pl.col("ctrx_quarter") * 1.2)
                .otherwise(pl.col("ctrx_quarter")).alias("ctrx_quarter_normalizado")
            )

    if atributos_presentes(data, ["mpayroll", "cliente_edad"]):
        data = data.with_columns((pl.col("mpayroll") / pl.col("cliente_edad")).alias("mpayroll_sobre_edad"))

    if atributos_presentes(data, ["Master_status", "Visa_status"]):
        data = data.with_columns(
            pl.max_horizontal(["Master_status", "Visa_status"]).alias("vm_status01"),
            (pl.col("Master_status") + pl.col("Visa_status")).alias("vm_status02"),
            pl.max_horizontal([
                pl.col("Master_status").fill_null(10),
                pl.col("Visa_status").fill_null(10)
            ]).alias("vm_status03"),
            (pl.col("Master_status").fill_null(10) + pl.col("Visa_status").fill_null(10)).alias("vm_status04"),
            (pl.col("Master_status").fill_null(10) + pl.col("Visa_status").fill_null(100)).alias("vm_status05"),
            pl.when(pl.col("Visa_status").is_null()).then(pl.col("Master_status").fill_null(10))
             .otherwise(pl.col("Visa_status")).alias("vm_status06"),
            pl.when(pl.col("Master_status").is_null()).then(pl.col("Visa_status").fill_null(10))
             .otherwise(pl.col("Master_status")).alias("mv_status07")
        )

    # Combining MasterCard and Visa limits and other features
    if atributos_presentes(data, ["Master_mfinanciacion_limite", "Visa_mfinanciacion_limite"]):
        data = data.with_columns(
            (pl.col("Master_mfinanciacion_limite") + pl.col("Visa_mfinanciacion_limite")).alias("vm_mfinanciacion_limite")
        )

    # Continue adding other transformations similarly...
    
    # Infinite value handling
    # Convert infinite values to nulls to handle edge cases (optional depending on your needs)
    data = data.with_columns(
        pl.when(pl.col("variable").is_infinite()).then(None).otherwise(pl.col("variable")).alias("variable_with_no_infinity")
    )

    return data


def add_features_manual( data):
    data = data.with_columns((pl.col('mcuenta_corriente') + pl.col('mcaja_ahorro')).alias('m_suma_CA_CC'))
    data = data.with_columns((pl.col('Master_mconsumospesos') + pl.col('Visa_mconsumospesos')).alias('Tarjetas_consumos_pesos'))
    data = data.with_columns((pl.col('Master_mconsumosdolares') + pl.col('Visa_msaldodolares')).alias('Tarjetas_consumos_colares'))
    data = data.with_columns((pl.col('mcuentas_saldo')/ pl.col('cliente_edad')).alias('saldo/edad'))
    # New features from the R code
    if "foto_mes" in data.columns:
        data = data.with_columns((pl.col('foto_mes') % 100).alias('kmes'))
        
    if "ctrx_quarter" in data.columns:
        data = data.with_columns(pl.col('ctrx_quarter').cast(pl.Float64).alias('ctrx_quarter_normalizado'))
        
        if "cliente_antiguedad" in data.columns:
            data = data.with_columns(
                pl.when(pl.col('cliente_antiguedad') == 1)
                .then(pl.col('ctrx_quarter') * 5)
                .when(pl.col('cliente_antiguedad') == 2)
                .then(pl.col('ctrx_quarter') * 2)
                .when(pl.col('cliente_antiguedad') == 3)
                .then(pl.col('ctrx_quarter') * 1.2)
                .otherwise(pl.col('ctrx_quarter_normalizado'))
                .alias('ctrx_quarter_normalizado')
            )
    
    if "mpayroll" in data.columns and "cliente_edad" in data.columns:
        data = data.with_columns((pl.col('mpayroll') / pl.col('cliente_edad')).alias('mpayroll_sobre_edad'))
  
    return data
import polars as pl

def AgregarVariables_IntraMes(dataset):
    print("inicio AgregarVariables_IntraMes()")
    #Credito WUBA (GD)
    # INICIO de la seccion donde se deben hacer cambios con variables nuevas

    # el mes 1,2, ..12
    if 'foto_mes' in dataset.columns:
        #dataset = dataset.with_column((pl.col('foto_mes') % 100).alias('kmes'))
        dataset = dataset.with_columns([(pl.col('foto_mes') % 100).alias('kmes')])

    # creo un ctr_quarter que tenga en cuenta cuando
    # los clientes hace 3 menos meses que estan
    # ya que seria injusto considerar las transacciones medidas en menor tiempo
    if 'ctrx_quarter' in dataset.columns:
        dataset = dataset.with_columns(pl.when(pl.col('cliente_antiguedad') == 1).then(pl.col('ctrx_quarter') * 5)
                                      .when(pl.col('cliente_antiguedad') == 2).then(pl.col('ctrx_quarter') * 2)
                                      .when(pl.col('cliente_antiguedad') == 3).then(pl.col('ctrx_quarter') * 1.2)
                                      .otherwise(pl.col('ctrx_quarter'))
                                      .alias('ctrx_quarter_normalizado'))

    # variable extraida de una tesis de maestria de Irlanda
    if 'mpayroll' in dataset.columns and 'cliente_edad' in dataset.columns:
        dataset = dataset.with_columns((pl.col('mpayroll') / pl.col('cliente_edad')).alias('mpayroll_sobre_edad'))

  
    # combino MasterCard y Visa
    if 'Master_mfinanciacion_limite' in dataset.columns and 'Visa_mfinanciacion_limite' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mfinanciacion_limite') + pl.col('Visa_mfinanciacion_limite')).alias('vm_mfinanciacion_limite'))

  


    if 'Master_msaldototal' in dataset.columns and 'Visa_msaldototal' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_msaldototal') + pl.col('Visa_msaldototal')).alias('vm_msaldototal'))

    if 'Master_msaldopesos' in dataset.columns and 'Visa_msaldopesos' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_msaldopesos') + pl.col('Visa_msaldopesos')).alias('vm_msaldopesos'))

    if 'Master_msaldodolares' in dataset.columns and 'Visa_msaldodolares' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_msaldodolares') + pl.col('Visa_msaldodolares')).alias('vm_msaldodolares'))

    if 'Master_mconsumospesos' in dataset.columns and 'Visa_mconsumospesos' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mconsumospesos') + pl.col('Visa_mconsumospesos')).alias('vm_mconsumospesos'))

    if 'Master_mconsumosdolares' in dataset.columns and 'Visa_mconsumosdolares' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mconsumosdolares') + pl.col('Visa_mconsumosdolares')).alias('vm_mconsumosdolares'))

    if 'Master_mlimitecompra' in dataset.columns and 'Visa_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mlimitecompra') + pl.col('Visa_mlimitecompra')).alias('vm_mlimitecompra'))

    if 'Master_madelantopesos' in dataset.columns and 'Visa_madelantopesos' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_madelantopesos') + pl.col('Visa_madelantopesos')).alias('vm_madelantopesos'))

    if 'Master_madelantodolares' in dataset.columns and 'Visa_madelantodolares' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_madelantodolares') + pl.col('Visa_madelantodolares')).alias('vm_madelantodolares'))

   
    if 'Master_mpagado' in dataset.columns and 'Visa_mpagado' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mpagado') + pl.col('Visa_mpagado')).alias('vm_mpagado'))

    if 'Master_mpagospesos' in dataset.columns and 'Visa_mpagospesos' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mpagospesos') + pl.col('Visa_mpagospesos')).alias('vm_mpagospesos'))

    if 'Master_mpagosdolares' in dataset.columns and 'Visa_mpagosdolares' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mpagosdolares') + pl.col('Visa_mpagosdolares')).alias('vm_mpagosdolares'))

    
    if 'Master_mconsumototal' in dataset.columns and 'Visa_mconsumototal' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mconsumototal') + pl.col('Visa_mconsumototal')).alias('vm_mconsumototal'))

    if 'Master_cconsumos' in dataset.columns and 'Visa_cconsumos' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_cconsumos') + pl.col('Visa_cconsumos')).alias('vm_cconsumos'))

    if 'Master_cadelantosefectivo' in dataset.columns and 'Visa_cadelantosefectivo' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_cadelantosefectivo') + pl.col('Visa_cadelantosefectivo')).alias('vm_cadelantosefectivo'))

    if 'Master_mpagominimo' in dataset.columns and 'Visa_mpagominimo' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mpagominimo') + pl.col('Visa_mpagominimo')).alias('vm_mpagominimo'))

    # a partir de aqui juego con la suma de Mastercard y Visa
    if 'Master_mlimitecompra' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Master_mlimitecompra') / pl.col('vm_mlimitecompra')).alias('vmr_Master_mlimitecompra'))

    if 'Visa_mlimitecompra' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('Visa_mlimitecompra') / pl.col('vm_mlimitecompra')).alias('vmr_Visa_mlimitecompra'))

    if 'vm_msaldototal' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_msaldototal') / pl.col('vm_mlimitecompra')).alias('vmr_msaldototal'))

    if 'vm_msaldopesos' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_msaldopesos') / pl.col('vm_mlimitecompra')).alias('vmr_msaldopesos'))

    if 'vm_msaldopesos' in dataset.columns and 'vm_msaldototal' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_msaldopesos') / pl.col('vm_msaldototal')).alias('vmr_msaldopesos2'))


 
    if 'vm_mconsumospesos' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_mconsumospesos') / pl.col('vm_mlimitecompra')).alias('vmr_mconsumospesos'))

    if 'vm_mconsumosdolares' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_mconsumosdolares') / pl.col('vm_mlimitecompra')).alias('vmr_mconsumosdolares'))

    if 'vm_madelantopesos' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_madelantopesos') / pl.col('vm_mlimitecompra')).alias('vmr_madelantopesos'))

    if 'vm_madelantodolares' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_madelantodolares') / pl.col('vm_mlimitecompra')).alias('vmr_madelantodolares'))

    if 'vm_mpagado' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_mpagado') / pl.col('vm_mlimitecompra')).alias('vmr_mpagado'))

    if 'vm_mpagospesos' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_mpagospesos') / pl.col('vm_mlimitecompra')).alias('vmr_mpagospesos'))

    if 'vm_mpagosdolares' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_mpagosdolares') / pl.col('vm_mlimitecompra')).alias('vmr_mpagosdolares'))

    if 'vm_mconsumototal' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_mconsumototal') / pl.col('vm_mlimitecompra')).alias('vmr_mconsumototal'))

    if 'vm_mpagominimo' in dataset.columns and 'vm_mlimitecompra' in dataset.columns:
        dataset = dataset.with_columns((pl.col('vm_mpagominimo') / pl.col('vm_mlimitecompra')).alias('vmr_mpagominimo'))

    return dataset





def add_moth_encode( data):
    data = data.with_columns(
        [
            (pl.col('foto_mes') % 100).alias('month'),  # Extract month
            (np.sin((pl.col('foto_mes') % 100) * (2 * np.pi / 12))).alias('month_sin'),  # Sine encoding
            (np.cos((pl.col('foto_mes') % 100) * (2 * np.pi / 12))).alias('month_cos')   # Cosine encoding
        ]
    )
    return data




def create_random_normal(num_vectors, len_rnd ):
    
    mean_range = (-2, 2)
    sd_value = 4  # Fixed standard deviation
    
    # Generate random normal vectors
    random_vectors = []
    for _ in range(num_vectors):
        # Generate a random mean from the specified range
        u = np.random.uniform(*mean_range)
        # Generate a random vector with the specified mean and standard deviation
        vector = np.random.normal(loc=u, scale=sd_value, size=len_rnd)
        random_vectors.append(vector)
    
    # Convert to a NumPy array for easier manipulation if needed
    random_vectors = np.array(random_vectors).reshape((-1,num_vectors) )
    col_names = ['canarito_'+ str(i) for i in range(num_vectors)]
    #random_vectors= pd.DataFrame( random_vectors, columns= col_names)
    pl_random_vectors = pl.DataFrame(random_vectors, schema=col_names)
    return pl_random_vectors









import optuna
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

#claude
def regression_per_client(data: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    predictions = []
    
    for client in data.get_column('numero_de_cliente').unique():
        # Get client data and sort months
        client_df = data.filter(pl.col('numero_de_cliente') == client)
        meses = client_df.get_column('foto_mes').unique().sort()
        
        if len(meses) < 2:
            continue
            
        # Process each month except the first
        for mes in meses[1:]:
            # Get 2-month window
            window_df = client_df.filter(
                pl.col('foto_mes').is_in([mes - 1, mes])
            )
            
            if len(window_df) < 2:
                continue
                
            X = window_df.get_column('foto_mes').to_numpy().reshape(-1, 1)
            future_month = mes + 2
            
            # Calculate predictions for each feature
            for feat in features:
                try:
                    y = window_df.get_column(feat).to_numpy()
                    model = LinearRegression().fit(X, y)
                    prediction = model.predict([[future_month]])[0]
                    
                    predictions.append({
                        'numero_de_cliente': client,
                        'foto_mes': future_month,
                        f'{feat}plus_2M': prediction
                    })
                except Exception:
                    continue
    
    return pl.DataFrame(predictions)


        



import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error

def objective(trial, X, y,seeds):
    params ={}    
    params['metric'] = 'mape'
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.3)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.01, 1.0)
    params['num_leaves'] = trial.suggest_int("num_leaves", 31, 256)  # Example of leaf size
    params['min_child_samples'] = trial.suggest_int("min_child_samples", 1, 100)  # Example of coverage   
    params['max_depth'] = trial.suggest_int("max_depth", 3, 12)  # Control tree depth
    params['min_gain_to_split'] = trial.suggest_float("min_gain_to_split", 0.0, 1.0)  # Minimum gain to split
    #params['lambda_l1'] = trial.suggest_float("lambda_l1", 0.0, 10.0)  # L1 regularization
    #params['lambda_l2'] = trial.suggest_float("lambda_l2", 0.0, 10.0)  # L2 regularization
    params['num_iterations'] = 100
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.1, 1.0)   
   
   
    res=[]
    for seed in seeds:
        params['seed']= seed
        model = MultiOutputRegressor(LGBMRegressor(**params))       
        model.fit(X_train, y_train)    
       
        y_pred = model.predict(X_test)
        res.append(  mean_squared_error(y_test, y_pred) )


    return np.mean(res)

def objective(trial, X_train, y_train, X_test, y_test, seeds):
    params = {}
    params['alpha'] = trial.suggest_float("alpha", 0.01, 20.0)  # Regularization strength
    params['l1_ratio'] = trial.suggest_float("l1_ratio", 0.0, 1.0)  # Mix of L1 and L2 regularization
    params['max_iter'] = 500
    params['tol'] =  1e-3# trial.suggest_float("tol", 1e-4, 1e-1)  # Tolerance for stopping criteria
   
  
    model = MultiOutputRegressor(ElasticNet(**params))
    model.fit( X_train.fillna(X_train.mean()), y_train)

    y_pred = model.predict( X_test.fillna(X_test.mean()))
    res = mean_squared_error(y_test, y_pred) 

    return res



def forectas_numeric():
    trains= [202101,202102 ]
    mes_test= 202104
    full_trains= [202102,202103,202104, 202105, 202106 ]
    future_pred= 202106
    
    df_train_3 = data.filter(pl.col('foto_mes').is_in(trains))
    df_test = data.filter(pl.col('foto_mes') ==  mes_test)
    
    df_train_3 = subsample_data_time_polars(df_train_3, 0.02,  'CONTINUA', 'clase_ternaria',4)
    y_train = df_train_3[features]
    y_test = df_test[features]
    df_train_3.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
    X_train = df_train_3.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
    X_test = df_test.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
    
    X_train = X_train[ features+['month', 'month_sin','month_cos'] ] 
    X_test = X_test[ features+['month', 'month_sin','month_cos'] ] 
    
    y_train = y_train.to_pandas().dropna()
    y_test = y_test.to_pandas().dropna()
    X_train = X_train.loc[ y_train.index]
    X_test = X_test.loc[ y_test.index]
    
    variance_threshold =1E2
    
    variances = y_train.var()
    high_var = variances[variances > variance_threshold].index
    
    y_train = y_train[high_var]
    y_test = y_test[high_var]
       
    #def optimize_model(X, y, n_trials=100, n_seeds=5):

    for t_col in high_var:
        study = optuna.create_study(direction='minimize')
        for i in range(100):
            # Use different seeds for robustness
            print(f"Running trial with seed {seed}")
            #study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, seeds), n_trials=1)
            
            study.optimize(lambda trial: objective(trial, X_train, y_test[t_col], seeds), n_trials=1)
    
    # Return the optimized model based on best parameters
    best_params = study.best_params
    best_model = MultiOutputRegressor(LGBMRegressor(**best_params))
    
    best_model = MultiOutputRegressor(ElasticNet(**best_params))
    best_model.fit(X_train.fillna(0), y_train)
    y_pred = best_model.predict(X_test.fillna(0))
    
    col_train = y_train.columns.to_list()
    df_test = df_test.to_pandas()
    col_train_pred= [ col+'pred' for col in col_train]
    y_pred= pd.DataFrame( y_pred, columns= col_train_pred )
    w_df_test = df_test[col_train+ ['clase_ternaria', 'foto_mes', 'numero_de_cliente'] ]
    
    w_df_test2 = pd.concat( [w_df_test, y_pred], axis=1)
    w_df_test2= w_df_test2.loc[ w_df_test2['clase_ternaria']!='CONTINUA']
    
    return best_model

import numpy as np
import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import polars as pl
import pandas as pd

def objective(trial, X_train, y_train, X_train_final, y_train_final):
    """
    Objective function for single target optimization
    """
    params = {
        'alpha': trial.suggest_float("alpha", 0.01, 20.0),
        'l1_ratio': trial.suggest_float("l1_ratio", 0.0, 1.0),
        'max_iter': 500,
        'tol': 1e-3
    }
    
    model = ElasticNet(**params)
    model.fit(X_train , y_train)
    
    y_pred = model.predict( X_train_final)
    
    return mean_squared_error(y_pred, y_train_final)
   
    
def  optimize_and_predict(X_train, X_train_final, X_future, y_train,y_train_final,  n_trials):
    """
    Optimize model for a single target column and return predictions
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_train_final, y_train_final  ), 
        n_trials=n_trials,
        n_jobs=-1
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    best_value = study.best_value
    best_params.update({'max_iter': 50, 'tol': np.mean(y_train)/100})
    
    final_model = ElasticNet(**best_params)
    final_model.fit(X_train_final.fillna(X_train_final.mean().fillna(0)), y_train_final)
    
    # Make predictions
    #train_pred = final_model.predict(X_train.fillna(X_train.mean()))
    #test_pred = final_model.predict(X_test.fillna(X_test.mean()))
    future_pred = final_model.predict(X_future.fillna(X_future.mean().fillna(0)))
    
    #return train_pred, test_pred,future_pred, best_params
    return future_pred, best_params,best_value**.5/np.mean(y_train_final)

def forecast_numeric(features_above_canritos, data, features, wres,  subsample_ratio=0.02, variance_threshold=1E2):
  
    #df_train = data.filter(pl.col('foto_mes').is_in(trains))
    #df_test = data.filter(pl.col('foto_mes') == mes_test)
    
    
    df_train = data.filter(pl.col('foto_mes').is_in ( wres['X_train'] ))
    df_y_train = data.filter(pl.col('foto_mes') == wres['y_train'])
    
    df_final_train = data.filter(pl.col('foto_mes').is_in ( wres['finaL_train'] ))
    df_y_final_train = data.filter(pl.col('foto_mes') == wres['y_final_train'])
    
    df_X_future = data.filter(pl.col('foto_mes').is_in ( wres['X_future'] ))
    
    
    df_train = subsample_data_time_polars(df_train, subsample_ratio, 'CONTINUA', 'clase_ternaria', 4)
    df_final_train = subsample_data_time_polars(df_final_train, subsample_ratio, 'CONTINUA', 'clase_ternaria', 4)
        
    def get_data(col, df_train, df_final_train, df_y_train, df_y_final_train,df_X_future):
        #feature_cols_X =  ['month', 'month_sin', 'month_cos']
        X_train = df_train.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente'] ).to_pandas()
        X_final_train = df_final_train.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
        y_train = df_y_train.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
        y_final_train = df_y_final_train.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
        X_future = df_X_future.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
    
    
        y_train = y_train[col].dropna()
        y_final_train = y_final_train[col].dropna()
     
        common_index = y_train.index.intersection(X_train.index)    
        X_train = X_train.loc[common_index]
        y_train = y_train.loc[common_index]
        
        common_index = y_final_train.index.intersection(X_final_train.index)    
        X_final_train = X_final_train.loc[common_index]   
        y_final_train = y_final_train.loc[common_index]   
      
        X_final_train = X_final_train.fillna(X_final_train.mean().fillna(0) ) 
        X_train = X_train.fillna(X_train.mean().fillna(0) )
        X_future = X_future.fillna(X_future.mean().fillna(0) )
        """
        if type(high_var) == type(None):
            variances = y_train.var()
            high_var = variances[variances > variance_threshold].index
        
        # Store results"""
        return X_train, X_final_train, X_future, y_train,y_final_train
    
    results = {
        'train_predictions': {},
        'test_predictions': {},
        'best_params': {},
        'future_pred': {},
        'best_value': {},
    }
        
    
    # Optimize and predict for each target column
    for col in features_above_canritos:
        print(f"Optimizing for column: {col}")
        X_train, X_final_train, X_future, y_train,y_final_train = get_data(col, df_train, df_final_train, df_y_train, df_y_final_train,df_X_future)
        future_pred, best_params,best_value = optimize_and_predict(X_train, X_final_train, X_future, y_train,y_final_train,   15)
        
        #results['train_predictions'][col] = train_pred
        #results['test_predictions'][col] = test_pred
        print( 'best_value', best_value)
        results['best_value'][col] = best_value
        results['best_params'][col] = best_params
        results['future_pred'][col] = future_pred
    
    # Convert predictions to dataframes
    #train_predictions_df = pd.DataFrame(results['train_predictions'], index=X_train.index)
    #test_predictions_df = pd.DataFrame(results['test_predictions'], index=X_test.index)
    future_predictions_df = pd.DataFrame(results['future_pred'], index=X_future.index)
    #future_predictions_df['foto_mes'] = df_future['foto_mes'].to_pandas()
    #future_predictions_df['numero_de_cliente'] = df_future['numero_de_cliente'].to_pandas()
    future_predictions_df= future_predictions_df.add_suffix('_elstic2')
    future_predictions_pl = pl.from_pandas(future_predictions_df)

    df_future_updated = pl.concat([df_X_future, future_predictions_pl], how='horizontal')   
 
    return df_future_updated 

def add_forecast_elasticnet( data,features_above_canritos):
    subsample_ratio=0.01
    variance_threshold= 1E2
    all_times= data['foto_mes'].unique().to_list()
    all_times. reverse()
    stride=2
    window=2
    
    res={}
    for w_fut_test in all_times[:-stride-window] :
        print(w_fut_test)
        res[w_fut_test]={}
        i_w_fut_test = all_times.index(w_fut_test)
        i_final_train_end = i_w_fut_test + stride
        i_final_train_start = i_final_train_end + window        
        finaL_train = all_times[i_final_train_end: i_final_train_start]
        y_final_train = w_fut_test
        
        X_future= all_times[i_w_fut_test: i_w_fut_test+window]
        
        i_X_train_end = i_final_train_end + 1
        i_X_train_start = i_X_train_end + window
        X_train = all_times[i_X_train_end: i_X_train_start]
        y_train = all_times[i_X_train_end-2]
               
       
        res[w_fut_test][ 'finaL_train'] = finaL_train
        res[w_fut_test][ 'y_final_train'] = y_final_train
        res[w_fut_test][ 'X_future'] = X_future
        res[w_fut_test][ 'X_train'] = X_train
        res[w_fut_test][ 'y_train'] = y_train
       
        print (res[w_fut_test])
        
    data_fut2 =[]    
    #high_var=None
    for mes_future in res.keys():    
        wres=  res[mes_future]
       
        df_future_updated = forecast_numeric(features_above_canritos, data, features_above_canritos, wres,subsample_ratio, variance_threshold)
        data_fut2.append( df_future_updated)
    
    data_fut2 = pl.concat( data_fut2, how='vertical')
    return data_fut2








def get_top_and_least_important_boruta( data, ganancia_acierto,  mes_train, mes_test  ):
      
    
    try:   
        data = data.with_columns(pl.col('Master_Finiciomora').cast(pl.Float64))    
    except Exception as e :
        pass
    
    
    #df_train_3 = data.filter(pl.col('foto_mes') == mes_train)
    df_train_3 = data.filter(pl.col('foto_mes').is_in(mes_train))
    df_train_3= subsample_data_time_polars(df_train_3, 0.05, 'CONTINUA', 'clase_ternaria', 42)      
    df_test = data.filter(pl.col('foto_mes') == mes_test)
    
    y_train = df_train_3['clase_ternaria'].to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    y_test = df_test['clase_ternaria'].to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    
    X_train = df_train_3.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
    X_test = df_test.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
   
   
    rf = RandomForestClassifier(n_jobs=-1, class_weight={0:1, 1:ganancia_acierto}, max_depth=5)
   
    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=2)
    
    # find all relevant features - 5 features should be selected
    feat_selector.fit(X_train.fillna(0), y_train)   
    
 
    feature_names = X_train.columns
    feature_importance_df_ranking = pd.DataFrame({'feature': feature_names, 'importance_split': feat_selector.ranking_}).sort_values(by='importance_split', ascending=True)
    feature_importance_df_bool = pd.DataFrame({'feature': feature_names, 'importance_split': feat_selector.support_}).sort_values(by='importance_split', ascending=False)
     

    return feature_importance_df_ranking, feature_importance_df_bool
    


def get_top_and_least_important_boruta_res_mean( data, ganancia_acierto,  mes_train, mes_test, res_mean  ):
      
    
    try:   
        data = data.with_columns(pl.col('Master_Finiciomora').cast(pl.Float64))    
    except Exception as e :
        pass
    
    try:
        res_mean_series = pl.Series("res_mean", res_mean)
        data = data.filter(res_mean_series > 0.0125)
    except Exception as e:
        res_mean_series = pl.Series("res_mean", res_mean[0])
        data = data.filter(res_mean_series > 0.0125)
    df_train_3 = data.filter(pl.col('foto_mes').is_in(mes_train))
    #df_train_3 = data.filter(pl.col('foto_mes') == mes_train)
    df_train_3= subsample_data_time_polars(df_train_3, 0.2, 'CONTINUA', 'clase_ternaria', 42)      
    print('len entremaiento en boruta', len(df_train_3))
    df_test = data.filter(pl.col('foto_mes') == mes_test)
    
    y_train = df_train_3['clase_ternaria'].to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    y_test = df_test['clase_ternaria'].to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    
    X_train = df_train_3.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
    X_test = df_test.drop(['clase_ternaria', 'foto_mes', 'numero_de_cliente']).to_pandas()
   
   
    rf = RandomForestClassifier(n_jobs=-1, class_weight={0:1, 1:ganancia_acierto}, max_depth=5)
   
    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=2)
    
    # find all relevant features - 5 features should be selected
    feat_selector.fit(X_train.fillna(0), y_train)   
    
 
    feature_names = X_train.columns
    feature_importance_df_ranking = pd.DataFrame({'feature': feature_names, 'importance_split': feat_selector.ranking_}).sort_values(by='importance_split', ascending=True)
    feature_importance_df_bool = pd.DataFrame({'feature': feature_names, 'importance_split': feat_selector.support_}).sort_values(by='importance_split', ascending=False)
     

    return feature_importance_df_ranking, feature_importance_df_bool



def get_top_and_least_important_y_canaritos( data, N_top, N_least, N_least_ampliado,  mes_train, mes_test  ):
      
    data = data.with_columns(pl.lit(1.0).alias('clase_peso'))
    num_vectors = int( data.shape[1] * .2)
    len_rnd =   data.shape[0]
    
    pd_caos=  create_random_normal(num_vectors, len_rnd ) # Seteado en 20%
    
   
    data = pl.concat([data, pd_caos], how='horizontal')
    data = data.with_columns(
    pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(2.00002)
    .when(pl.col('clase_ternaria') == 'BAJA+1').then(2.00001)
    .otherwise(pl.col('clase_peso'))
    .alias('clase_peso')
    )
    data = data.with_columns(pl.col('Master_Finiciomora').cast(pl.Float64))    
    
    df_train_3 = data.filter(pl.col('foto_mes') == mes_train)
    df_test = data.filter(pl.col('foto_mes') == mes_test)
    
    y_train = df_train_3['clase_ternaria'].to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    y_test = df_test['clase_ternaria'].to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    
    X_train = df_train_3.drop(['clase_ternaria', 'foto_mes','clase_peso', 'numero_de_cliente']).to_pandas()
    X_test = df_test.drop(['clase_ternaria', 'foto_mes','clase_peso', 'numero_de_cliente']).to_pandas()
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    w_train = df_train_3['clase_peso'].to_pandas()
    w_test = df_test['clase_peso'].to_pandas()

    params_basicos = {
    'objective': 'binary',
    #'metric': 'custom',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'verbosity': -100,
    'seed': 123,
    'max_depth': -1,
    'min_gain_to_split': 0.0,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'max_bin': 31,
    'num_iterations': 9999,
    'force_row_wise': True,
    'learning_rate': 0.065,
    'feature_fraction': 1.0,
    'min_data_in_leaf': 50,
    'num_leaves': 120,
    'num_threads': -1
    }    
    
    """
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    test_data = lgb.Dataset(X_test, label=y_test, weight=w_test, reference=train_data)
    weights_backup = test_data.get_weight().copy()
    def lgb_gan_eval(y_pred, test_data):
        #weight = data.get_weight()
    # Reassign weights if needed
        #test_data.set_weight(weights_backup)
        weight = weights_backup
    
        #weight = test_data.weight.copy()
        ganancia = np.where(weight == 2.00002, ganancia_acierto, 0) - np.where(weight < 2.00002, costo_estimulo, 0)
        ganancia = ganancia[np.argsort(y_pred)[::-1]]
        ganancia = np.cumsum(ganancia)
    
        return 'gan_eval', np.max(ganancia) , True
    # Create LightGBM Datasets with weights
    
    callbacks = [lgb.early_stopping(stopping_rounds=200)]
    
    # Train the model with the custom metric and weights
    model = lgb.train(
        params_basicos,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'eval'],
        feval=lgb_gan_eval,
        callbacks=callbacks
    )

    
    
    """
    callbacks = [lgb.early_stopping(stopping_rounds=200)]    
    
    model = lgb.train(
    params_basicos,
    lgb_train,
    valid_sets=[lgb_train, lgb_test],
    valid_names=['train', 'eval'],
    callbacks=callbacks
    )
 
    importance_split = model.feature_importance(importance_type='split')
    importance_gain = model.feature_importance(importance_type='gain')    

    feature_names = X_train.columns
    feature_importance_df_split = pd.DataFrame({'feature': feature_names, 'importance_split': importance_split})
    feature_importance_df_gain = pd.DataFrame({'feature': feature_names, 'importance_gain': importance_gain})   
  
    pattern = r'canarito'
  

    feature_importance_df_gain = feature_importance_df_gain[feature_importance_df_gain['importance_gain'] > 0]
    feature_importance_df_gain = feature_importance_df_gain.sort_values(by='importance_gain', ascending=False)

    mask = feature_importance_df_gain['feature'].str.contains(pattern, regex=True)   
    feature_importance_df_gain.index= range(len(mask))
    matching_positions = feature_importance_df_gain.index[mask].tolist()
    mediana_canaritos = int(np.median(matching_positions ))
    features_above_canritos = list(feature_importance_df_gain['feature'])[: mediana_canaritos]
    features_above_canritos = [feature for feature in features_above_canritos if pattern not in feature]
    
    features_below_canritos = list(feature_importance_df_gain['feature'])[ mediana_canaritos: ]
    features_below_canritos = [feature for feature in features_below_canritos if pattern not in feature]      

    return features_above_canritos, features_below_canritos



def get_top_and_least_important( data, N_top, N_least, N_least_ampliado,  mes_train, mes_test  ):
    data = data.with_columns( pl.lit(1.0).alias('clase_peso')  )
    data = data.with_columns(
        pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(2.00002)
        .when(pl.col('clase_ternaria') == 'BAJA+1').then(2.00001)
        .otherwise(pl.col('clase_peso'))  # Keep the original value if no condition matches
        .alias('clase_peso')
    )
    data = data.with_columns(  pl.col('Master_Finiciomora').cast(pl.Float64)  )
    df_train_3 = data.filter(pl.col('foto_mes') == mes_train)
    df_test = data.filter(pl.col('foto_mes') == mes_test)
    
    y_train = df_train_3.with_columns( pl.when(pl.col('clase_ternaria') == 'CONTINUA').then(0) .otherwise(1).alias('y_train'))
    y_train = df_train_3['clase_ternaria']
    y_train = y_train.to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    
    y_test = df_test['clase_ternaria']
    y_test = y_test.to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    
   
    w_train = df_train_3['clase_peso']
    w_test = df_test['clase_peso']
    
 
    X_train = df_train_3.drop(['clase_ternaria', 'foto_mes', 'clase_peso'])
    X_test = df_test.drop(['clase_ternaria', 'foto_mes', 'clase_peso'])
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)    
    
    try:     
        lgb_train = lgb.Dataset(X_train,
                              label=y_train, # eligir la clase
                              weight=w_train.values )
        
        lgb_test = lgb.Dataset(X_test,
                              label=y_test, # eligir la clase
                              weight=w_test.values)
    
    except Exception as e:
        lgb_train = lgb.Dataset(X_train.to_pandas(),
                              label=y_train, # eligir la clase
                              weight=w_train)
       
        lgb_test = lgb.Dataset(X_test.to_pandas(),
                              label=y_test, # eligir la clase
                              weight=w_test ,
                              reference=lgb_train )
    
    
    lgb_train = lgb.Dataset(X_train.to_pandas(), y_train)
    lgb_test = lgb.Dataset(X_test.to_pandas(), y_test, reference=lgb_train)
    
    
    params_basicos = {
        'objective': 'binary',
        'metric': 'custom',
        'boost_from_average': True,
        'feature_pre_filter': True,
        'verbosity': -1,
        'max_bin': 31,
        'num_iterations': 200,
        'force_row_wise': True,
        'seed': 378821,
        'learning_rate': 0.026746294124634,
        'num_leaves': 351,
        'feature_fraction': 0.665080004152842,
        'min_data_in_leaf': 2500
    }
    
    
    model = lgb.train(params_basicos, lgb_train, valid_sets=[lgb_test], feval=lgb_gan_eval)
    
    model = lgb.train(params_basicos, lgb_train, feval=lgb_gan_eval)
    model = lgb.train(params_basicos, lgb_train)

    importance = model.feature_importance(importance_type='split')
    feature_names = X_train.columns
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    top_15_features = feature_importance_df.head(N_top)
    least_15_features = feature_importance_df.tail(N_least)
    least_15_features = least_15_features['feature'].tolist()
    top_15_feature_names = top_15_features['feature'].tolist()
    
    least_ampliado = feature_importance_df.tail(N_least_ampliado)
    least_ampliado = least_ampliado['feature'].tolist()
    return top_15_feature_names , least_15_features, least_ampliado


def get_top_and_least_important( data, N_top, N_least, N_least_ampliado,  mes_train, mes_test  ):
    data = data.with_columns(  pl.col('Master_Finiciomora').cast(pl.Float64)  )
    df_train_3 = data.filter(pl.col('foto_mes') == mes_train)
    df_test = data.filter(pl.col('foto_mes') == mes_test)
    
    y_train = df_train_3.with_columns( pl.when(pl.col('clase_ternaria') == 'CONTINUA').then(0) .otherwise(1).alias('y_train'))
    y_train = df_train_3['clase_ternaria']
    y_train = y_train.to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    
    y_test = df_test['clase_ternaria']
    y_test = y_test.to_pandas().map(lambda x: 0 if x == "CONTINUA" else 1)
    
    
    X_train = df_train_3.drop(['clase_ternaria', 'foto_mes'])
    X_test = df_test.drop(['clase_ternaria', 'foto_mes'])
    
    lgb_train = lgb.Dataset(X_train.to_pandas(), y_train)
    lgb_test = lgb.Dataset(X_test.to_pandas(), y_test, reference=lgb_train)
    
    
    params_basicos = {
        'objective': 'binary',
        'metric': 'custom',
        'boost_from_average': True,
        'feature_pre_filter': True,
        'verbosity': -1,
        'max_bin': 31,
        'num_iterations': 200,
        'force_row_wise': True,
        'seed': 378821,
        'learning_rate': 0.026746294124634,
        'num_leaves': 351,
        'feature_fraction': 0.665080004152842,
        'min_data_in_leaf': 2500
    }
    
    model = lgb.train(params_basicos, lgb_train)

    importance = model.feature_importance(importance_type='split')
    feature_names = X_train.columns
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    top_15_features = feature_importance_df.head(N_top)
    least_15_features = feature_importance_df.tail(N_least)
    least_15_features = least_15_features['feature'].tolist()
    top_15_feature_names = top_15_features['feature'].tolist()
    
    least_ampliado = feature_importance_df.tail(N_least_ampliado)
    least_ampliado = least_ampliado['feature'].tolist()
    return top_15_feature_names , least_15_features, least_ampliado



def div_sum_top_features_pandas ( data, top_15_feature_names):
    data_top = data[top_15_feature_names]
    data_top = data_top.add_suffix('_std')
    scaler = StandardScaler()
    scaled= scaler.fit_transform(data_top)
    data_top[:]= scaled
    columns_top= data_top.columns
    for i in columns_top:
        for k in  columns_top:
            if i!=k:
               data_top[i+'_sum_'+k]= data_top[i] + data_top[k]
               data_top[i+'_div_'+k]= data_top[i] / data_top[k]

    data_x = pd.concat( [data, data_top.astype('float32')], axis=1)
    return data_x


def div_sum_top_features_polars(data, top_15_feature_names):
    # Select and standardize the top features
    data_top = data.select(top_15_feature_names)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_top.to_numpy())
    
    # Create a DataFrame from the standardized data with "_std" suffix
    data_top_std = pl.DataFrame(
        {f"{col}_std": scaled[:, idx] for idx, col in enumerate(top_15_feature_names)}
    )

    # Create sum and division combinations, without returning `_std` columns
    new_columns = []
    """ for i in data_top_std.columns:
        for k in data_top_std.columns:
            if i != k:
                new_columns.extend([
                    (pl.col(i)* np.cos(np.pi/4) - np.sin(np.pi/4)*pl.col(k)  ).alias(f"{i}_rot45c1_{k}"),
                    (pl.col(i)* np.sin(np.pi/4) + np.cos(np.pi/4)*pl.col(k)  ).alias(f"{i}_rot45c2_{k}"),
                    (pl.col(i) - pl.col(k)).alias(f"{i}_minus_{k}"),
                    (pl.col(i) + pl.col(k)).alias(f"{i}_sum_{k}"),
                    (pl.col(i) / pl.col(k)).alias(f"{i}_div_{k}")
                ])
            """
    #for i in data_top_std.columns:
    #    for k in data_top_std.columns:
    for i in data_top_std.columns:
        i_pos = data_top_std.columns.index(i)
        for k in data_top_std.columns[i_pos:]:
            if i != k:
               
                # Perform the operations and cast the results to float32
                new_columns.extend([
                    # Rotation component 1 (rot45c1)
                    (pl.col(i)* np.cos(np.pi/4) - np.sin(np.pi/4)*pl.col(k)).cast(pl.Float32).alias(f"{i}_rot45c1_{k}"),
                    
                    # Rotation component 2 (rot45c2)
                    (pl.col(i)* np.sin(np.pi/4) + np.cos(np.pi/4)*pl.col(k)).cast(pl.Float32).alias(f"{i}_rot45c2_{k}"),
                    
                    # Difference (minus)
                    #(pl.col(i) - pl.col(k)).cast(pl.Float32).alias(f"{i}_minus_{k}"),
                    
                    # Sum
                    #(pl.col(i) + pl.col(k)).cast(pl.Float32).alias(f"{i}_sum_{k}"),
                    
                    # Division
                    (pl.col(i) / pl.col(k)).cast(pl.Float32).alias(f"{i}_div_{k}"),
                    
                   # (pl.col(i) * pl.col(k)).cast(pl.Float32).alias(f"{i}_mult_{k}")
                ])
    # Create a DataFrame with just the new columns
    data_calculations = data_top_std.select(new_columns)
    
    # Concatenate the new columns with the original DataFrame, excluding `_std` columns
    data_x = data.hstack(data_calculations)

    return data_x

    
def drop_columns_nan_zero(data, threshold, original_columns):
    row_count = data.height
    # Filter numeric columns only
    numeric_cols = [col for col in data.columns if ( data[col].dtype.is_numeric() and col not in original_columns)]
    
    # Identify columns to drop based on the threshold
    cols_to_drop = [
        col for col in numeric_cols 
        if (data.select(pl.col(col).is_null().sum() + (pl.col(col) == 0).sum()).item() / row_count) > threshold
    ]
    
    # Drop the identified columns
    return data.drop(cols_to_drop)

def print_nan_columns(data, threshold,original_columns):
    row_count = data.height
    # Filter numeric columns only
    numeric_cols = [col for col in data.columns if ( data[col].dtype.is_numeric() and col not in original_columns)]
    
    # Identify columns to drop based on the threshold
    cols_to_drop = [
        col for col in numeric_cols 
        if (data.select(pl.col(col).is_null().sum() + (pl.col(col) == 0).sum()).item() / row_count) > threshold
    ]
    
    # Drop the identified columns
    print(len(cols_to_drop))

"""
data_w = pd.read_parquet( path_set_con_ternaria)
grouped_data = [(numero_cliente, group) for numero_cliente, group in data_w.groupby("numero_de_cliente")]

a= grouped_data[0]



for i in grouped_data:
    a2 = i[1]
    if a2['numero_de_cliente'].iloc[0] not in [249420051,249988447]:
        if 6>len (  a2['clase_ternaria'].to_list())>3:
            if 'BAJA+2' in a2['clase_ternaria'].to_list():
                break
"""


        
        
def crete_ternaria( path_set_crudo, path_set_con_ternaria):
    df= pd.read_csv( path_set_crudo)    
    meses = df['foto_mes'].unique()
    def create_batches(meses):   
        meses_sorted = sorted(meses, reverse=False)    
    
        lotes = [(meses_sorted[i], meses_sorted[i + 1], meses_sorted[i + 2]) for i in range(len(meses_sorted) - 2)]
        
        return lotes
    
    lotes = create_batches(meses)
    
    for mes_anterior, mes_actual, mes_siguiente in lotes:
        clientes_mes_actual = df[df['foto_mes'] == mes_actual]['numero_de_cliente']
        clientes_mes_siguiente = df[df['foto_mes'] == mes_siguiente]['numero_de_cliente']
    
        #df.loc[(df['foto_mes'] == mes_anterior) & (~df['numero_de_cliente'].isin(clientes_mes_actual)) & (~df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'BAJA+1'
        df.loc[(df['foto_mes'] == mes_anterior) & (~df['numero_de_cliente'].isin(clientes_mes_actual)), 'clase_ternaria'] = 'BAJA+1'
        df.loc[(df['foto_mes'] == mes_anterior) & (df['numero_de_cliente'].isin(clientes_mes_actual)) & (~df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'BAJA+2'
        df.loc[(df['foto_mes'] == mes_anterior) & (df['numero_de_cliente'].isin(clientes_mes_actual)) & (df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'CONTINUA'
    
    
    indx = df['foto_mes'].isin(meses[2:])
    df.loc[indx & df['clase_ternaria'].isna(), 'clase_ternaria'] = 'CONTINUA'

    df.to_parquet( path_set_con_ternaria )

def print_nan_columns(data, threshold,original_columns):
    row_count = data.height
    # Filter numeric columns only
    numeric_cols = [col for col in data.columns if ( data[col].dtype.is_numeric() and col not in original_columns)]
    
    # Identify columns to drop based on the threshold
    cols_to_drop = [
        col for col in numeric_cols 
        if (data.select(pl.col(col).is_null().sum() + (pl.col(col) == 0).sum()).item() / row_count) > threshold
    ]
    
    # Drop the identified columns
    print(len(cols_to_drop))
    
    numeric_cols = [col for col in data.columns if ( data[col].dtype.is_numeric() )]
    
    # Identify columns to drop based on the threshold
    cols_to_drop = [
        col for col in numeric_cols 
        if (data.select(pl.col(col).is_null().sum() + (pl.col(col) == 0).sum()).item() / row_count) > threshold
    ]
   
   
    print(len(cols_to_drop))



#
#data_w =data.iloc[:165000]
def create_data(ganancia_acierto,  path_set_crudo, path_set_con_ternaria,  mes_train, mes_test ):
    if not os.path.exists(path_set_con_ternaria):
        crete_ternaria( path_set_crudo, path_set_con_ternaria)     
        
    data = pl.read_parquet(path_set_con_ternaria)    
    #data= pl.from_pandas(data)
    #data = data.rename({'': 'indicex'})
    #data = data.drop('indicex')
    data.columns = [col.replace(r'[^a-zA-Z0-9_]', '_') for col in data.columns]
    data = data.with_columns(  pl.col('Master_Finiciomora').cast(pl.Float64)  )
    data = data.with_columns(  pl.col('tmobile_app').cast(pl.Float32)  )
    data = data.with_columns(  pl.col('cmobile_app_trx').cast(pl.Float32)  ) 
   
    data= AgregarVariables_IntraMes(data)
    original_columns_inta_mes= data.columns
    
    
    
    #data= pl.from_pandas(data)
    
    data = percentage_binning(data)    
    
    data = time_features(data)
    
    data = add_forecast(data.to_pandas())
    n_lags= 2
    lag_flag=True
    delta_lag_flag= False
    data = add_lags_diff(data, lag_flag, n_lags,  delta_lag_flag )
    
    
  
    data= replace_infinities_with_limits(data)    
    data= convert_to_int_float32_polars(data)
    
    feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta( data, ganancia_acierto,  mes_train, mes_test  )
    joblib.dump( [feature_importance_df_ranking, feature_importance_df_bool], exp_folder + 'first_boruta.joblib')
    feature_importance_df_ranking, feature_importance_df_bool = joblib.load( exp_folder + 'first_boruta.joblib')
    #baja_importancia = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==False ]['feature'].to_list()    
    features_finales = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==True ]['feature'].to_list()    
    data= data[['numero_de_cliente','foto_mes','clase_ternaria']+ features_finales]
    data = div_sum_top_features_polars(data, feature_importance_df_ranking['feature'][:50].to_list())
   
    data= convert_to_int_float32_polars(data)
  
    data= replace_infinities_with_limits(data)    
    
    #feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta_res_mean( data,ganancia_acierto,  mes_train, mes_test,res_mean  )
    #baja_importancia = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==False ]['feature'].to_list()       
    #data, new_features = enhanced_feature_binning(data, baja_importancia, N_bins)
    
    

    feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta( data, ganancia_acierto,  mes_train, mes_test  )
    #feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta_res_mean( data,ganancia_acierto,  mes_train, mes_test, res_mean  )
    features_finales = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==True ]['feature'].to_list()    
    joblib.dump( [feature_importance_df_ranking, feature_importance_df_bool], exp_folder + 'second_boruta.joblib')
    #joblib.dump( [feature_importance_df_ranking, feature_importance_df_bool], exp_folder + 'second_boruta.joblib')
    #feature_importance_df_ranking, feature_importance_df_bool = joblib.load( exp_folder + 'second_boruta.joblib')
    
    data= data[['numero_de_cliente','foto_mes','clase_ternaria']+ features_finales]
    
    data= add_moth_encode( data)
    data.write_parquet( desktop_folder+'data_x.parquet')

    return original_columns_inta_mes,  data,  features_finales, feature_importance_df_ranking, feature_importance_df_bool




# fin preparacion de datos.
    

########################################################

#/////////////////////////////////////////
#/////////////////////////////////////////
#/////////////////////////////////////////#/////////////////////////////////////////
#/////////////////////////////////////////

#/////////////////////////////////////////
#/////////////////////////////////////////
#/////////////////////////////////////////#/////////////////////////////////////////
#/////////////////////////////////////////

def subsample_continua(df_train, foto_mes_train, foto_mes_test, fraction):
    df_train_3 = df_train[df_train['foto_mes'] == foto_mes_train]
    df_continua = df_train_3[df_train_3['clase_ternaria'] == 'CONTINUA']
    df_continua_sampled = df_continua.sample(frac=fraction, random_state=42)
    df_other_classes = df_train_3[df_train_3['clase_ternaria'] != 'CONTINUA']
    df_train_sampled = pd.concat([df_other_classes, df_continua_sampled])
    
    df_test = df_train[df_train['foto_mes'] == foto_mes_test]
    
    y_train = df_train_sampled["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)
    y_test = df_test["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)
    
    return df_train_sampled, df_test, y_train, y_test
#df_train_sampled, df_test, y_train, y_test = subsample_continua(df_train, 202103, 202105, 0.5)


def subsample_data_time(df, fraction, target_class, target_column, random_state):
    df_target = df[df[target_column] == target_class]
    df_other = df[df[target_column] != target_class]
    #random_state = int(time())
    df_target_sampled = df_target.sample(frac=fraction, random_state=random_state) 
    df_sampled = pd.concat([df_other, df_target_sampled])    
    return df_sampled
def subsample_data_time_polars(df: pl.DataFrame, fraction: float, target_class: str, target_column: str, random_state: int) -> pl.DataFrame:
    # Filter target and other classes
    df_target = df.filter(pl.col(target_column) == target_class)
    df_other = df.filter(pl.col(target_column) != target_class)

    # Sample the target class with a random seed
    df_target_sampled = df_target.sample(n=int(len(df_target) * fraction), seed=random_state)

    # Concatenate the other class with the sampled target class
    df_sampled = pl.concat([df_other, df_target_sampled])

    return df_sampled

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
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_pred_lgm[piso_envios:techo_envios], ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
    plt.title('Curva de Ganancia')
    plt.xlabel('Predicci�n de probabilidad')
    plt.ylabel('Ganancia')
    plt.axvline(x=optimal_threshold, color='g', linestyle='--', label='Punto de corte a 0.025')
    plt.legend()
    plt.show()    
    """       
    ganancia_max = ganancia_cum.max()
    gan_max_idx = np.where(ganancia_cum == ganancia_max)[0][0]
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(piso_envios, len(ganancia_cum[piso_envios:techo_envios]) + piso_envios), ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
    plt.axvline(x=gan_max_idx, color='g', linestyle='--', label=f'Punto de corte a la ganancia m�xima {gan_max_idx}')
    plt.axhline(y=ganancia_max, color='r', linestyle='--', label=f'Ganancia m�xima {ganancia_max}')
    plt.title('Curva de Ganancia')
    plt.xlabel('Clientes')
    plt.ylabel('Ganancia')
    plt.legend()
    plt.show()
    """
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

def fit_predict(trains , mes_test,data_x, params, min_data_in_leaf, cant_semillas_ensamble, trial_number) :
  
    cant_exp = 1
     
    random_state= trial_number
    clase_peso_lgbm = 1
    final_selection= None
        
    res= []
    res_train= []
    start= time.time()  
    for i in range(0, cant_exp):
        print('en  for i in range(0, cant_exp): i=', i)
        random_state+= i
        params['data_random_seed'] =  random_state
        random_numbers= generate_random_list_numpy(random_state, size=cant_semillas_ensamble, min_val=1, max_val=32767)
                         
        train_data, X_test, y_test= create_LGBM_dataset_basic(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params)
        assert np.sum(y_test.to_numpy().reshape(-1) ) !=0
        times= []
        for rnd in random_numbers:            
            print('trial', trial_number, ' ensamble n: ', random_numbers.index(rnd))         
            params['bagging_seed'] = rnd
            params['feature_fraction_seed'] = rnd+1
            params['seed'] =   rnd+5                       
        
            y_test_pred = exectue_model_basic(  train_data,X_test , params, min_data_in_leaf )
            print( 'exectue_model hs = ', (time.time()  - start)/60/60)
            res.append( y_test_pred)
            #res_train.append( y_train_pred)
            times . append(  time.time() -start)
           
    elapsed_time =  np.min( times)
   
    res =np.max( res, axis=0)
    return res, y_test
    
    

def generate_random_list_numpy(seed, size, min_val, max_val ):
    """Generate random integers using NumPy"""
    np.random.seed(seed)
    return np.random.randint(min_val, max_val , size=size).tolist()
 

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
    (data_x_selected["clase_ternaria"].isin(['BAJA+2', 'BAJA+1'])) )
    
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

    


def create_LGBM_datasetCV(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params, mode_stacking, mode_FM, th, n_folds):
    
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
    (data_x_selected["res_mean"] > th) | 
    (data_x_selected["clase_ternaria"].isin(['BAJA+2', 'BAJA+1'])) )
    
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
    
        
    X_train = X_train.drop(['clase_ternaria','clase_peso'])
    
   
                 
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
   
    X_test = X_test.drop(['clase_ternaria','clase_peso'])
    
    
    
    train_data = lgb.Dataset( X_train.to_pandas(),
                         label=y_train.to_pandas(),
                         weight=w_train.to_pandas(), params=params)    
    
    return train_data, X_test.to_pandas(), y_test.to_pandas() , X_train.to_pandas(), y_train.to_pandas() 


def create_LGBM_dataset_basic(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params):
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
    (data_x_selected["clase_ternaria"].isin(['BAJA+2', 'BAJA+1'])) )
    
    
    resample = data_x_selected[
                           (data_x_selected["clase_ternaria"] == 'CONTINUA')]
    #pesos = np.exp(resample['res_mean']+0.01)
    chosen_rows = resample.sample(n=int(len(resample)*params['neg_bagging_fraction']),  random_state= params['seed'])
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
   
    X_train = df_train_3
  
          
    X_train = X_train.drop(['clase_ternaria', 'clase_peso'])
    
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
   
    X_test = df_test   
    X_test = X_test.drop(['clase_ternaria', 'clase_peso'])
    
    assert np.sum(y_train.to_numpy().reshape(-1) ) !=0
    assert np.sum(y_test.to_numpy().reshape(-1) ) !=0
    
    return train_data, X_test.to_pandas(), y_test.to_pandas()
   




#final_selection,trains, mes_test, data_x, fraction, params,trial_number,feature_selection = objective_params(mock_trial)    
def exectue_model(  train_data,X_test , params, trial_number, min_data_in_leaf ,exp_folder,random_state, X_train):  
    
    params['min_data_in_leaf'] = int( len(train_data.get_label())  * min_data_in_leaf )                                  
      
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_test_pred = model.predict(X_test)       
    
    y_train_pred = model.predict(X_train)     
    file_path = os.path.join(exp_folder, 'comp_2_dict.joblib')
    
    if os.path.exists(file_path) :
       res_dict = joblib.load(file_path)
       if trial_number not in res_dict.keys():
           res_dict[trial_number]= {}
       res_dict[trial_number][random_state] = y_test_pred
       joblib.dump(res_dict, file_path)       
    else:
       res_dict = {}
       res_dict[trial_number] = {}
       res_dict[trial_number][random_state] = y_test_pred
       joblib.dump(res_dict, file_path)           
    #print(res_dict)
    return y_test_pred, y_train_pred


def lgb_fn_eval(preds, train_data):
    labels = train_data.get_label()
    
    # Apply a threshold of 0.5 by default (you can adjust it later for fine-tuning)
    preds_binary = (preds > 0.5).astype(int)

    # Compute False Negatives: True Positive = 1, False Negative = 0, True Negative = 0, False Positive = 1
    false_negatives = np.sum((labels == 1) & (preds_binary == 0))
    
    # You can also return other metrics (e.g., accuracy, AUC) if needed
    return 'false_negatives', false_negatives, False  # False means it should be minimized



def exectue_model_basic(  train_data,X_test , params, min_data_in_leaf ):  
    
    params['min_data_in_leaf'] = int( len(train_data.get_label())  * min_data_in_leaf )                                  
    #print( 'WARNING MINIMIZING FALSE NEGATIVES')  
    model = lgb.train(params, train_data, feval=lgb_gan_eval) #lgb_gan_eval
    feature_importance = model.feature_importance(importance_type='split')  # You can use 'split' or 'gain'
    feature_names = model.feature_name()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)   
   
    print(importance_df)
    y_test_pred = model.predict(X_test)           
   
    return y_test_pred


def data_classification(data_x, suggest_params,iter_limit, cant_exp,  seed):   
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
    del params['min_data_in_leaf']
            
    cant_semillas_ensamble = int( iter_limit/params['num_iterations']/cant_exp)        
    
    clase_peso_lgbm = 1
       
    random_state= seed   
    #res_final= []
    #res_future= []
    start= time.time()  
    
    futures= []
    gan= []
    final_train= list( data_x['foto_mes'].unique() )
    for mes_future in final_train[2:]:
        if suggest_params != None:
            if mes_future!= final_train[-3]:
                continue
        i_mes = final_train.index(mes_future)
        trains= final_train[: i_mes-1]
        print(trains )
        print(mes_future )
        y_pred_future, y_true  = fit_predict(trains , mes_future, data_x, params, min_data_in_leaf,  cant_semillas_ensamble, int(mes_future))
        if suggest_params != None:
            res_mean = y_pred_future
            y_test = y_true
            for threshold in thresholds:
                y_pred_threshold = (res_mean >= threshold).astype(int)
                # Calculate false negatives
                false_negs = ((y_pred_threshold == 0) & ( y_test.to_numpy().reshape(-1) == 1)).sum()
                false_neg_counts.append(false_negs)
                droped_data.append(np.sum( (res_mean < threshold).astype(int) ))
          
            fig, ax1 = plt.subplots(figsize=(10, 6))            
            # Plot the first data (false_neg_counts) on the primary y-axis
            ax1.plot(thresholds, false_neg_counts, 'b-', linewidth=2, label='False Negatives')
            ax1.fill_between(thresholds, false_neg_counts, color='blue', alpha=0.2)
            ax1.set_xlabel('Probability Threshold')
            ax1.set_ylabel('Number of False Negatives', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xlim(0, 1.1)
            ax1.set_ylim(0, max(false_neg_counts) * 1.1)
            
            # Create the second y-axis sharing the same x-axis
            ax2 = ax1.twinx()
            ax2.plot(thresholds, droped_data, 'r-', linewidth=2, label='Dropped Data')
            ax2.fill_between(thresholds, droped_data, color='red', alpha=0.2)
            ax2.set_ylabel('Dropped Data', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, max(droped_data) * 0.1)            
            # Add grid and title
            ax1.grid(True, linestyle='--', alpha=0.7)
            fig.suptitle('Comparison of False Negatives and Dropped Data vs Probability Threshold')           
            # Tight layout
            plt.tight_layout()
            plt.show()
            
            return lgb_gan_eval(y_pred_future, y_true, ganancia_acierto, costo_estimulo)[1]
        
        #mes_future  = 202103
        
        filtered_data = data_x.filter(pl.col('foto_mes') == mes_future).select('clase_ternaria')        
        y_test = filtered_data.with_columns(
            pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(1).otherwise(0).alias('mapped_clase_ternaria')
        )['mapped_clase_ternaria']

        gan.append ( lgb_gan_eval(y_pred_future.reshape(-1), y_test, ganancia_acierto, costo_estimulo) )
        futures.append(y_pred_future)
    
    futures_array = np.hstack(futures). reshape(-1) 
    data_x_pd = data_x.to_pandas()
    data_x_pd['future_predictions']  = np.nan
    data_x_pd.loc[data_x_pd['foto_mes'].isin( final_train[2:]), 'future_predictions' ]= futures_array
    return      pl.from_pandas(data_x_pd) , gan
   
    
    thresholds = np.linspace(0, 1.1, 100)
    false_neg_counts = []
    droped_data=[]
    mes= 202104
    y_test = data_x_pd.loc[data_x_pd['foto_mes'] == mes, 'clase_ternaria'].map(lambda x: 1 if x == 'BAJA+2' else 0)
    res_mean =  data_x_pd.loc[data_x_pd['foto_mes']==mes, 'res_mean' ]
    
    res_mean = y_pred_future
    y_test = y_true
    for threshold in thresholds:
        y_pred_threshold = (res_mean >= threshold).astype(int)
        # Calculate false negatives
        false_negs = ((y_pred_threshold == 0) & ( y_test.to_numpy().reshape(-1) == 1)).sum()
        false_neg_counts.append(false_negs)
        droped_data.append(np.sum( (res_mean < threshold).astype(int) ))
  
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the first data (false_neg_counts) on the primary y-axis
    ax1.plot(thresholds, false_neg_counts, 'b-', linewidth=2, label='False Negatives')
    ax1.fill_between(thresholds, false_neg_counts, color='blue', alpha=0.2)
    ax1.set_xlabel('Probability Threshold')
    ax1.set_ylabel('Number of False Negatives', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, max(false_neg_counts) * 1.1)
    
    # Create the second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(thresholds, droped_data, 'r-', linewidth=2, label='Dropped Data')
    ax2.fill_between(thresholds, droped_data, color='red', alpha=0.2)
    ax2.set_ylabel('Dropped Data', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, max(droped_data) * 0.1)
    
    # Add grid and title
    ax1.grid(True, linestyle='--', alpha=0.7)
    fig.suptitle('Comparison of False Negatives and Dropped Data vs Probability Threshold')
    
    # Tight layout
    plt.tight_layout()
    plt.show()
    
    
    
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the first data (false_neg_counts) on the primary y-axis
    ax1.plot(thresholds, false_neg_counts, 'b-', linewidth=2, label='False Negatives')
    ax1.fill_between(thresholds, false_neg_counts, color='blue', alpha=0.2)
    ax1.set_xlabel('Probability Threshold')
    ax1.set_ylabel('Number of False Negatives', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlim(0, 0.2)
    ax1.set_ylim(0, max(false_neg_counts) * 1.1)
    
    # Create the second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(thresholds, droped_data, 'r-', linewidth=2, label='Dropped Data')
    ax2.fill_between(thresholds, droped_data, color='red', alpha=0.2)
    ax2.set_ylabel('Dropped Data', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, max(droped_data) * 1.1)
    
    # Add grid and title
    ax1.grid(True, linestyle='--', alpha=0.7)
    fig.suptitle('Comparison of False Negatives and Dropped Data vs Probability Threshold')
    
    # Tight layout
    plt.tight_layout()
    plt.show()
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, false_neg_counts, 'b-', linewidth=2)
    plt.fill_between(thresholds, false_neg_counts, alpha=0.2)
    plt.xlabel('Probability Threshold')
    plt.ylabel('Number of False Negatives')
    plt.title('False Negatives vs Probability Threshold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, max(false_neg_counts) * 1.1)
    plt.tight_layout()
    plt.show()"""
       
    max_threshold= find_threshold_by_cumsum(res_mean, np.sum(y_train.to_pandas().values == 1)*10)
  
    auc_false_neg, auc_true_neg = calculate_false_negatives_auc(res_mean, y_train, max_threshold)
    
    
    cv_pred_fut_pred= np.hstack( [res_mean, res_future_mean])
    if mode_production:
        return data_x.with_columns(pl.Series('res_mean', cv_pred_fut_pred)), cv_pred_fut_pred
    else:
        return auc_false_neg, auc_true_neg 
    


def objective_data_classification(trial,data_w):
   
    params= {}
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.25, 1)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.05, 0.9)
    params['num_leaves'] = trial.suggest_int("num_leaves", 8, 500)
    params['num_iterations'] = trial.suggest_int("num_iterations", 1, 50)
    params['min_data_in_leaf'] = trial.suggest_float("min_data_in_leaf",  1.5E-05, 0.008)  # Example of leaf size        
    params['neg_bagging_fraction'] =  trial.suggest_float("neg_bagging_fraction", 0.02, 1)  
    params['pos_bagging_fraction'] = 1
    params['bagging_fraction'] = trial.suggest_float("bagging_fraction", 0.1, 1)
    #print(f"data_w: {data_w}")
    
    trial_number = trial.number   
    cant_exp= 1#trial.suggest_int("cant_exp", 1, 20)
    iter_limit= 300
    
    
    seed = trial_number
    start= time.time()                      
    suggest_params = params
    ganancia= data_classification(data_w, suggest_params,iter_limit, cant_exp,  seed)
    elapsed_time=  time.time() - start 
   
    return  ganancia ,elapsed_time #- len(feature_selection )*penalty, time

def optuna_objective_data_classification():
    if os.path.exists(exp_folder+nombre_exp_study):
        study= joblib.load(exp_folder+nombre_exp_study )
    else: 
        data_w= pl.read_parquet( '/home/reinaldomedina_robledo/Documents/competencia_01_mini.parquet')
        #study = optuna.create_study(direction="maximize")
        #study = optuna.create_study(direction="minimize")
        study = optuna.create_study( directions=[StudyDirection.MINIMIZE,StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] ) #, timeout=60*60*2  )
        study = optuna.create_study( directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] ) #, timeout=60*60*2  )
        #study = optuna.create_study( directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] ) #, timeout=60*60*2  )

    for i in range(0, 40):
        #study.optimize(objective, n_trials=1)  # You can specify the number of trials
        study.optimize(lambda trial: objective_data_classification(trial, data_w), n_trials=1, n_jobs=-1)
        sdf =study.trials_dataframe()
        study.trials[10]
        #study.optimize(objective_data_classification, n_trials=2, n_jobs=-1)
        joblib.dump( study, exp_folder+ nombre_exp_study)     
    """ 3 finished with values: [63.21171213214609, 30514.22332104275, 1037.7145915031433] and parameters: {'learning_rate': 0.8207775171619601, 'feature_fraction': 0.23640639561437643, 'num_leaves': 402, 'num_iterations': 29, 'min_data_in_leaf': 0.005473464714277627, 'neg_bagging_fraction': 0.1931051708608652, 'bagging_fraction': 0.193374290064174}.
    5 finished with values: [39.999552374643855, 24161.56068348571, 965.3486881256104] and parameters: {'learning_rate': 0.5750344086417372, 'feature_fraction': 0.1356758876690755, 'num_leaves': 111, 'num_iterations': 33, 'min_data_in_leaf': 0.005414715141237464, 'neg_bagging_fraction': 0.17228894720624818, 'bagging_fraction': 0.45431199818328394}.
    """


#####################################
#####################################
##########################################################################
#####################################
#####################################
#####################################




import numpy as np
from scipy import stats

def calculate_rolling_features(df, window=2, target_columns=None):
  
    # Create copy to avoid modifying original
    result_df = df.copy()
    matrix = result_df[numeric_features].fillna(0).values
   
    for col in numeric_features:
        # Create time index for slope calculation
        time_idx = np.array(range(window))
        
        # Calculate rolling slope
        def calc_slope(values):
            if len(values) != window:
                return np.nan
            slope, _, _, _, _ = stats.linregress(time_idx, values)
            return slope
        
        slope_name = f'{col}_slope_w{window}'
        result_df[slope_name] = df[col].rolling(window=window).apply(calc_slope)
        
        # Calculate linear extrapolation (next predicted value)
        def calc_extrapolation(values):
            if len(values) != window:
                return np.nan
            slope, intercept, _, _, _ = stats.linregress(time_idx, values)
            # Predict next value (at time_idx + 1)
            return slope * window + intercept
            
        extrap_name = f'{col}_extrap_w{window}'
        result_df[extrap_name] = df[col].rolling(window=window).apply(calc_extrapolation)
    
    return result_df

# Example usage:
"""
# Assuming group_df is sorted by foto_mes and reset_index is already done
group_df = calculate_rolling_features(
    group_df,
    window=2,
    target_columns=['column1', 'column2']  # Specify your columns here
)
"""
def rolling_forecast_pandas(group_df):
    #numeric_features = [col for col in group_df.columns if col not in ["numero_cliente", "foto_mes"]]
    """numeric_features= [
'mrentabilidad', 'mrentabilidad_annual', 'mactivos_margen', 'mpasivos_margen', 'mcuenta_corriente', 'mcaja_ahorro', 'mcuentas_saldo', 'mcaja_ahorro_dolares', 'mtarjeta_visa_consumo', 'mtarjeta_master_consumo', 'cpayroll_trx',
'mpayroll', 'mpayroll2', 'cpayroll2_trx', 'ccuenta_debitos_automaticos', 'mcuenta_debitos_automaticos', 'ctarjeta_visa_debitos_automaticos', 'mttarjeta_visa_debitos_automaticos', 'ctarjeta_master_debitos_automaticos', 'mttarjeta_master_debitos_automaticos', 'cpagodeservicios', 'mpagodeservicios', 'cpagomiscuentas', 'mpagomiscuentas' ,'mcomisiones_otras' , 'chomebanking_transacciones', 'ctrx_quarter' ,
 'Master_msaldototal', 'Master_msaldopesos', 'Master_msaldodolares' ,'Master_mconsumospesos', 'Master_mconsumosdolares', 'Master_mlimitecompra', 'Master_madelantopesos', 'Master_madelantodolares', 'Master_fultimo_cierre', 'Master_mpagado', 'Master_mpagospesos',
 'Visa_msaldototal', 'Visa_msaldopesos', 'Visa_msaldodolares', 'Visa_mconsumospesos', 'Visa_mconsumosdolares', 'Visa_mlimitecompra', 'Visa_madelantopesos']
    """
    numeric_features= list( set(group_df.columns)- set( ['foto_mes', 'numero_de_cliente', 'clase_ternaria'] ) )
    
    
    group_df = group_df.sort_values("foto_mes").reset_index(drop=True)   
    #print(group_df)
    #res= group_df
    #res[::]= np.nan    
    if len(group_df)>1:               
        matrix = group_df[numeric_features].fillna(0).values
        res= []
        
        for i in range(1, len(group_df) ):
            coef = np.polyfit([1,2], matrix[i-1:i+1,:],1, rcond=None, full=False, w=None, cov=False)
            forecast = coef[0] * 4 + coef[1]
            res.append(forecast)            
        res= np.vstack(res)
        res= pd.DataFrame(res , columns= [ col+'_forcst' for col in numeric_features ], index =group_df.index[1:] )
        res_f = pd.concat([ group_df, res], axis =1)
        #print(res_f )
        return res_f
    return  pd.DataFrame( columns= [ col+'_forcst' for col in numeric_features ], index =group_df.index[1:] )



"""grouped_data = [(numero_cliente, group) for numero_cliente, group in data_w.groupby("numero_de_cliente")]
res = []
for g in grouped_data:
    res.append( rolling_forecast_pandas(g[1])  )
df_parallel_result = pd.concat(res).reset_index(drop=True)
 
 """


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd

def process_group(group):
    return  rolling_forecast_pandas(group[1]) 

def add_forecast( data):
    grouped_data = [(numero_cliente, group) for numero_cliente, group in data.groupby("numero_de_cliente")]
    
    # Define a function to handle each group
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Wrap the parallel process with tqdm for a progress bar
        results2 = list(tqdm(executor.map(process_group, grouped_data), total=len(grouped_data)))
    
    # Concatenate the results into a single DataFrame
    for i in range(0,)
    df_parallel_result = pd.concat(results2).reset_index(drop=True)
    """
    
    # Convert pandas DataFrames to Polars DataFrames
    polar_results = [pl.from_pandas(df) for df in results]
    
    # Concatenate the list of Polars DataFrames
    df_parallel_result = pl.concat(results)"""
    return df_parallel_result




feature_importance_df_ranking = ['Master_mconsumosdolares_rank_rolling_6_forcst', 'vmr_mpagospesos_roll_3_diff_6_forcst', 'ctarjeta_visa_descuentos_rank_rolling_3','vm_mconsumototal_rank_roll_2_diff_3_forcst'	,'mtarjeta_visa_descuentos_rolling_6_forcst','minversion1_pesos',
'mcaja_ahorro_rank_rolling_3_forcst', 'vm_mconsumototal_rank_roll_2_diff_3'	,'vm_cconsumos_rank_rolling_6', 'vm_cconsumos_rank_rolling_3', 'cproductos_forcst','cextraccion_autoservicio_rank_rolling_3_forcst','vm_mpagado_rank_rolling_3_forcst',
'ccajeros_propios_descuentos_rank_roll_2_diff_3',	'mcajeros_propios_descuentos_rank_rolling_6', 'mcajeros_propios_descuentos_rank_rolling_3',	'mcajeros_propios_descuentos_rank_roll_3_diff_6','mcajeros_propios_descuentos_rank_roll_2_diff_3',
'cpayroll_trx_rolling_3', 'Visa_mconsumospesos_rank_rolling_3_forcst', 'cinversion2_rolling_3','cinversion2_roll_3_diff_6', 'cinversion2_roll_2_diff_3', 'mtarjeta_visa_descuentos_rank_rolling_3',	'mtarjeta_visa_descuentos_rank_roll_2_diff_3',
'Master_mconsumototal_rank_rolling_3_forcst', 	'mtarjeta_visa_descuentos_rolling_3_forcst', 	'mpagodeservicios_rolling_3_forcst', 	'Master_madelantopesos_rank_rolling_3_forcst', 'catm_trx_other_roll_2_diff_3_forcst', 
'vm_mlimitecompra_roll_2_diff_3'	,'vm_mpagado_roll_2_diff_3'	,'vm_cconsumos_rank_roll_2_diff_3', 'Master_cadelantosefectivo_rank_rolling_3_forcst', 	'Master_msaldopesos_rank_rolling_6_forcst', 'cprestamos_personales_rank_roll_2_diff_3_forcst',
'mcajeros_propios_descuentos_rolling_6_forcst', 'mpayroll_sobre_edad_rank_rolling_6_forcst', 'ctarjeta_master_descuentos_rank_roll_3_diff_6_forcst', 	'mpasivos_margen_rolling_3_forcst', 'Visa_mconsumosdolares_rank_rolling_6', 
'Visa_mpagado_forcst', 'cinversion2', 'minversion2', 'cpayroll_trx', 'ccuenta_debitos_automaticos', 'ccajeros_propios_descuentos_rank_roll_3_diff_6_forcst', 'ccheques_emitidos_forcst', 'vm_mconsumototal_rank_rolling_6', 'vm_mconsumototal_rank_rolling_3',
'vm_mconsumospesos_rank_rolling_6', 'mpagomiscuentas_rank_rolling_3_forcst'	,'mpagomiscuentas', 'ccajeros_propios_descuentos', 'mcajeros_propios_descuentos', 	'mtarjeta_visa_descuentos', 'ctarjeta_master_descuentos', 'ctarjeta_visa_transacciones_rolling_3_forcst',
'vm_cconsumos', 'vmr_mpagosdolares_rank_rolling_6']



"""



a= grouped_data[0]

for i in grouped_data:
    a2 = i[1]
    if 6>len (  a2['clase_ternaria'].to_list())>3:
        if 'BAJA+2' in a2['clase_ternaria'].to_list():
            break

# Use multiprocessing to process groups in parallel

df_parallel_result

"""
"""

# Apply rolling forecast group by numero_cliente

df_result = data_w.to_pandas().groupby("numero_de_cliente").apply(rolling_forecast_pandas).reset_index(drop=True)

df_result = data_w.groupby("numero_de_cliente").apply(rolling_forecast_pandas).reset_index(drop=True)








df_result = df.groupby("numero_cliente").apply(rolling_forecast_pandas).reset_index(drop=True)

df_result

df_result"""






data =pd.read_parquet( path_set_con_ternaria)
X = data.drop(columns= ['clase_ternaria'])
y = data ['clase_ternaria']
y = data ['clase_ternaria'].map({'CONTINUA':0, 'BAJA+1':0, 'BAJA+2':1})




train_data = lgb.Dataset( X, label=y )
                      model = 

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()

model = lgb.LGBMClassifier(**params, feval=lgb_gan_eval)
model.fit(X ,y.values)

y_pred = cross_val_predict( model , X, y, cv=3)

 model = lgb.train(params, train_data, feval=lgb_gan_eval) #lgb_gan_eval

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
      "num_iterations": 10,   # Large number, controlled by early stopping
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
  
    params['min_data_in_leaf'] = int( len(train_data.get_label())  * min_data_in_leaf )                                  
      
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_test_pred = model.predict(X_test)       
    
    y_train_pred = model.predict(X_train)     
    file_path = os.path.join(exp_folder, 'comp_2_dict.joblib')
    
    if os.path.exists(file_path) :
       res_dict = joblib.load(file_path)
       if trial_number not in res_dict.keys():
           res_dict[trial_number]= {}
       res_dict[trial_number][random_state] = y_test_pred
       joblib.dump(res_dict, file_path)       
    else:
       res_dict = {}
       res_dict[trial_number] = {}
       res_dict[trial_number][random_state] = y_test_pred
       joblib.dump(res_dict, file_path)           
    #print(res_dict)
    return y_test_pred, y_train_pred
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class CustomLightGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, params, n_models=20):
        self.params = params
        self.n_models = n_models
        self.models = []

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_models):
            model = lgb.LGBMClassifier(**self.params)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])  # Shape: (n_models, n_samples)
        sorted_preds = np.sort(preds.T, axis=1)  # Sort each row (for each sample)
        return sorted_preds

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters for LightGBM
lgb_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 5}

# Base model
custom_lgbm = CustomLightGBM(params=params, n_models=20)

# Stacking classifier
stack_clf = StackingClassifier(
    estimators=[('custom_lgbm', custom_lgbm)],
    final_estimator=LogisticRegression(),
    stack_method='predict'  # Use sorted predictions as input for meta-model
)

# Fit and evaluate
stack_clf.fit(X_train, y_train)
print(f"Accuracy: {stack_clf.score(X_test, y_test):.4f}")




ds()

#path_set_con_ternaria = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01.csv'
path_set_crudo = '/home/reinaldo/Downloads/competencia_03_crudo.csv.gz' 
path_set_crudo = '/home/reinaldomedina_robledo/Desktop/competencia_03_crudo.csv.gz' 
#data= pl.read_csv( path_set_con_ternaria)
desktop_folder = '/home/reinaldomedina_robledo/Desktop/'
#path_set_con_ternaria = '/home/reinaldomedina_robledo/Documents/exp6/competencia_02_crudo.csv'
path_set_con_ternaria = '/home/reinaldomedina_robledo/Desktop/competencia_03_ternaria.parquet'

exp_folder = '/home/reinaldomedina_robledo/buckets/b2/exp/escopeta_3_1/'

ganancia_acierto = 273000
costo_estimulo = 7000
path_set_crudo= path_set_con_ternaria
boruta= True
#mes_train, mes_test = 202102, 202104
mes_train_boruta, mes_test_boruta = [202101, 202102, 202104], 202106
mes_train, mes_test = mes_train_boruta, mes_test_boruta 

original_columns_inta_mes,  data_x,  features_finales, feature_importance_df_ranking, feature_importance_df_bool =  create_data (ganancia_acierto,  path_set_crudo, path_set_con_ternaria,  mes_train, mes_test , res_mean)



data_w_f = slopes_lin_forecast(data_w, 3)









N_bins=5
data_w = mini_create_data(ganancia_acierto, path_set_crudo, path_set_con_ternaria,  mes_train, mes_test,boruta)
#data_w= pl.read_parquet( '/home/reinaldomedina_robledo/Documents/competencia_01_mini.parquet')
#data_w.write_parquet( '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01_mini.parquet')
#data_w.write_parquet( '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01_mini.parquet')
#data_w.write_parquet( '/home/reinaldomedina_robledo/Documents/competencia_01_mini.parquet')
#   data_w= pl.read_parquet( '/home/reinaldomedina_robledo/Documents/competencia_01_mini.parquet')
#   data_w= pl.read_parquet( '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01_mini.parquet')
# data_w= pd.read_csv( path_set_con_ternaria)
cant_exp= 1
iter_limit= 800
n_folds= 2
mode_cv=True
mode_production= True
trains= None
suggest_params=None
seed = 42
mes_train, mes_test = [202101,202102, 202103, 202104 ], 202106
mes_train, mes_test = [202101, 202104 ], 202106
data_slim_mean ,res_mean = data_classification(data_w, mode_cv, mode_production ,trains, suggest_params,iter_limit, cant_exp, n_folds, seed)

joblib.dump( [ res_mean ], exp_folder+ 'res_mean_0.joblib')
res_mean= joblib.load( exp_folder+ 'res_mean_0.joblib')
original_columns_inta_mes,  data,  features_finales, feature_importance_df_ranking, feature_importance_df_bool =   create_data(ganancia_acierto, path_set_crudo, path_set_con_ternaria,  mes_train, mes_test , res_mean, N_bins)

desktop_folder = '/home/reinaldomedina_robledo/Documents/'

joblib.dump( [ original_columns_inta_mes,   features_finales, feature_importance_df_ranking, feature_importance_df_bool ], exp_folder+ 'aacc1.joblib')
data.write_parquet(desktop_folder+'data_x.parquet' )







data_x = pl.read_parquet( desktop_folder+ 'data_x.parquet')


"""


original_columns,original_columns_inta_mes,  data_x,  features_finales, feature_importance_df_ranking, feature_importance_df_bool, new_features =create_data(ganancia_acierto, last_date_to_consider, path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_bins,lag_flag, delta_lag_flag)
original_columns,original_columns_inta_mes,  data_x,  features_finales, feature_importance_df_ranking, feature_importance_df_bool, new_features = mini_create_data(ganancia_acierto, last_date_to_consider, path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_bins,lag_flag, delta_lag_flag)


joblib.dump( [ original_columns,original_columns_inta_mes, features_finales, feature_importance_df_ranking, feature_importance_df_bool, new_features], exp_folder+ 'aacc1.joblib')
data_x.write_parquet(desktop_folder+'data_x.parquet' )
"""
ds()

exp_folder = '/home/reinaldomedina_robledo/buckets/b2/exp/escopeta_5_k2/'
desktop_folder = '/home/reinaldomedina_robledo/Documents/exp5/'
data_x = pl.read_parquet(desktop_folder+'data_x.parquet')
#original_columns_inta_mes,  features_finales, feature_importance_df_ranking, feature_importance_df_bool   = joblib.load(  exp_folder+ 'aacc1.joblib')
res_mean= joblib.load( exp_folder+ 'res_mean_0.joblib')
data_x= data_x.with_columns(pl.Series('res_mean', res_mean[0]))
nombre_exp_study = 'study_MiniLGBM_corr.joblib'

mes_test=  202108

competition_name = 'dm-ey-f-2024-segunda'
#competition_name = 'dm-ey-f-2024-primera'  


kaggle_mode = True

ganancia_acierto = 273000
costo_estimulo = 7000

#trains= [202004,202005,202006,202007,202008,202009,202010,202011,202012,202101,202102, 202103, 202104]
#final_train = [202006,202007,202008,202009,202010,202011,202012,202101,202102, 202103, 202104, 202105, 202106]
final_train=[
    202106, 202105, 202104, 202103, 202102, 202101, 
    202012, 202011, 202010, 202009, 202008, 202007, 
    # 202006  Excluyo por variables rotas
    202005, 202004, 202003, 202002, 202001,
    201912, 201911,
    # 201910 Excluyo por variables rotas
    201909, 201908, 201907, 201906,
    # 201905  Excluyo por variables rotas
    201904, 201903
  ]
final_train_full =[
    202106, 202105, 202104, 202103, 202102, 202101, 
    202012, 202011, 202010, 202009, 202008, 202007, 
     202006, #  Excluyo por variables rotas
    202005, 202004, 202003, 202002, 202001,
    201912, 201911,
     201910,# Excluyo por variables rotas
    201909, 201908, 201907, 201906,
     201905,#  Excluyo por variables rotas
    201904, 201903
  ]
#final_train = [202011,202012,202101,202102, 202103, 202104]


#trains= [ 202103, 202104]
#trains= [202011,202012,202101,202102, 202103, 202104]
#trains= [202102, 202103, 202104]
#final_train = [202006,202007,202008,202009,202010,202011,202012,202101,202102, 202103, 202104, 202105, 202106]

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
      "max_bin": 65,            # Maximum number of bins
      #"num_iterations": 9999,   # Large number, controlled by early stopping
      #"bagging_fraction": 1.0,  # Fraction of data used for bagging
      #"pos_bagging_fraction": 1.0,  # Fraction of positive data used for bagging
      #"neg_bagging_fraction": 1.0,  # Fraction of negative data used for bagging
      "is_unbalance": False,    # Do not balance the classes
      "scale_pos_weight": 1.0,  # Weighting for positive class
      #"drop_rate": 0.1,         # Drop rate for DART (if used)
      #"max_drop": 50,           # Maximum number of drops for DART
      #"skip_drop": 0.5,         # Probability of skipping a drop for DART
      "extra_trees": False,     # Disable extra trees
  }
  






#top_15_feature_names= features_finales[:50]
#top_15_feature_names= feature_importance_df_ranking['feature'][:50]



from optuna.samplers import TPESampler
n_exploration=50222222


def objective(trial):
    #global best_result, best_predictions, penalty, top_15_feature_names, data,random_state, trains,mes_test
    
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.05, 1)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.05, 0.9)
    params['num_leaves'] = trial.suggest_int("num_leaves", 8, 500)
    params['max_bin'] = trial.suggest_int("max_bin", 31, 100)
    params['num_iterations'] = trial.suggest_int("num_iterations", 1, 200)
    min_data_in_leaf = trial.suggest_float("min_data_in_leaf",  1.5E-05, 0.004)  # Example of leaf size    
    
    params['neg_bagging_fraction'] =  trial.suggest_float("neg_bagging_fraction", 0.0, 0.5)  
    params['pos_bagging_fraction'] = 1
    params['bagging_fraction'] = trial.suggest_float("bagging_fraction", 0.1, 1)
    th = trial.suggest_float("th", 0.0001, 0.1)
    mode_stacking= trial.suggest_categorical("mode_stacking", [True, False])
    mode_FM = trial.suggest_categorical("mode_FM", [True, False])
    
    #if 
    trial_number = trial.number
    
    clase_peso_lgbm = trial.suggest_float('clase_peso_lgbm',1, 5)   
    #max_semillas = int( min( 500, 50+ trial_number*4 ))
    
    
    if trial_number<n_exploration:
        #cant_exp= trial.suggest_int("cant_exp", 1, 1)
        iter_limit= trial.suggest_int("iter_limit", 1000, 1500) #600
        
        cant_semillas_ensamble = int( iter_limit/params['num_iterations'])        
       
        #n_envios = [12000]
    if trial_number >=n_exploration:
        iter_limit= 2000 #trial.suggest_int("iter_limit", 20000, 40000)
        #iter_limit = 4000
        #cant_exp= trial.suggest_int("cant_exp", 1, 4)
        cant_semillas_ensamble = int( iter_limit/params['num_iterations'])        
        #n_envios = [11000, 11500, 12000, 12500, 13000, 13500]
    
    
    n_envios = [trial.suggest_int("n_envios", 10000, 12000)]
    cant_exp = 1
    #cant_semillas_ensamble = trial.suggest_int('cant_semillas_ensamble',50, 51)   
    print( 'cant_semillas_ensamble ', cant_semillas_ensamble)
#    fraction = 0.1# trial.suggest_float('fraction', 0.01, 1)             
    #fraction  = trial.suggest_float('fracccion', 0.01, 0.2)   
    #cantidad_meses = trial.suggest_int('cantidad_meses', 10, 11)   
    
    #if trial_number>60 :
   # params['max_bin']  = trial.suggest_int("max_bin",  31 , 32)  
    
    
    """ 
    params['learning_rate'] = 0.3
    params['feature_fraction'] = 0.8
    params['num_leaves'] = 7
    min_data_in_leaf = 0.002
    params['num_iterations'] = 5
    params['num_iterations'] = 5
    clase_peso_lgbm =  ganancia_acierto+1000
    cant_semillas_ensamble =5
#    fraction = 0.1# trial.suggest_float('fraction', 0.01, 1)             
    fraction  = 0.25
    cantidad_meses = 2
    trial_number=33
    params['max_bin'] = 31
    mode_stacking= False
    mode_FM = False
    th=0
    
    """
    
    random_state= trial_number
    #trains = final_train #[-cantidad_meses:]
    trains = final_train[2:]
    if trial.suggest_categorical("Full_trian", [True, False]):
        trains = final_train_full[2:]
    
       
 
    #random_numbers= generate_random_list_numpy(random_state, size=cant_semillas_ensamble, min_val=1, max_val=32767)
    #random_numbers = np.random.randint(low=1, high=32767, size=cant_semillas_ensamble, dtype=np.int16).tolist()
    final_selection= None
    
    
    res= []
    res_train= []
    start= time.time()  
    for i in range(0, cant_exp):
        print('en  for i in range(0, cant_exp): i=', i)
        random_state+= i
        params['data_random_seed'] =  random_state
        random_numbers= generate_random_list_numpy(random_state, size=cant_semillas_ensamble, min_val=1, max_val=32767)
        #train_data, X_test, y_test = create_LGBM_dataset(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params)
        train_data, X_test, y_test, X_train, y_train = create_LGBM_dataset(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params, mode_stacking, mode_FM, th)
        times= []
        for rnd in random_numbers:
            
            print('trial', trial_number, ' ensamble n: ', random_numbers.index(rnd))
            #buenos_aires_time = datetime.now(pytz.timezone('America/Argentina/Buenos_Aires'))
            #print(buenos_aires_time.strftime('%Y-%m-%d %H:%M:%S'))
            params['bagging_seed'] = rnd
            params['feature_fraction_seed'] = rnd+1
            params['seed'] =   rnd+5
            #train_data, X_test, y_test = create_LGBM_dataset(final_selection, trains, mes_test, data_x, clase_peso_lgbm,params)
            #params['data_random_seed'] =   rnd+3
            
            w_start_exc= time.time()  
            params['neg_bagging_fraction'] =1    
            y_test_pred, y_train_pred =  exectue_model(  train_data, X_test , params, trial_number, min_data_in_leaf ,exp_folder,rnd ,X_train)
            print( 'exectue_model hs = ', (time.time()  - w_start_exc)/60/60)
            res.append( y_test_pred)
            res_train.append( y_train_pred)
            times . append(  time.time() -start)
            if not kaggle_mode:
                welapsed_time =  time.time() -start
                welapsed_time= welapsed_time/60/60
                if welapsed_time >1:
                    elapsed_time =  time.time() -start
                   
                    res =np.max( res, axis=0)
                    res_train =np.max( res_train, axis=0)
                    print( 'Avg Gan en Train',  lgb_gan_eval(res_train, y_train, ganancia_acierto, costo_estimulo)[1] / len(trains))    
                    res0 = lgb_gan_eval(res, y_test, ganancia_acierto, costo_estimulo)[1]  
                    return res0 ,elapsed_time  * cant_semillas_ensamble/random_numbers.index(rnd)#- len(feature_selection )*penalty, time
                
    elapsed_time =  np.min( times)
   
    res =np.max( res, axis=0)
    res_train =np.max( res_train, axis=0)
    joblib.dump( res, exp_folder+'rs.joblib')
    joblib.dump( y_test, exp_folder+'y_test.joblib')
    #y_red = joblib.load(exp_folder+'rs.joblib')
    #y_true = joblib.load(exp_folder+'y_test.joblib')
    if kaggle_mode:
        kagle_res= [ ]
        for env in n_envios: 
            submission_path = to_kaggle_file (env, res, exp_folder , X_test,trial_number)  #to_kaggle_file (n_envios, y_future, exp_folder , X_future, trial_number) :
            print( submission_path)
            res0 = get_kaggle_score( submission_path, competition_name  )
            kagle_res.append( res0)
        return np.max(kagle_res), elapsed_time
    
    #to_kaggle_file (n_envios, y_test_pred, exp_folder , X_test, trial_number)
    print( 'Avg Gan en Train',  lgb_gan_eval(res_train, y_train, ganancia_acierto, costo_estimulo)[1] / len(trains))    
    print( 'Gan Max',  lgb_gan_eval(y_test, y_test, ganancia_acierto, costo_estimulo) )    
    res0 = lgb_gan_eval(res, y_test, ganancia_acierto, costo_estimulo)[1]  
    return res0 ,elapsed_time #- len(feature_selection )*penalty, time


ds()
if os.path.exists(exp_folder+nombre_exp_study):
    study= joblib.load(exp_folder+nombre_exp_study )
else: 
    #study = optuna.create_study(direction="maximize")
    #study = optuna.create_study(direction=["maximize", "minimize"])
    study = optuna.create_study( directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] ) #, timeout=60*60*2  )

for i in range(0, 50):
    #study.optimize(objective, n_trials=1)  # You can specify the number of trials
    s_start= time.time()  
    study.optimize(objective, n_trials=1, n_jobs=-1)
    s_lapse= (time.time() - s_start)/60/60
    print( 'timepo_iter= ', s_lapse)
    joblib.dump( study, exp_folder+ nombre_exp_study)     
    


                   
"""

sdf= study.trials_dataframe()


trials_to_keep = [ t for t in study.trials if t.number <=2 ]
new_study =optuna.create_study( directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE] )
new_study.add_trials(trials_to_keep)
"""


ds()

#study= joblib.load( '/home/a_reinaldomedina/Documents/comp1_study2_rank.joblib')
#study = optuna.create_study(direction="maximize")
optuna.visualization.plot_optimization_history(study).show(renderer="browser")
#optuna.visualization.plot_intermediate_values(study).show(renderer="browser")

optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0]).show(renderer="browser")

