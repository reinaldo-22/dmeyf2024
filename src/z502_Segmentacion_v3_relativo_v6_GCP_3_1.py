
import pandas as pd
import numpy as np
#import seaborn as sns
#pip install polars
#from umap import UMAP
import matplotlib.pyplot as plt
#from sklearn.cluster import DBSCAN
#from sklearn.ensemble import  RandomForestClassifier
#from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import dask.dataframe as dd

import time

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt

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



def add_lags_diff(data, lag_flag, delta_lag_flag ):
    campitos = ["numero_de_cliente", "foto_mes", "clase_ternaria"]
    numeric_cols_lagueables = [
        col for col in data.columns if col not in campitos and data.schema[col] in (pl.Int32, pl.Int64, pl.Float32, pl.Float64)
    ]
    
    # Sort data by 'numero_de_cliente' and 'foto_mes'
    data = data.sort(["numero_de_cliente", "foto_mes"])
    
    
    for col in numeric_cols_lagueables:
        data = data.with_columns(
            pl.col(col).shift(1).over("numero_de_cliente").alias(f"{col}_lag1")
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
def time_features( df_train):
    w_df_train = df_train.sort_values(by=['numero_de_cliente', 'foto_mes'])    
    threshold = 0.1 * len(w_df_train)    
   
    for feature in w_df_train.columns:
        print(feature)
        if feature  in ['numero_de_cliente', 'foto_mes','clase_ternaria']:
            continue
        if not((w_df_train[feature].ne(0).sum() + w_df_train[feature].notna().sum()) > threshold):
            continue
        w_df_train[ feature+'_rolling_3'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.rolling(window=3).mean()).astype('float32')
        w_df_train[ feature+'_rolling_2'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.rolling(window=2).mean()).astype('float32')
        w_df_train[ feature+'_roll_3_diff_1'] = w_df_train.groupby('numero_de_cliente')[feature+'_rolling_3'].fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
        w_df_train[ feature+'_roll_2_diff_1'] = w_df_train.groupby('numero_de_cliente')[feature+'_rolling_3'].fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
        w_df_train[ feature+'_diff_1'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
    return w_df_train

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


def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 2.00002, ganancia_acierto, 0) - np.where(weight < 2.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True



def add_features_manual( data):
    data = data.with_columns((pl.col('mcuenta_corriente') + pl.col('mcaja_ahorro')).alias('m_suma_CA_CC'))
    data = data.with_columns((pl.col('Master_mconsumospesos') + pl.col('Visa_mconsumospesos')).alias('Tarjetas_consumos_pesos'))
    data = data.with_columns((pl.col('Master_mconsumosdolares') + pl.col('Visa_msaldodolares')).alias('Tarjetas_consumos_colares'))
    data = data.with_columns((pl.col('mcuentas_saldo')/ pl.col('cliente_edad')).alias('saldo/edad'))
    return data



def add_moth_encode( data):
    data = data.with_columns(
        [
            (pl.col('foto_mes') % 100).alias('month'),  # Extract month
            (np.sin((pl.col('foto_mes') % 100) * (2 * np.pi / 12))).alias('month_sin'),  # Sine encoding
            (np.cos((pl.col('foto_mes') % 100) * (2 * np.pi / 12))).alias('month_cos')   # Cosine encoding
        ]
    )
    return data

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
    for i in data_top_std.columns:
        for k in data_top_std.columns:
            if i != k:
                new_columns.extend([
                    (pl.col(i) + pl.col(k)).alias(f"{i}_sum_{k}"),
                    (pl.col(i) / pl.col(k)).alias(f"{i}_div_{k}")
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
    
        df.loc[(df['foto_mes'] == mes_anterior) & (~df['numero_de_cliente'].isin(clientes_mes_actual)) & (~df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'BAJA+1'
        df.loc[(df['foto_mes'] == mes_anterior) & (df['numero_de_cliente'].isin(clientes_mes_actual)) & (~df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'BAJA+2'
        df.loc[(df['foto_mes'] == mes_anterior) & (df['numero_de_cliente'].isin(clientes_mes_actual)) & (df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'CONTINUA'
    
    
    indx = df['foto_mes'].isin(meses[2:])
    df.loc[indx & df['clase_ternaria'].isna(), 'clase_ternaria'] = 'CONTINUA'

    df.to_csv( path_set_con_ternaria )




def create_data(path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_least_ampliado, N_bins, lag_flag, delta_lag_flag):
    if not os.path.exists(path_set_con_ternaria):
        crete_ternaria( path_set_crudo, path_set_con_ternaria)     
        
    data = pl.read_csv(path_set_con_ternaria)    
    data = data.rename({'': 'indicex'})
    data = data.drop('indicex')
    data.columns = [col.replace(r'[^a-zA-Z0-9_]', '_') for col in data.columns]
    data = data.with_columns(  pl.col('Master_Finiciomora').cast(pl.Float64)  )
    data = data.with_columns(  pl.col('tmobile_app').cast(pl.Float32)  )
    data = data.with_columns(  pl.col('cmobile_app_trx').cast(pl.Float32)  )

   
    original_columns= data.columns
    # data = data.iloc[:500000,:]
    # data= data.slice(0, 100000)    
    
    top_15_feature_names , least_15_features, least_ampliado=   get_top_and_least_important( data, N_top, N_least, N_least_ampliado,  mes_train, mes_test  )
    data= add_features_manual( data)
    data = div_sum_top_features_polars(data, top_15_feature_names)
    print_nan_columns(data, 0.75, original_columns)
    data= add_moth_encode( data)
    print_nan_columns(data, 0.75, original_columns)
    data= bins_least_importatn( data,least_15_features, N_bins ) 
    print_nan_columns(data, 0.75, original_columns)
    data= standardize_columns(data) 
    print_nan_columns(data, 0.75, original_columns)
    data= convert_to_int_float32_polars(data)
    #data = drop_columns_nan_zero(data, 0.75, original_columns)
    
    #data= add_lags_diff(data ) 
    data= add_lags_diff(data, lag_flag, delta_lag_flag )
    return original_columns, data,  top_15_feature_names , least_15_features, least_ampliado


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
    plt.xlabel('Predicción de probabilidad')
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
    plt.axvline(x=gan_max_idx, color='g', linestyle='--', label=f'Punto de corte a la ganancia máxima {gan_max_idx}')
    plt.axhline(y=ganancia_max, color='r', linestyle='--', label=f'Ganancia máxima {ganancia_max}')
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


#####################################
#####################################
##########################################################################
#####################################
#####################################
#####################################

params = {
    'boosting_type': 'gbdt',  # can be 'dart', 'rf', etc.
    'objective': 'binary',
    'metric': 'custom',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'force_row_wise': True,  # reduce warnings
    'verbosity': -100,
    'max_depth': -1,  # -1 means no limit
    'min_gain_to_split': 0.0,  # min_gain_to_split >= 0.0
    'min_sum_hessian_in_leaf': 0.001,  # min_sum_hessian_in_leaf >= 0.0
    'lambda_l1': 0.0,  # lambda_l1 >= 0.0
    'lambda_l2': 0.0,  # lambda_l2 >= 0.0
    'max_bin': 31,  # fixed, does not participate in BO

    'bagging_fraction': 1.0,  # 0.0 < bagging_fraction <= 1.0
    'pos_bagging_fraction': 1.0,  # 0.0 < pos_bagging_fraction <= 1.0
    'neg_bagging_fraction': 1.0,  # 0.0 < neg_bagging_fraction <= 1.0
    'is_unbalance': False,
    'scale_pos_weight': 1.0,  # scale_pos_weight > 0.0

    'drop_rate': 0.1,  # 0.0 < drop_rate <= 1.0
    'max_drop': 50,  # <= 0 means no limit
    'skip_drop': 0.5,  # 0.0 <= skip_drop <= 1.0

    'extra_trees': False,
   
}

#ds()
# !pip install polars
ganancia_acierto = 273000
costo_estimulo = 7000

N_top, N_least,  mes_train, mes_test = 15, 20, 202104, 202106
N_least_ampliado = 30
N_bins=7
# path_set_crudo = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01_crudo.csv'
# path_set_con_ternaria = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_02.csv'
#path_set_con_features_eng = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_02_features_eng.joblib'


path_set_crudo = "/home/a_reinaldomedina/buckets/b2/datasets/competencia_02_crudo.csv"
path_set_con_ternaria = "/home/a_reinaldomedina/buckets/b2/datasets/competencia_02.csv"

"""
if not os.path.exists(path_set_con_features_eng):
    original_columns, data_x,  top_15_feature_names , least_15_features, least_ampliado = create_data(path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_least_ampliado, N_bins)
    data_x.write_parquet(path_set_con_features_eng, compression='gzip') 
    
    
    joblib.dump([ original_columns,  top_15_feature_names , least_15_features, least_ampliado],  path_datos_accesorios)
    joblib.dump([ original_columns, data_x,  top_15_feature_names , least_15_features, least_ampliado],  path_datos_accesorios)
    data_x.write_parquet( path_set_con_features_eng )
    data_x.write_csv(path_set_con_features_eng, compression='gzip')
    joblib.dump([original_columns, data_x, top_15_feature_names, least_15_features, least_ampliado], 
            path_set_con_features_eng, compress=('zlib', 3))
else:
    original_columns, data_x,  top_15_feature_names , least_15_features, least_ampliado= joblib.load( path_set_con_features_eng)
    data_x = pl.read_parquet(path_set_con_features_eng)
"""
lag_flag, delta_lag_flag = False, True
original_columns, data_x,  top_15_feature_names , least_15_features, least_ampliado = create_data(path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_least_ampliado, N_bins,lag_flag, delta_lag_flag)
#data_x= data
leaks=[]
for col in data_x.columns:
    if 'clase_peso'in col.lower():
        leaks.append(col)

#data_x['clase_peso'] = 1.0
data_x = data_x.with_columns( pl.lit(1.0).alias('clase_peso')  )
data_x = data_x.with_columns(
    pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(2.00002)
    .when(pl.col('clase_ternaria') == 'BAJA+1').then(2.00001)
    .otherwise(pl.col('clase_peso'))  # Keep the original value if no condition matches
    .alias('clase_peso')
)



penalty=0
exp_folder = '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/'
#exp_folder = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/exp/Python_optuna1/'
#exp_folder = "~/buckets/b2/exp/comp2/"
nombre_exp_study = 'comp2_study2.joblib'
random_state=42
max_semillas= 1
trains= [202004,202005,202006,202007,202008,202009,202010,202011,202012,202101,202102, 202103, 202104]

test_date= 202106




def objective(trial):
    global best_result, best_predictions, penalty, top_15_feature_names, data,random_state, trains,test_date
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.3)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.01, 1.0)
    params['num_leaves'] = trial.suggest_int("num_leaves", 31, 256)  # Example of leaf size
    params['min_child_samples'] = trial.suggest_int("min_child_samples", 1, 100)  # Example of coverage
    params['seed'] =   int(time())
    params['max_depth'] = trial.suggest_int("max_depth", 3, 15)  # Control tree depth
    params['min_gain_to_split'] = trial.suggest_float("min_gain_to_split", 0.0, 1.0)  # Minimum gain to split
    #params['lambda_l1'] = trial.suggest_float("lambda_l1", 0.0, 10.0)  # L1 regularization
    #params['lambda_l2'] = trial.suggest_float("lambda_l2", 0.0, 10.0)  # L2 regularization
    params['num_iterations'] = trial.suggest_int("num_iterations", 50, 2500)  # Number of boosting iterations    
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.1, 1.0)   
   
    fraction = 0.1# trial.suggest_float('fraction', 0.01, 1)             

    woriginal_columns = list( set(original_columns) -{'clase_ternaria'})    
    trial_number= trial.number
    if trial_number>400:
        feature_selection=[]
        for  col in least_ampliado:
            w_col = trial.suggest_categorical(col, [True, False])
            if w_col:
                feature_selection.append(col)            
    else: 
        feature_selection = woriginal_columns
    
        
    columns = data_x.columns
    final_selection=[]
    for col in columns:
        if any(orig_col.lower() in col.lower() for orig_col in feature_selection):
            final_selection.append(col)
          
    
    final_selection= list( set(final_selection))
    
    final_selection = set(final_selection).union({'clase_ternaria', 'foto_mes', 'clase_peso'}) 
    final_selection = list( final_selection.union(top_15_feature_names) )
    
    
    random_state+=     trial_number
    #train_for_predict= [wt +2 for wt in trains]
    random.seed(random_state)
    #random_numbers = [random.random() for _ in range(max_semillas)]
    random_numbers = np.random.randint(low=-32768, high=32767, size=max_semillas, dtype=np.int16).tolist()
    res= []
    for rnd in random_numbers:
        w_res = exectue_model(final_selection,trains, test_date, data_x, fraction, params,trial_number,feature_selection,random_state)
        res.append( w_res)
    res =np.mean( res)
    return res- len(feature_selection )*penalty



#final_selection,trains, test_date, data_x, fraction, params,trial_number,feature_selection = objective_params(mock_trial)    
def exectue_model(final_selection,trains, test_date, data_x, fraction, params, trial_number, feature_selection, random_state):
    
    global best_result, best_predictions, penalty,exp_folder, mode_recalc
    data_x_selected= data_x[final_selection]
    #df_train_3 = data_x_selected[data_x_selected['foto_mes'].isin(trains)]  
    df_train_3 = data_x_selected.filter(pl.col('foto_mes').is_in(trains))
    #df_test = data_x_selected[data_x_selected['foto_mes'] == test_date]    
    df_test = data_x_selected.filter(pl.col('foto_mes') == test_date)
    """
    df_train_3= subsample_data_time_polars(df_train_3, fraction, 'CONTINUA', 'clase_ternaria', random_state)    
        
    y_train = df_train_3["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)    
    y_test = df_test["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)
    
    w_train = df_train_3['clase_peso']
    w_test = df_test['clase_peso']    
    
    X_train = df_train_3.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])    
    X_test = df_test.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])  
    """
   
    y_train = df_train_3.select(
        pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0)
        .otherwise(1)
        .alias("y_train")
    )
    
    y_test = df_test.select(
        pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0)
        .otherwise(1)
        .alias("y_test")
    )
    
    # Extract weights
    w_train = df_train_3['clase_peso']
    w_test = df_test['clase_peso']
    
    # Drop specified columns for X_train and X_test
    X_train = df_train_3.drop(['clase_ternaria', 'foto_mes', 'clase_peso'])
    X_test = df_test.drop(['clase_ternaria', 'foto_mes', 'clase_peso'])
    
    
    print( X_train.shape)
    #X_train = subsample_data_time(X_train,  fraction, target_class='CONTINUA')
    #X_pred = subsample_data_time(X_pred,  fraction, target_class='CONTINUA')  
        
    train_data = lgb.Dataset(X_train.to_pandas(),
                          label=y_train.to_pandas(), # eligir la clase
                          weight=w_train.to_pandas())
    test_data = lgb.Dataset(X_test.to_pandas(),
                          label=y_test.to_pandas(), # eligir la clase
                          weight=w_test.to_pandas())
    print(params)
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_test_pred = model.predict(X_test)
    res0= lgb_gan_eval(y_test_pred, test_data)[1]  
    
    file_path = os.path.join(exp_folder, 'comp_2_dict.joblib')
  
    if os.path.exists(file_path):
        res_dict = joblib.load(file_path)
        res_dict[trial_number] = y_test_pred
    else:
        res_dict = {}
        res_dict[trial_number] = y_test_pred
        joblib.dump(res_dict, file_path)
           
       
    return res0



if os.path.exists(exp_folder+nombre_exp_study):
    study= joblib.load(exp_folder+nombre_exp_study )
else: 
    study = optuna.create_study(direction="maximize")

for i in range(0, 3000):
    #study.optimize(objective, n_trials=1)  # You can specify the number of trials
    study.optimize(objective, n_trials=1, n_jobs=-1)
    joblib.dump( study, exp_folder+ nombre_exp_study)     
    
    
ds()

#study= joblib.load( '/home/a_reinaldomedina/Documents/comp1_study2_rank.joblib')
#study = optuna.create_study(direction="maximize")
optuna.visualization.plot_optimization_history(study).show(renderer="browser")
#optuna.visualization.plot_intermediate_values(study).show(renderer="browser")

