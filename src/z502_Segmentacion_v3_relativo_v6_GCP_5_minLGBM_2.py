
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
from boruta import BorutaPy
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

import time
import optuna
from optuna.study import StudyDirection




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
def convert_to_int_float32_polars(df_train):
    # Define the maximum and minimum values for float32
    max_float32 = np.float32(3.4028235e+38)
    min_float32 = np.float32(-3.4028235e+38)
    
    # Select float64 columns and cast them to float32, while handling infinities
    float32_cols = [
        pl.when(pl.col(col).is_infinite())  # When column has infinity values
        .then(pl.when(pl.col(col) == np.inf)  # If positive infinity
              .then(pl.lit(max_float32))  # Replace +∞ with max float32
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
            feat_col.rolling_mean(window_size=3)
                .over("numero_de_cliente")
                .cast(pl.Float32)
                .alias(f"{feature}_rolling_3"),
                
            feat_col.rolling_mean(window_size=2)
                .over("numero_de_cliente")
                .cast(pl.Float32)
                .alias(f"{feature}_rolling_2"),
            
            # Rolling diffs
            feat_col.rolling_mean(window_size=3)
                .over("numero_de_cliente")
                .diff()
                .cast(pl.Float32)
                .alias(f"{feature}_roll_3_diff_1"),
                
            feat_col.rolling_mean(window_size=2)
                .over("numero_de_cliente")
                .diff()
                .cast(pl.Float32)
                .alias(f"{feature}_roll_2_diff_1"),
                
            # Regular diff
            feat_col.diff()
                .over("numero_de_cliente")
                .cast(pl.Float32)
                .alias(f"{feature}_diff_1")
        ])
    
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

def lgb_gan_eval(y_pred, data):
    #weight = data.get_weight()
    weight = data.weight.copy()
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
    
    
    df_train_3 = data.filter(pl.col('foto_mes') == mes_train)
    df_train_3= subsample_data_time_polars(df_train_3, 0.1, 'CONTINUA', 'clase_ternaria', 42)      
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
    for i in data_top_std.columns:
        for k in data_top_std.columns:
            if i != k:
                new_columns.extend([
                    (pl.col(i)* np.cos(np.pi/4) - np.sin(np.pi/4)*pl.col(k)  ).alias(f"{i}_rot45c1_{k}"),
                    (pl.col(i)* np.sin(np.pi/4) + np.cos(np.pi/4)*pl.col(k)  ).alias(f"{i}_rot45c2_{k}"),
                    (pl.col(i) - pl.col(k)).alias(f"{i}_minus_{k}"),
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




def create_data(ganancia_acierto, last_date_to_consider, path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_least_ampliado, N_bins, lag_flag, delta_lag_flag):
    if not os.path.exists(path_set_con_ternaria):
        crete_ternaria( path_set_crudo, path_set_con_ternaria)     
        
    data = pl.read_csv(path_set_con_ternaria)    
    data = data.rename({'': 'indicex'})
    data = data.drop('indicex')
    data.columns = [col.replace(r'[^a-zA-Z0-9_]', '_') for col in data.columns]
    data = data.with_columns(  pl.col('Master_Finiciomora').cast(pl.Float64)  )
    data = data.with_columns(  pl.col('tmobile_app').cast(pl.Float32)  )
    data = data.with_columns(  pl.col('cmobile_app_trx').cast(pl.Float32)  )
    # redusco dataset eliminando registros muy viejos
    # data = data.filter(pl.col('foto_mes') > 202100)
    last_date_to_consider= 201910
    data = data.filter(pl.col('foto_mes') > last_date_to_consider)
   
    original_columns= data.columns
    data= AgregarVariables_IntraMes(data)
    original_columns_inta_mes= data.columns
    # data = data.iloc[:500000,:]
    # data= data.slice(0, 100000)    
    #all_times[:-stride-window]
    #Out[25]: [202108, 202107, 202106, 202105]
    #mes_train, mes_test  =   202105, 202106
    #top_15_feature_names , least_15_features, least_ampliado=   get_top_and_least_important( data, N_top, N_least, N_least_ampliado,  mes_train, mes_test  )
    data= convert_to_int_float32_polars(data)
   # features_above_canritos, features_below_canritos = get_top_and_least_important_y_canaritos( data, N_top, N_least, N_least_ampliado,  mes_train, mes_test  )
    data= standardize_columns(data) 
    data= convert_to_int_float32_polars(data)
    col_dinero= ['mcaja_ahorro_std' , 'mcuenta_corriente_std', 'mcuentas_saldo_std', 'ctrx_quarter_std', 'mrentabilidad_std', 'ctrx_quarter_normalizado_std']
    #data = add_forecast_elasticnet( data, col_dinero )
    
    
    feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta( data,ganancia_acierto,  mes_train, mes_test  )
    baja_importancia = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==False ]['feature'].to_list()   
    
    data, new_features = enhanced_feature_binning(data, baja_importancia, N_bins=5)
    #data.write_parquet(exp_folder+'data_x_w0_elas.parquet' )
    #data = add_forecast_elasticnet( data,features_above_canritos[:7])
    data = time_features(data)
    data= convert_to_int_float32_polars(data)
    #data.write_parquet(exp_folder+'data_x_w0_time2.parquet' , compression='gzip')
    #data = pl.read_parquet(exp_folder+'data_x_w0_time.parquet' )
    
    feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta( data,ganancia_acierto,  mes_train, mes_test  )
    features_finales = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==True ]['feature'].to_list()   
    data= data[['numero_de_cliente','foto_mes','clase_ternaria']+ features_finales]
# 16% 624GB

    
    lag_flag, delta_lag_flag = True, True
    data= add_lags_diff(data, lag_flag, delta_lag_flag )
    
    feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta( data,ganancia_acierto,  mes_train, mes_test  )
    features_finales = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==True ]['feature'].to_list()   
    data= data[['numero_de_cliente','foto_mes','clase_ternaria']+ features_finales]
    
    data.write_parquet(exp_folder+'data_x_w0_lags.parquet' , compression='gzip')
    
    #data = time_features(data)
    #data_reg = regression_per_client(data ,features_below_canritos) #muy lento Usae el codigo de R
    data = div_sum_top_features_polars(data, feature_importance_df_ranking['feature'][:50].to_list())
    
    
    
    #print_nan_columns(data, 0.75, original_columns)
    data= add_moth_encode( data)
    #print_nan_columns(data, 0.75, original_columns)
    #data= bins_least_importatn( data,features_below_canritos, N_bins ) 
    
    
    data= convert_to_int_float32_polars(data)
    #features_above_canritos, features_above_canritos = get_top_and_least_important_y_canaritos( data, N_top, N_least, N_least_ampliado,  mes_train, mes_test  )
    feature_importance_df_ranking, feature_importance_df_bool = get_top_and_least_important_boruta( data,ganancia_acierto,  mes_train, mes_test  )
    features_finales = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==True ]['feature'].to_list()
    #features_below_canritos = feature_importance_df_bool[ feature_importance_df_bool['importance_split']==False ]['feature'].to_list()
    data= data[['numero_de_cliente','foto_mes','clase_ternaria']+ features_finales]
    #data = drop_columns_nan_zero(data, 0.75, original_columns)
    data.write_parquet(exp_folder+'data_x_w0_final.parquet' , compression='gzip')
    #data_x=data
    
    data.write_parquet(exp_folder+'data_x_w0_final.parquet' )
    data_x=data
    joblib.dump( [ original_columns,original_columns_inta_mes, features_finales, feature_importance_df_ranking, feature_importance_df_bool], exp_folder+'dataset_201911s_elsatic.joblib')
    data_x.write_parquet(exp_folder+'data_x.parquet' )
    return original_columns,original_columns_inta_mes,  data,  features_finales, feature_importance_df_ranking, feature_importance_df_bool


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


# !pip install polars
ganancia_acierto = 273000
costo_estimulo = 7000

N_top, N_least,  mes_train, mes_test, test_future = 15, 20, 202104, 202106, 202108
last_date_to_consider = 202000
last_date_to_consider = 202006
last_date_to_consider = 201911
N_least_ampliado = 30
N_bins=5
path_set_con_ternaria = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_02.csv'

# path_set_crudo = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01_crudo.csv'
# path_set_con_ternaria = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_02.csv'
#path_set_con_features_eng = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_02_features_eng.joblib'


path_set_crudo = "/home/a_reinaldomedina/buckets/b2/datasets/competencia_02_crudo.csv"
path_set_con_ternaria = "/home/a_reinaldomedina/buckets/b2/datasets/competencia_02.csv"
exp_folder = '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/'


path_set_crudo = "/home/medina_robledo/buckets/b1/datasets/competencia_02_crudo.csv"
path_set_con_ternaria = "/home/medina_robledo/buckets/b1/datasets/competencia_02.csv"
exp_folder = '/home/medina_robledo/buckets/b1/exp/escopeta_1/'

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

lag_flag, delta_lag_flag = True, True
#original_columns, data_x,  top_15_feature_names , least_15_features, least_ampliado = create_data(last_date_to_consider, path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_least_ampliado, N_bins,lag_flag, delta_lag_flag)
ds()
original_columns,original_columns_inta_mes,  data_x,  features_finales, feature_importance_df_ranking, feature_importance_df_bool =create_data(ganancia_acierto, last_date_to_consider, path_set_crudo, path_set_con_ternaria, N_top, N_least,  mes_train, mes_test , N_least_ampliado, N_bins,lag_flag, delta_lag_flag)

joblib.dump( [ original_columns,original_columns_inta_mes, features_finales, feature_importance_df_ranking, feature_importance_df_bool], '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/dataset_202000s_elsatic.joblib')
data_x.write_parquet(exp_folder+'data_x.parquet' )

#data_x= data
del(data)
leaks=[]
for col in data_x.columns:
    if 'clase_peso'in col.lower():
        leaks.append(col)




penalty=0

#exp_folder = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/exp/Python_optuna1/'
#exp_folder = "~/buckets/b2/exp/comp2/"
nombre_exp_study = 'comp2_study_4_1.joblib'
random_state=42
cant_semillas_ensamble= 100
cant_ensambples_ensamble= 10
trains= [202004,202005,202006,202007,202008,202009,202010,202011,202012,202101,202102, 202103, 202104]
final_train = [202006,202007,202008,202009,202010,202011,202012,202101,202102, 202103, 202104, 202105, 202106]


#trains= [ 202103, 202104]
trains= [202011,202012,202101,202102, 202103, 202104]
#final_train = [202006,202007,202008,202009,202010,202011,202012,202101,202102, 202103, 202104, 202105, 202106]



#top_15_feature_names= features_finales[:50]
top_15_feature_names= feature_importance_df_ranking['feature'][:50]
def objective(trial):
    global best_result, best_predictions, penalty, top_15_feature_names, data,random_state, trains,mes_test
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.7)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.01, 1.0)
   
    params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 100, 15000)  # Example of leaf size    
    #params['min_child_samples'] = trial.suggest_int("min_child_samples", 1, 100)  # Example of coverage
    #params['seed'] =   int(time())
    #params['max_depth'] = trial.suggest_int("max_depth", 3, 15)  # Control tree depth
    #params['min_gain_to_split'] = trial.suggest_float("min_gain_to_split", 0.0, 1.0)  # Minimum gain to split
    #params['lambda_l1'] = trial.suggest_float("lambda_l1", 0.0, 10.0)  # L1 regularization
    #params['lambda_l2'] = trial.suggest_float("lambda_l2", 0.0, 10.0)  # L2 regularization
    params['num_iterations'] = trial.suggest_int("num_iterations", 3, 500)  # Number of boosting iterations    
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.01, 1.0)   
    clase_peso_lgbm = trial.suggest_int('clase_peso_lgbm',1, ganancia_acierto+10000)   
    cant_semillas_ensamble = trial.suggest_int('cant_semillas_ensamble',100, 2000)   
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
    random_numbers = np.random.randint(low=-32768, high=32767, size=cant_semillas_ensamble, dtype=np.int16).tolist()
    res= []
    start= time.time()
    for rnd in random_numbers:
        y_test_pred,test_data = exectue_model(final_selection,trains, mes_test, data_x, fraction, params,trial_number,feature_selection,random_state)
        res.append( w_res)
    elapsed_time =  time.time() -start
    res =np.mean( res)
    res0= lgb_gan_eval(res, test_data)[1]  
    return res0 ,elapsed_time #- len(feature_selection )*penalty, time



#final_selection,trains, mes_test, data_x, fraction, params,trial_number,feature_selection = objective_params(mock_trial)    
def exectue_model(final_selection,trains, mes_test, data_x, fraction, params, trial_number, feature_selection, random_state):
    #data_x['clase_peso'] = 1.0
    data_x = data_x.with_columns( pl.lit(1.0).alias('clase_peso')  )
    data_x = data_x.with_columns(
        pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(clase_peso_lgbm+0.00002)
        .when(pl.col('clase_ternaria') == 'BAJA+1').then(clase_peso_lgbm+0.00001)
        .otherwise(pl.col('clase_peso'))  # Keep the original value if no condition matches
        .alias('clase_peso')
    )

    global best_result, best_predictions, penalty,exp_folder, mode_recalc
    data_x_selected= data_x[final_selection]
    #df_train_3 = data_x_selected[data_x_selected['foto_mes'].isin(trains)]  
    df_train_3 = data_x_selected.filter(pl.col('foto_mes').is_in(trains))
    #df_test = data_x_selected[data_x_selected['foto_mes'] == mes_test]    
    df_test = data_x_selected.filter(pl.col('foto_mes') == mes_test)
   
    df_train_3= subsample_data_time_polars(df_train_3, fraction, 'CONTINUA', 'clase_ternaria', random_state)                              
    """
    
        
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
    params['seed'] =   random_state
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_test_pred = model.predict(X_test)
    #res0= lgb_gan_eval(y_test_pred, test_data)[1]  
    
    file_path = os.path.join(exp_folder, 'comp_2_dict.joblib')
  
    if os.path.exists(file_path):
        res_dict = joblib.load(file_path)
        res_dict[trial_number] = y_test_pred
    else:
        res_dict = {}
        res_dict[trial_number] = y_test_pred
        joblib.dump(res_dict, file_path)
        """
    if trial_number>400:
        df_future = data_x_selected.filter(pl.col('foto_mes') == test_future)
        X_future = df_future.drop(['clase_ternaria', 'foto_mes', 'clase_peso'])
        future_data = lgb.Dataset(X_future.to_pandas() )       
        y_future = model.predict( future_data )"""
       
    return y_test_pred,test_data #res0



if os.path.exists(exp_folder+nombre_exp_study):
    study= joblib.load(exp_folder+nombre_exp_study )
else: 
    #study = optuna.create_study(direction="maximize")
    #study = optuna.create_study(direction=["maximize", "minimize"])
    study = optuna.create_study(
        directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    )

for i in range(0, 3000):
    #study.optimize(objective, n_trials=1)  # You can specify the number of trials
    study.optimize(objective, n_trials=1, n_jobs=-1)
    joblib.dump( study, exp_folder+ nombre_exp_study)     
    
    
ds()

#study= joblib.load( '/home/a_reinaldomedina/Documents/comp1_study2_rank.joblib')
#study = optuna.create_study(direction="maximize")
optuna.visualization.plot_optimization_history(study).show(renderer="browser")
#optuna.visualization.plot_intermediate_values(study).show(renderer="browser")

