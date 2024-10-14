
import pandas as pd
import numpy as np
#import seaborn as sns

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



base_path = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/'
base_path="~/buckets/b2/"
dataset_path = base_path + 'datasets/'
#modelos_path = base_path + 'modelos/'

dataset_file = 'competencia_01.csv'


ganancia_acierto = 273000
costo_estimulo = 7000



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

def create_rank(df_train):
    numeric_cols = df_train.select_dtypes(include=['int', 'float'])    
    cols_numeric_cols = list(numeric_cols.columns)
    
    cols_numeric_cols.pop(cols_numeric_cols.index('numero_de_cliente'))
    cols_numeric_cols.pop(cols_numeric_cols.index('foto_mes'))
    #cols_numeric_cols.pop(cols_numeric_cols.index('clase_ternaria'))
    
    threshold = 0.1 * len(df_train)
    numeric_cols= numeric_cols[cols_numeric_cols]
    valid_columns = numeric_cols.columns[(numeric_cols.ne(0).sum() + numeric_cols.notna().sum()) > threshold]
    
   # df_train = df_train[valid_columns].apply(lambda x: x.rank(pct=True), inplace=True).astype('float32')
    
    df_train[valid_columns] = df_train[valid_columns].apply(lambda x: x.rank(pct=True)).astype('float32')

    #res.columns = [col + '_rank' for col in res.columns if col  in [ 'numero_de_cliente','foto_mes'] continue]
    #res.columns = [col + '_rank' if col in ['numero_de_cliente', 'foto_mes'] else col for col in res.columns]
    new_col_names = []
    for col in res.columns:
        if col in ['numero_de_cliente', 'foto_mes']:
            print(col)
            new_col_names.append(col)
        else:
            new_col_names.append(col+'_rank')
    df_train[valid_columns].columns = new_col_names
    return res

def rank_time_features( df_train):
    w_df_train = df_train.sort_values(by=[ 'foto_mes'])    
    
    w_df_train_cols = list( w_df_train.select_dtypes(include=['int', 'float']).columns    ) 
   
    threshold = 0.1 * len(w_df_train)
    
    
    #cols = w_df_train.columns[(w_df_train.ne(0).sum() + w_df_train.notna().sum()) > threshold]
    #w_df_train = w_df_train[cols]
    for feature in w_df_train_cols:
        
        if feature  in ['numero_de_cliente', 'foto_mes','clase_ternaria']:
            continue
        if not((w_df_train[feature].ne(0).sum() + w_df_train[feature].notna().sum()) > threshold):
            continue
        print(feature)
        w_df_train[ feature+'_rank'] = w_df_train.groupby('foto_mes')[feature].rank(pct=True)#.apply(lambda x: x.rank(pct=True)).astype('float32')
        
    return w_df_train


def convert_to_int_float32(df_train):
    for col in df_train.select_dtypes(include=['float64']).columns:
        df_train[col] = df_train[col].astype('float32')
    
    for col in df_train.select_dtypes(include=['int64']).columns:
        df_train[col] = df_train[col].astype('int32')
    return df_train

def time_features( df_train):
    w_df_train = df_train.sort_values(by=['numero_de_cliente', 'foto_mes'])    
    threshold = 0.1 * len(w_df_train)
    
    
    #cols = w_df_train.columns[(w_df_train.ne(0).sum() + w_df_train.notna().sum()) > threshold]
    #w_df_train = w_df_train[cols]
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



def process_feature(w_df_train, feature):
    if feature in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']:
        return None
    
    result = pd.DataFrame()
    #result[feature+'_rolling_3'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.rolling(window=3).mean()).astype('float32')
    result[feature+'_rolling_2'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.rolling(window=2).mean()).astype('float32')
    #result[feature+'_roll_3_diff_1'] = result[feature+'_rolling_3'].groupby(w_df_train['numero_de_cliente']).fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
    result[feature+'_roll_2_diff_1'] = result[feature+'_rolling_2'].groupby(w_df_train['numero_de_cliente']).fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
    result[feature+'_diff_1'] = w_df_train.groupby('numero_de_cliente')[feature].fillna(0).transform(lambda x: x.diff(periods=1)).astype('float32')
    
    return result

def time_features_parallel(df_train):
    w_df_train = df_train.sort_values(by=['numero_de_cliente', 'foto_mes'])
    cols = w_df_train.select_dtypes(include=['int', 'float'])
    threshold = 0.1 * len(w_df_train)
    cols = w_df_train.columns[(w_df_train.ne(0).sum() + w_df_train.notna().sum()) > threshold]
    #w_df_train = w_df_train[cols]
    
    # Parallel processing of features
    results = Parallel(n_jobs=-1)(delayed(process_feature)(w_df_train, feature) for feature in cols if feature not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria'])
    
    # Combine all results into the original DataFrame
    results_df = pd.concat(results, axis=1)
    w_df_train = pd.concat([ results_df], axis=1)
    w_df_train = pd.concat([ df_train, results_df], axis=1)
    
    return w_df_train

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
data = pd.read_csv(dataset_path + dataset_file)    
data['m_suma_CA_CC'] = data['mcuenta_corriente']+ data['mcaja_ahorro']
data['Tarjetas_consumos_pesos'] = data['Master_mconsumospesos']+ data['Visa_mconsumospesos']
data['Tarjetas_consumos_colares'] = data['Master_mconsumosdolares']+ data['Visa_msaldodolares']
data['saldo/edad'] = data['mcuentas_saldo']/ data['cliente_edad']



data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

df_train_3 = data[data['foto_mes'] == 202102]
df_test = data[data['foto_mes'] == 202104]
y_train = df_train_3["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)        
y_test = df_test["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)

X_train = df_train_3.drop(columns=['clase_ternaria', 'foto_mes'])
X_test = df_test.drop(columns=['clase_ternaria', 'foto_mes'])

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

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

top_15_features = feature_importance_df.head(10)
from sklearn.preprocessing import StandardScaler


top_15_feature_names = top_15_features['feature'].tolist()
"""

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
"""
########################################################
"""df_stinky
data = pd.read_csv(dataset_path + dataset_file)    
data['m_suma_CA_CC'] = data['mcuenta_corriente']+ data['mcaja_ahorro']
data['Tarjetas_consumos_pesos'] = data['Master_mconsumospesos']+ data['Visa_mconsumospesos']
data['Tarjetas_consumos_colares'] = data['Master_mconsumosdolares']+ data['Visa_msaldodolares']
data['saldo/edad'] = data['mcuentas_saldo']/ data['cliente_edad']

data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
data['clase_peso'] = 1.0
data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

df_train_3 = data[data['foto_mes'] == 202102]
df_test = data[data['foto_mes'] == 202104]
y_train = df_train_3["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)        
y_test = df_test["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)

X_train = df_train_3.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])
X_test = df_test.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])


lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

w_train = df_train_3['clase_peso']
w_test = df_test['clase_peso']


train_data = lgb.Dataset(X_train,
                      label=y_train, # eligir la clase
                      weight=w_train)
test_data = lgb.Dataset(X_test,
                      label=y_test, # eligir la clase
                      weight=w_test)
model = lgb.train(params, train_data, feval=lgb_gan_eval)
y_pred = model.predict(X_test)
res0= lgb_gan_eval(y_pred, test_data)[1]
res_dict={}
for col in X_train.columns:    
    train_data = lgb.Dataset(X_train.drop(columns=[col]),
                          label=y_train, # eligir la clase
                          weight=w_train)
    test_data = lgb.Dataset(X_test.drop(columns=[col]),
                          label=y_test, # eligir la clase
                          weight=w_test)
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_pred = model.predict(X_test.drop(columns=[col]))
    res= lgb_gan_eval(y_pred, test_data)[1]
    res_dict[col]=res-res0"""
#/////////////////////////////////////////
#/////////////////////////////////////////
#/////////////////////////////////////////#/////////////////////////////////////////
#/////////////////////////////////////////

data = pd.read_csv(dataset_path + dataset_file)    
data['m_suma_CA_CC'] = data['mcuenta_corriente']+ data['mcaja_ahorro']
data['Tarjetas_consumos_pesos'] = data['Master_mconsumospesos']+ data['Visa_mconsumospesos']
data['Tarjetas_consumos_colares'] = data['Master_mconsumosdolares']+ data['Visa_msaldodolares']
data['saldo/edad'] = data['mcuentas_saldo']/ data['cliente_edad']

data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
"""
data['clase_peso'] = 1.0
data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

df_train_3 = data[data['foto_mes'] == 202101]
df_test = data[data['foto_mes'] == 202104]
y_train = df_train_3["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)        
y_test = df_test["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)

X_train = df_train_3.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])
X_test = df_test.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])


lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

w_train = df_train_3['clase_peso']
w_test = df_test['clase_peso']


train_data = lgb.Dataset(X_train,
                      label=y_train, # eligir la clase
                      weight=w_train)
test_data = lgb.Dataset(X_test,
                      label=y_test, # eligir la clase
                      weight=w_test)
model = lgb.train(params, train_data, feval=lgb_gan_eval)
y_pred = model.predict(X_test)
res0= lgb_gan_eval(y_pred, test_data)[1]
res_dict_enero={}
for col in X_train.columns:    
    train_data = lgb.Dataset(X_train.drop(columns=[col]),
                          label=y_train, # eligir la clase
                          weight=w_train)
    test_data = lgb.Dataset(X_test.drop(columns=[col]),
                          label=y_test, # eligir la clase
                          weight=w_test)
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_pred = model.predict(X_test.drop(columns=[col]))
    res= lgb_gan_eval(y_pred, test_data)[1]
    res_dict_enero[col]=res-res0"""
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

def subsample_data_time(df,  fraction, target_class='CONTINUA'):       
    df_target = df[df[y.name] == target_class]
    df_other = df[df[y.name] != target_class]
    random_state = int(time())
    df_target_sampled = df_target.sample(frac=fraction, random_state=random_state)
    df_sampled = pd.concat([df_other, df_target_sampled])
    return df_sampled
def subsample_data_time(df, fraction, target_class='CONTINUA', target_column='clase_ternaria'):
    df_target = df[df[target_column] == target_class]
    df_other = df[df[target_column] != target_class]
    random_state = int(time())
    df_target_sampled = df_target.sample(frac=fraction, random_state=random_state) 
    df_sampled = pd.concat([df_other, df_target_sampled])    
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

"""
df_stinky = pd.DataFrame(list(res_dict.items()), columns=['Feature', 'Value'])
df_stinky_enero = pd.DataFrame(list(res_dict_enero.items()), columns=['Feature', 'Value'])


"""
#####################################
#####################################
##########################################################################
#####################################
#####################################
#####################################
#df_stinky= joblib.load('/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/df_stinky.joblib')
#study_load= joblib.load('/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/comp_1_optuna0.joblib')
#a=study_load.trials_dataframe
#data_x= joblib.load('/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/data_x_rank.joblib')
#data_x = rank_time_features( data_x)
#data_x = time_features_parallel(data_x)
#joblib.dump(data_x, '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/data_x_rank.joblib')
#joblib.dump(data_x, '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/data_x_time.joblib')
data_x= joblib.load('/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/data_x_time.joblib')
leaks=[]
for col in data_x.columns:
    if 'clase_peso'in col.lower():
        leaks.append(col)
data_x.columns = data_x.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
data_x['clase_peso'] = 1.0
data_x.loc[data_x['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data_x.loc[data_x['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

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

#sel_kind_cols=['std', 'roll']
#cant_drop_stinky=3

mode_recalc= False
best_result = -float('inf')  # Start with a high value for minimization
best_predictions = None
penalty=100000
exp_folder = '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/'
#exp_folder = '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/exp/Python_optuna1/'
def objective(trial):
    global best_result, best_predictions, penalty, top_15_feature_names, data
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.02, 0.3)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.01, 1.0)
    params['num_leaves'] = trial.suggest_int("num_leaves", 31, 256)  # Example of leaf size
    params['min_child_samples'] = trial.suggest_int("min_child_samples", 1, 100)  # Example of coverage
    params['seed'] =   int(time())
    #params['max_depth'] = trial.suggest_int("max_depth", 3, 15)  # Control tree depth
   # params['min_gain_to_split'] = trial.suggest_float("min_gain_to_split", 0.0, 1.0)  # Minimum gain to split
    #params['lambda_l1'] = trial.suggest_float("lambda_l1", 0.0, 10.0)  # L1 regularization
    #params['lambda_l2'] = trial.suggest_float("lambda_l2", 0.0, 10.0)  # L2 regularization
    params['num_iterations'] = trial.suggest_int("num_iterations", 50, 1000)  # Number of boosting iterations    
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.1, 1.0)
    
   

    fraction = trial.suggest_float('fraction', 0.01, 1)
    
    categories= ['rank', 'suma', 'div', 'std', 'roll', 'dif']
    sel_cat_cols=[]
    for cat in categories:
        if  trial.suggest_categorical(cat, [True, False]):
            sel_cat_cols.append(cat) 
        
   

    original_columns = list( set(data.columns) -{'clase_ternaria'})    
    trial_number= trial.number
    if trial_number>100:
        feature_selection=[]
        for  col in original_columns:
            w_col = trial.suggest_categorical(col, [True, False])
            if w_col:
                feature_selection.append(col)            
    else: 
        feature_selection = original_columns
    
        
    columns = data_x.columns.tolist()
    final_selection=[]
    for col in columns:
        if any(orig_col.lower() in col.lower() for orig_col in feature_selection):
            final_selection.append(col)
        if  any(cat in col.lower() for cat in sel_cat_cols):
            if any(orig_col.lower() in col.lower() for orig_col in feature_selection):
                final_selection.append(col)
    
    
    final_selection= list( set(final_selection))
    
    final_selection = set(final_selection).union({'clase_ternaria', 'foto_mes', 'clase_peso'}) 
    final_selection = list( final_selection.union(top_15_feature_names) )
    
    test_date= 202104
    future_date= [202106 ]
    posible_trains= [202101, 202102]
    trains = []
    for wt in posible_trains:
        if posible_trains.index(wt)+1==len(posible_trains):
            if trains==[]:
                trains.append(wt)
                break            
        if trial.suggest_categorical(wt, [True, False]):
            trains.append (wt)
    
    #train_for_predict= [wt +2 for wt in trains]
    
    w_res = exectue_model(final_selection,trains,future_date, test_date, data_x, fraction, params,trial_number,feature_selection)
    return w_res- len(feature_selection )*penalty



#final_selection,trains,future_date, test_date, data_x, fraction, params,trial_number,feature_selection = objective_params(mock_trial)    
def exectue_model(final_selection,trains,future_date, test_date, data_x, fraction, params, trial_number, feature_selection):
    
    global best_result, best_predictions, penalty,exp_folder, mode_recalc
    data_x_selected= data_x[final_selection]
    df_train_3 = data_x_selected[data_x_selected['foto_mes'].isin(trains)]  
    df_test = data_x_selected[data_x_selected['foto_mes'] == test_date]    
    
    df_train_3= subsample_data_time(df_train_3, fraction, target_class='CONTINUA', target_column='clase_ternaria')    
        
    y_train = df_train_3["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)    
    y_test = df_test["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)
    
    w_train = df_train_3['clase_peso']
    w_test = df_test['clase_peso']    
    
    X_train = df_train_3.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])    
    X_test = df_test.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])  
    
    print( X_train.shape)
    #X_train = subsample_data_time(X_train,  fraction, target_class='CONTINUA')
    #X_pred = subsample_data_time(X_pred,  fraction, target_class='CONTINUA')  
        
    train_data = lgb.Dataset(X_train,
                          label=y_train, # eligir la clase
                          weight=w_train)
    test_data = lgb.Dataset(X_test,
                          label=y_test, # eligir la clase
                          weight=w_test)
     
    print(params)
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_test_pred = model.predict(X_test)
    res0= lgb_gan_eval(y_test_pred, test_data)[1]
    
    if (res0 >= best_result) &(trial_number>70) or mode_recalc:
       best_result = res0
       ganancia_real= penalty*len(X_test.columns)+ best_result
       k_result_vs_train_periods={}
       k_result_vs_train_periods['ganancia_real']=ganancia_real
       train_for_predict= [  [202101, 202102, 202103, 202104], [ 202102, 202103, 202104], [ 202103, 202104],[  202104] ]
       for train_set in train_for_predict:
           df_future = data_x_selected[data_x_selected['foto_mes'].isin(future_date)]      
           #df_future= subsample_data_time(df_future, fraction, target_class='CONTINUA', target_column='clase_ternaria')    
           
           X_future = df_future.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])
           
           df_pred = data_x_selected[data_x_selected['foto_mes'].isin(train_set)]      
           df_pred= subsample_data_time(df_pred, fraction, target_class='CONTINUA', target_column='clase_ternaria')    
           y_pred = df_pred["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)        
           w_pred = df_pred['clase_peso']
           X_pred = df_pred.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])
           pred_data = lgb.Dataset(X_pred, label=y_pred, weight=w_pred)
           model = lgb.train(params, pred_data,feval=lgb_gan_eval)
           final_pred = model.predict(X_future)       
           
           y_test_true, y_pred_lgm = y_test, y_test_pred
           y_future, X_future= final_pred, X_future
           k_result= calculate_treshold_cant_envios(y_test_true, y_pred_lgm, y_future, X_future)
           k_result_vs_train_periods[str(train_set)]=k_result
           
           importances = model.feature_importance()
           feature_names = X_train.columns.tolist()
           importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
           importance_df = importance_df.sort_values('importance', ascending=False)
           importance_df[importance_df['importance'] > 0]
           
           k_result_vs_train_periods[str(train_set)+'importan']=importance_df

           
       ganancia_real= penalty*len(feature_selection)+ best_result
       joblib.dump(k_result_vs_train_periods, exp_folder+ 'comp_1_dict_'+str(int(ganancia_real)) +'.joblib')
       #joblib.dump(k_result_vs_train_periods, '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/exp/Python_optuna1_b/'+ 'comp_1_dict_'+str(int(ganancia_real)) +'.joblib')
    return res0


ds()
study= joblib.load(exp_folder+'comp1_study2_rank.joblib')
#study = optuna.create_study(direction="maximize")

for i in range(0, 3000):
    #study.optimize(objective, n_trials=1)  # You can specify the number of trials
    study.optimize(objective, n_trials=3, n_jobs=-1)
    joblib.dump( study, exp_folder+ 'comp1_study2_rank.joblib')


ds()

#study= joblib.load( '/home/a_reinaldomedina/Documents/comp1_study2_rank.joblib')
#study = optuna.create_study(direction="maximize")
optuna.visualization.plot_optimization_history(study).show(renderer="browser")
#optuna.visualization.plot_intermediate_values(study).show(renderer="browser")


"""
study= joblib.load(base_path+'/exp/Python_optuna1/'+'exp_Python_optuna1_comp1_study2_rank.joblib' )

best_params = study.best_params

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
    
    
best_params = study.trials[300]
mock_trial = MockTrial(best_params)
best_objective_value = objective(mock_trial)

dic_best_trial = joblib.load('/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/exp/Python_optuna1_b/'+'comp_1_dict_163677000.joblib' )
"""

train_set
Out[69]: [202104]

#n_envios = 21500
#trial_number= 401

#cant_train_for_predict= 1
def simple_objective( trial):
    piso_envios = 8000
    techo_envios = 25000
    #n_envios = trial.suggest_int("n_envios", piso_envios, techo_envios, 500)   
    n_envios = trial.suggest_int(name="n_envios", low=piso_envios, high=techo_envios, step=500)

    cant_train_for_predict = trial.suggest_int("cant_train_for_predict", 0, 2)   
    trial_number = trial.suggest_categorical("trial_number", [346, 389, 401])   
    print ('trial_number', trial_number) 
    #trial_number = [346, 389, 401][trial_number]
    #trial_number= 300
    specific_trial = study.trials[trial_number]
    mock_trial = MockTrial(specific_trial.params)
    
    final_selection,trains,future_date, test_date, data_x, fraction, params,trial_number_2,feature_selection = objective_params(mock_trial)    
    train_for_predict= [  [202101, 202102, 202103, 202104], [202102, 202103, 202104], [202103, 202104],[  202104] ] 
    train_for_predict= [   [202102, 202103, 202104], [202103, 202104],[  202104] ] 
    train_set=  train_for_predict[cant_train_for_predict]
    if type(future_date)==int:
        future_date = [future_date]
    if type(train_set)==int:
        train_set = [train_set]    
    print ('n_envios', n_envios)
    print ('cant_train_for_predict', cant_train_for_predict) 
    print ('trial_number', trial_number) 
    print ('train_set', train_set) 
    mode_recalc=True
    K_dict = exectue_model_return_dict(final_selection,trains,future_date, test_date, data_x, fraction, params, trial_number, feature_selection,train_set)
    #joblib.dump(K_dict, exp_folder+ 'comp_1_dict_w_K_dict_898_set2.joblib')
    joblib.dump(K_dict, exp_folder+'K_dict'+ str(train_set) + 'Trial_' + str(trial_number) + 'Env_'+ str(n_envios) +'.joblib')
    entrega = K_dict[str(train_set)][n_envios]
    #exp_folder= '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/exp/Python_optuna1_b/'
    entrega.to_csv( exp_folder+ 'K'+'_Trains_'+ str(train_set) + 'Trial_' + str(trial_number) + 'Env_'+ str(n_envios) + '.csv')
    
    #joblib.dump(entrega, exp_folder+ 'comp_1_dict_'+str(int(ganancia_real)) +'.joblib')
    kagle_resutl = input("Please enter a number: ")
    print('kagle_resutl', kagle_resutl)
    return kagle_resutl


study= joblib.load(exp_folder+'comp1_study2_rank.joblib')
study_mock = optuna.create_study(direction="maximize")
mode_recalc=True
for i in range(0, 3000):
    #study.optimize(objective, n_trials=1)  # You can specify the number of trials
    study_mock.optimize(simple_objective, n_trials=1)
    
    joblib.dump( study_mock, exp_folder+ 'comp1_sstudy_mock_rank.joblib')







def objective_params(trial):
    global best_result, best_predictions, penalty, top_15_feature_names, data
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.02, 0.3)   
    params['feature_fraction'] = trial.suggest_float("feature_fraction", 0.01, 1.0)
    params['num_leaves'] = trial.suggest_int("num_leaves", 31, 256)  # Example of leaf size
    params['min_child_samples'] = trial.suggest_int("min_child_samples", 1, 100)  # Example of coverage
    params['seed'] =   int(time())
    #params['max_depth'] = trial.suggest_int("max_depth", 3, 15)  # Control tree depth
    #params['min_gain_to_split'] = trial.suggest_float("min_gain_to_split", 0.0, 1.0)  # Minimum gain to split
    #params['lambda_l1'] = trial.suggest_float("lambda_l1", 0.0, 10.0)  # L1 regularization
    #params['lambda_l2'] = trial.suggest_float("lambda_l2", 0.0, 10.0)  # L2 regularization
    params['num_iterations'] = trial.suggest_int("num_iterations", 50, 1000)  # Number of boosting iterations    
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.1, 1.0)
    
    fraction = trial.suggest_float('fraction', 0.01, 1)
    
    categories= ['rank', 'suma', 'div', 'std', 'roll', 'dif']
    sel_cat_cols=[]
    for cat in categories:
        if  trial.suggest_categorical(cat, [True, False]):
            sel_cat_cols.append(cat) 

    original_columns = list( set(data.columns) -{'clase_ternaria'})    
    trial_number= trial.number
    if trial_number>100:
        feature_selection=[]
        for  col in original_columns:
            w_col = trial.suggest_categorical(col, [True, False])
            if w_col:
                feature_selection.append(col)            
    else: 
        feature_selection = original_columns    
        
    columns = data_x.columns.tolist()
    final_selection=[]
    for col in columns:
        if any(orig_col.lower() in col.lower() for orig_col in feature_selection):
            final_selection.append(col)
        if  any(cat in col.lower() for cat in sel_cat_cols):
            if any(orig_col.lower() in col.lower() for orig_col in feature_selection):
                final_selection.append(col)    
    
    final_selection= list( set(final_selection))
    
    final_selection = set(final_selection).union({'clase_ternaria', 'foto_mes', 'clase_peso'}) 
    final_selection = list( final_selection.union(top_15_feature_names) )
    
    test_date= 202104
    pred_date= 202106    
    posible_trains= [202101, 202102]
    trains = []
    for wt in posible_trains:
        if posible_trains.index(wt)+1==len(posible_trains):
            if trains==[]:
                trains.append(wt)
                break            
        if trial.suggest_categorical(wt, [True, False]):
            trains.append (wt)
    
    #train_for_predict= [wt +2 for wt in trains]    
    #w_res = exectue_model(final_selection,trains,pred_date, test_date, data_x, fraction, params,trial_number,feature_selection)
    return final_selection,trains,pred_date, test_date, data_x, fraction, params,trial_number,feature_selection


def exectue_model_return_dict(final_selection,trains,future_date, test_date, data_x, fraction, params, trial_number, feature_selection,train_set):
    
    global best_result, best_predictions, penalty,exp_folder, mode_recalc
    data_x_selected= data_x[final_selection]
    df_train_3 = data_x_selected[data_x_selected['foto_mes'].isin(trains)]  
    df_test = data_x_selected[data_x_selected['foto_mes'] == test_date]    
    
    df_train_3= subsample_data_time(df_train_3, fraction, target_class='CONTINUA', target_column='clase_ternaria')    
        
    y_train = df_train_3["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)    
    y_test = df_test["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)
    
    w_train = df_train_3['clase_peso']
    w_test = df_test['clase_peso']    
    
    X_train = df_train_3.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])    
    X_test = df_test.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])  
    
    print( X_train.shape)
    #X_train = subsample_data_time(X_train,  fraction, target_class='CONTINUA')
    #X_pred = subsample_data_time(X_pred,  fraction, target_class='CONTINUA')  
        
    train_data = lgb.Dataset(X_train,
                          label=y_train, # eligir la clase
                          weight=w_train)
    test_data = lgb.Dataset(X_test,
                          label=y_test, # eligir la clase
                          weight=w_test)
     
    print(params)
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_test_pred = model.predict(X_test)
    res0= lgb_gan_eval(y_test_pred, test_data)[1]
    
    if (res0 >= best_result) &(trial_number>70) or mode_recalc:
       best_result = res0
       ganancia_real= penalty*len(X_test.columns)+ best_result
       k_result_vs_train_periods={}
       k_result_vs_train_periods['ganancia_real']=ganancia_real
       train_for_predict= [ [ 202102, 202103, 202104], [ 202103, 202104],[  202104] ]
       train_for_predict= train_set
       for train_set in train_for_predict:
           if type(train_set)==int:
               train_set = [train_set]    
           df_future = data_x_selected[data_x_selected['foto_mes'].isin(future_date)]      
           #df_future= subsample_data_time(df_future, fraction, target_class='CONTINUA', target_column='clase_ternaria')    
           
           X_future = df_future.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])
           
           df_pred = data_x_selected[data_x_selected['foto_mes'].isin(train_set)]      
           df_pred= subsample_data_time(df_pred, fraction, target_class='CONTINUA', target_column='clase_ternaria')    
           y_pred = df_pred["clase_ternaria"].map(lambda x: 0 if x == "CONTINUA" else 1)        
           w_pred = df_pred['clase_peso']
           X_pred = df_pred.drop(columns=['clase_ternaria', 'foto_mes', 'clase_peso'])
           pred_data = lgb.Dataset(X_pred, label=y_pred, weight=w_pred)
           model = lgb.train(params, pred_data,feval=lgb_gan_eval)
           final_pred = model.predict(X_future)       
           
           y_test_true, y_pred_lgm = y_test, y_test_pred
           y_future, X_future= final_pred, X_future
           k_result= calculate_treshold_cant_envios(y_test_true, y_pred_lgm, y_future, X_future)
           k_result_vs_train_periods[str(train_set)]=k_result
           
           importances = model.feature_importance()
           feature_names = X_train.columns.tolist()
           importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
           importance_df = importance_df.sort_values('importance', ascending=False)
           importance_df[importance_df['importance'] > 0]
           
           k_result_vs_train_periods[str(train_set)+'importan']=importance_df

           
       ganancia_real= penalty*len(feature_selection)+ best_result
       #joblib.dump(k_result_vs_train_periods, exp_folder+ 'comp_1_dict_'+str(int(ganancia_real)) +'.joblib')
       #joblib.dump(k_result_vs_train_periods, '/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/exp/Python_optuna1_b/'+ 'comp_1_dict_'+str(int(ganancia_real)) +'.joblib')
    return k_result_vs_train_periods


































"""
# Given variables
piso_envios = 4000
techo_envios = 20000

# Calculate maximum ganancia and index
ganancia_max = ganancia_cum.max()
gan_max_idx = np.where(ganancia_cum == ganancia_max)[0][0]

# Determine the optimal threshold from y_pred_lgm
threshold = y_pred_lgm[gan_max_idx]

# Create the final result array (1 for sent, 0 for not sent)
final_result = np.where(y_pred_lgm >= threshold, 1, 0)
np.sum(final_result)
# Ensure the number of "envios" matches gan_max_idx
if final_result.sum() > gan_max_idx:
    # If too many "envios", adjust to the desired quantity
    # This will set the lowest probabilities to 0 to match the count
    indices_to_adjust = np.argsort(y_pred_lgm)[:final_result.sum() - gan_max_idx]
    final_result[indices_to_adjust] = 0
elif final_result.sum() < gan_max_idx:
    # If too few "envios", ensure to keep as many as possible
    # Only consider the top gan_max_idx predictions
    indices_to_keep = np.argsort(y_pred_lgm)[-gan_max_idx:]
    final_result[:] = 0  # Reset all to 0
    final_result[indices_to_keep] = 1  # Set the top predictions to 1

# Plotting the gain curve with the maximum point
plt.figure(figsize=(10, 6))
plt.plot(range(piso_envios, len(ganancia_cum[piso_envios:techo_envios]) + piso_envios), ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
plt.axvline(x=gan_max_idx, color='g', linestyle='--', label=f'Punto de corte a la ganancia máxima {gan_max_idx}')
plt.axhline(y=ganancia_max, color='r', linestyle='--', label=f'Ganancia máxima {ganancia_max}')
plt.title('Curva de Ganancia')
plt.xlabel('Clientes')
plt.ylabel('Ganancia')
plt.legend()
plt.show()

# Print the final results for verification
print("Final Result Array:", final_result)
print("Total Envios Sent:", final_result.sum())













model = lgb.Booster(model_file=modelos_path + 'lgb_first.txt')
importances = model.feature_importance()
feature_names = X_train.columns.tolist()
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
importance_df[importance_df['importance'] > 0]

z=pd.read_csv('/home/reinaldo/Downloads/exp_KA4210_base_KA4210_base_12000.csv')

study = optuna.create_study(direction="maximize")

for i in range(0, 1000):
    study.optimize(objective, n_trials=1)  # You can specify the number of trials
    #study.optimize(objective, n_trials=3, n_jobs=3)
    joblib.dump( study, '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/'+ 'comp1_study1_rank.joblib')
    #joblib.dump( study,'/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/src/monday/sel_cols.joblib' )
study_df= study.trials_dataframe()
study_df= study.best_params #154098000
FrozenTrial(number=122, state=1, values=[154098000.0], datetime_start=datetime.datetime(2024, 10, 12, 18, 8, 46, 985360), datetime_complete=datetime.datetime(2024, 10, 12, 18, 9, 15, 301168), params={'rank': False, 'std': True, 'roll': False, 'dif': False, 'cant_drop_stinky': 7, 202101: True, 202102: True, 202103: True, 'fraction': 0.3703106548879275, 'learning_rate': 0.06060567413889884, 'feature_fraction': 0.48397537799069457, 'num_leaves': 59, 'min_child_samples': 31}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'rank': CategoricalDistribution(choices=(True, False)), 'std': CategoricalDistribution(choices=(True, False)), 'roll': CategoricalDistribution(choices=(True, False)), 'dif': CategoricalDistribution(choices=(True, False)), 'cant_drop_stinky': IntDistribution(high=20, log=False, low=0, step=1), 202101: CategoricalDistribution(choices=(True, False)), 202102: CategoricalDistribution(choices=(True, False)), 202103: CategoricalDistribution(choices=(True, False)), 'fraction': FloatDistribution(high=1.0, log=False, low=0.01, step=None), 'learning_rate': FloatDistribution(high=0.3, log=False, low=0.02, step=None), 'feature_fraction': FloatDistribution(high=1.0, log=False, low=0.01, step=None), 'num_leaves': IntDistribution(high=256, log=False, low=31, step=1), 'min_child_samples': IntDistribution(high=100, log=False, low=1, step=1)}, trial_id=122, value=None)
study.best_result()
study.best_trial












def psi(expected, actual, buckets=10):

    def psi_formula(expected_prop, actual_prop):
        result = (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)
        return result

    expected_not_null = expected.dropna()
    actual_not_null = actual.dropna()

    bin_edges = pd.qcut(expected_not_null, q=buckets, duplicates='drop').unique()
    bin_edges2 = [edge.left for edge in bin_edges] + [edge.right for edge in bin_edges]
    breakpoints = sorted(list(set(bin_edges2)))

    expected_counts, _ = np.histogram(expected_not_null, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_not_null, bins=breakpoints)

    expected_prop = expected_counts / len(expected_not_null)
    actual_prop = actual_counts / len(actual_not_null)

    psi_not_null = psi_formula(expected_prop, actual_prop).sum()

    psi_null = 0

    if expected.isnull().sum() > 0 and actual.isnull().sum() > 0 :
      expected_null_percentage = expected.isnull().mean()
      actual_null_percentage = actual.isnull().mean()
      psi_null = psi_formula(expected_null_percentage, actual_null_percentage)

    return psi_not_null + psi_null


psi_results = []
for column in train_data.columns:
  if column not in ['foto_mes', 'clase_ternaria']:
    train_variable = train_data[column]
    score_variable = score_data[column]
    psi_value = psi(train_variable, score_variable)
    psi_results.append({'feature': column, 'psi': psi_value})

psi_df = pd.DataFrame(psi_results)
psi_df = psi_df.sort_values('psi', ascending=False)
psi_df











ds()

from joblib import dump
storage = 'sqlite:///example.db'  # Use SQLite or any other database supported by Optuna

study_name = 'example_study'  # Unique study name
study2 = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)

# Use parallel optimization
def run_optuna():
    study2.optimize(objective, n_trials=1)
    dump(study, '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/comp1_study1_rank.joblib')

if __name__ == "__main__":
    from joblib import Parallel, delayed
    
    # Run 1000 trials in parallel
    Parallel(n_jobs=5)(delayed(run_optuna)() for _ in range(1000))


import optuna
from joblib import dump
from joblib import Parallel, delayed

import time
# Database setup
storage = 'sqlite:///example.db'  # Use SQLite or any other database supported by Optuna
study_name = 'example_study'  # Unique study name

# Create or load the study
study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)

# Define the optimization function
def run_optuna():
    try:
        study.optimize(objective, n_trials=1)
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(1)  # Wait before retrying
        run_optuna()  # Retry the optimization

if __name__ == "__main__":
    # Run 1000 trials in parallel
    Parallel(n_jobs=5)(delayed(run_optuna)() for _ in range(1000))
    
    # Save the study after all trials
    dump(study, '/home/a_reinaldomedina/buckets/b2/exp/Python_optuna1/comp1_study1_rank_sql.joblib')



import gc
gc()
   """

   
"""   
   
   
   selected_columns=[]
    for  col in columns:
        w_col = trial.suggest_categorical(col, [True, False])
        if w_col:
            selected_columns.append(col) 
    print(selected_columns)
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]
    assert len(X_train_selected) == len(y_train_binaria2), "Training data and labels length mismatch!"
    assert len(X_test_selected) == len(y_test_class), "Test data and labels length mismatch!"

    params = {'objective': 'binary',
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
             'min_data_in_leaf': 2500        }
    
    train_data = lgb.Dataset(X_train_selected,
                          label=y_train_binaria2, # eligir la clase
                          weight=w_train)
    test_data = lgb.Dataset(X_test_selected,
                          label=y_test_class, # eligir la clase
                          weight=w_test)
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_pred = model.predict(X_test_selected)
    res= lgb_gan_eval(y_pred, test_data)
    return res[1]- len (selected_columns)*1000






def objective(trial):
    columns = X_train.columns.tolist()
    selected_columns=[]
    for  col in columns:
        w_col = trial.suggest_categorical(col, [True, False])
        if w_col:
            selected_columns.append(col) 
    print(selected_columns)
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]
    assert len(X_train_selected) == len(y_train_binaria2), "Training data and labels length mismatch!"
    assert len(X_test_selected) == len(y_test_class), "Test data and labels length mismatch!"

    params = {'objective': 'binary',
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
             'min_data_in_leaf': 2500        }
    
    train_data = lgb.Dataset(X_train_selected,
                          label=y_train_binaria2, # eligir la clase
                          weight=w_train)
    test_data = lgb.Dataset(X_test_selected,
                          label=y_test_class, # eligir la clase
                          weight=w_test)
    model = lgb.train(params, train_data, feval=lgb_gan_eval)
    y_pred = model.predict(X_test_selected)
    res= lgb_gan_eval(y_pred, test_data)
    return res[1]- len (selected_columns)*1000

study_name = "exp_301_lgbm"
study = optuna.create_study(direction="maximize", study_name=study_name)

for i in range(0, 1000):
    study.optimize(objective, n_trials=1)  # You can specify the number of trials
    joblib.dump( study,'/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/src/monday/sel_cols.joblib' )


z= study.trials_dataframe()
a=study.best_params
for key in a.keys():
    if not a[key]:
        print (key, ', ')

print(', '.join(f"'{key}'" for key in a.keys() if not a[key]))



study.frame


study_name = "exp_301_lgbm"
study = optuna.create_study(
    direction="maximize",
    study_name=study_name,   
)
for i in range(1000):
    study
    
    
    study_name = "exp_301_lgbm"
study = optuna.create_study(direction="maximize", study_name=study_name)

for i in range(2000):
    study.optimize(lambda trial: lgb_eval(trial, X_train, y_train_binaria2, X_test, y_test_class, w_train), n_trials=1)
"""