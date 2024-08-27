#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:21:19 2024

@author: reinaldo
"""

import pandas as pd
df= pd.read_csv('/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01_crudo.csv')
df.columns
df['foto_mes'].unique()
df[file['numero_de_cliente']==249246268]




lotes = [(202101, 202102, 202103),
         (202102, 202103, 202104),
         (202103, 202104, 202105),
         (202104, 202105, 202106)]

for mes_anterior, mes_actual, mes_siguiente in lotes:
    clientes_mes_actual = df[df['foto_mes'] == mes_actual]['numero_de_cliente']
    clientes_mes_siguiente = df[df['foto_mes'] == mes_siguiente]['numero_de_cliente']

    df.loc[(df['foto_mes'] == mes_anterior) & (~df['numero_de_cliente'].isin(clientes_mes_actual)) & (~df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'BAJA+1'
    df.loc[(df['foto_mes'] == mes_anterior) & (df['numero_de_cliente'].isin(clientes_mes_actual)) & (~df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'BAJA+2'
    df.loc[(df['foto_mes'] == mes_anterior) & (df['numero_de_cliente'].isin(clientes_mes_actual)) & (df['numero_de_cliente'].isin(clientes_mes_siguiente)), 'clase_ternaria'] = 'CONTINUA'

indx = df['foto_mes'].isin([202104, 202105, 202106])
df.loc[indx & df['clase_ternaria'].isna(), 'clase_ternaria'] = 'CONTINUA'

df.to_csv('/home/reinaldo/7a310714-2a6d-44bd-bd76-c6a65540eb82/DMEF/datasets/competencia_01.csv')
