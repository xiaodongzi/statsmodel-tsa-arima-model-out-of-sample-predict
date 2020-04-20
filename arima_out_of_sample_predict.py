#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:05:39 2020

@author: wangyaodong
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA #模型
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample

## 生成随机数  

data=np.random.rand(200)

df= pd.DataFrame({'data':data})

## 分割train_dataset 和test_dataset 

length = len(df)

train_data = df.iloc[0:int(length/8)*7,:]

test_data = df.iloc[int(length/8)*7:,:]

## 生成模型 
result = ARMA(train_data['data'],order=(2,1)).fit()

pred = result.predict()

# 将模型参数提取，然后输入新的数据，得到预测值
params = result.params

residuals = result.resid

p = result.k_ar

q = result.k_ma

k_exog = result.k_exog

k_trend = result.k_trend

steps = 1

## 样本内测试
in_data = train_data.iloc[0:11,:]
in_resid = residuals.iloc[0:11]

a = _arma_predict_out_of_sample(params, steps, in_resid, \
                       p, q, k_trend, k_exog,\
                       endog = in_data['data'], exog=None, start=len(in_data))


test_pred=pred[8:13]
test_pred
a[0]##  会发现没什么误差

## 样本外预测,滑窗式操作

new_resid =  residuals.tail(3).copy()

new_data = train_data.tail(3).copy()

for  i in range(len(test_data)):
    
    print(i)
    
    ## temp_data  应该增加最后一行，减去第一行
    a = _arma_predict_out_of_sample(params, steps, new_resid, \
               p, q, k_trend, k_exog,\
               endog = new_data['data'], exog=None, start=len(new_data))
    
    tomorrow_index = test_data.index[i]

    temp_resid = test_data.loc[tomorrow_index,'data'] - a[0]
    
    new_resid[tomorrow_index] = temp_resid
    
    new_resid.drop(new_resid.index[0],inplace=True)
    
    new_data = new_data.append(test_data.iloc[i,:])
    
    new_data.drop(new_data.index[0],inplace=True)
    
    pred[tomorrow_index] = a[0]


        
    
    
    
   
    










