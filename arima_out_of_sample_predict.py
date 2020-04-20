#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:22:47 2020

@author: wangyaodong
"""


import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller #ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验

from statsmodels.tsa.arima_model import ARIMA #模型
from statsmodels.tsa.arima_model import ARMA #模型
from statsmodels.stats.stattools import durbin_watson #DW检验
import tushare as ts

from statsmodels.tsa.arima_model import _arma_predict_out_of_sample

def get_data(code):

    data = ts.get_k_data(code)

    #删除列code

    df = data.drop('code',axis = 1)

    df['ret'] = df['close'].pct_change()
    
    df.dropna(inplace=True)
    
    df.set_index('date',inplace=True)
    
    df.index=pd.to_datetime(df.index,format='%Y-%m-%d')
    
    dff = df[['ret','volume']].copy()

    return dff

# 预处理 
def  pre_process_data(df):
    
    # 把空值作为前值填充、，每列都相同的操作 
    df.fillna(method='ffill',inplace=True)
    
    df.to_csv('net_time_series.csv')

    return df    


## 数据进行重新采样 ，分为训练集和测试集
def  resampling_train_test(df):
    
    length=  len(df)
    
    train_len = int(length/8)*7 ## 
    
    train_data =df.loc[df.index[0]:df.index[train_len],:]
    
    test_data = df.loc[df.index[train_len+1]:df.index[-1],:]
    
    return train_data,test_data




## 确定阶数，根据bic,默认d=1
    
def determine_bic(timeseries):
        
    #设置遍历循环的初始条件，以热力图的形式展示，跟AIC定阶作用一样
    p_min = 0
    q_min = 0
    p_max = 3
    q_max = 3
    d = 1
    # 创建Dataframe,以BIC准则
    results_aic = pd.DataFrame(index=['AR{}'.format(i) \
                               for i in range(p_min,p_max+1)],\
            columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
    
    # itertools.product 返回p,q中的元素的笛卡尔积的元组
    for p,q in itertools.product(range(p_min,p_max+1),\
                                   range(q_min,q_max+1)):
        print(p,q)
        if p==0 and q==0:
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue
        try:
            model = sm.tsa.ARIMA(timeseries, order=(p, d, q))

            results = model.fit()
            #返回不同pq下的model的BIC值
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
        except:
            continue
        
    results_aic = results_aic[results_aic.columns].astype(float)
    
    p ,q= results_bic.stack().idxmin()
    
    p=int(p[-1])
    
    q=int(q[-1])
    
    return p,q,d

    
##  模型构建 ，应该是滑窗式进行构建模型预测 
    
def  ARIMA_Model(series,p=5,d=1,q=1):
    
    order =(p,d,q)
    
    arma_model = ARMA(series,order) #ARMA模型

    return arma_model


def model_fit(data):
    
    ## 需要改进，并不需要每次都跑模型 
    arma_model = ARIMA_Model(data)
    
    result = arma_model.fit()
    
    # 一步预测结果
    pre =  result.predict(start=len(data),end=len(data))
    
    return pre















if __name__=='__main__':

    code='000002'

    data=get_data(code)
    
    data=pre_process_data(data)
    
    train_data,test_data = resampling_train_test(data)
    
    p,q,d = determine_bic(train_data['ret'])
    
    
    #------------------------------------------------------------------------
    
    arma_model = ARMA(train_data['ret'],(1,1))
    
    result = arma_model.fit()

    merge_data=pd.concat([train_data.tail(20),test_data])
    
    ##  进行滑窗操作
    
    a=merge_data['ret'].rolling(window=5).apply(lambda x:model_fit(x))
    
    
    
    res = sm.tsa.ARMA(train_data['ret'], (5, 1)).fit()
    
    pred = res.predict()

    # get what you need for predicting one-step ahead
    params = res.params
    residuals = res.resid
    
    p = res.k_ar
    q = res.k_ma
    k_exog = res.k_exog
    k_trend = res.k_trend
    steps = 1
    
    a = _arma_predict_out_of_sample(params, steps, residuals, \
                               p, q, k_trend, k_exog,\
                               endog=data['ret'], exog=None, start=len(data))
    
    
    

    ## test   测试正确 
    
    new_resid =residuals['2019-11-02':'2019-11-14']
    
    new_data= train_data.loc['2019-11-02':'2019-11-14',:]
    
    a = _arma_predict_out_of_sample(params, steps, new_resid, \
                           p, q, k_trend, k_exog,\
                           endog = new_data['ret'], exog=None, start=len(new_data))



    test_pred = pred['2019-11-13':'2019-11-20']
    
    
    ##   重新开始 
    
    new_resid =  residuals.tail(10).copy()
    
    new_data = train_data.tail(10).copy()
    
    for  i in range(len(test_data)-1):
        
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
        
        

        
        
        
    
        
        
    
        
        
        
         
        
        
        
            
        
        
        
    
    
    
    

    

    

    
    

    

    
    
    
    
    
    
    
    
    
    

    
    
    
    
    

    
    
    

    
    
    
    
    
    
    
    
    
    
    

    
    
    

    
    

    
    

    
