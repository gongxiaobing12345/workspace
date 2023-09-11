#!/home/svu/e1127291/.conda/miniconda/4.9/envs/myenv3.8/bin/python3.8
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:49:07 2022

@author: 巩晓冰
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import keras as keras

from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
import time


data = pd.read_csv("PG.csv", encoding='gbk')
data = data['收盘']

seq_len = 60
res = []
for i in range(len(data)-seq_len-1+1):
    res.append(data[i:i+seq_len+1])

data_min=[]
data_max=[]
ad_res=[]
for i in range(np.array(res).shape[0]):
    datai=res[i]
    datai=datai.reset_index(drop=True)
    data_min.append(min(datai))
    data_max.append(max(datai))
    resi=[]
    for j in range(datai.shape[0]):
        resi.append((datai[j]-min(datai))/(max(datai)-min(datai)))
    ad_res.append(resi)
    
result=np.array(ad_res)
row=result.shape[0]-301
train=result[:row,:]
np.random.shuffle(train)
train_x=train[:,:-1]
train_y=train[:,-1]
test=result[row:,:]
test_x=test[:,:-1]
test_y=test[:,-1]
train_x=np.reshape(train_x, (train_x.shape[0],train_x.shape[1],1))
test_x=np.reshape(test_x, (test_x.shape[0],test_x.shape[1],1))

model=Sequential()
model.add(LSTM(units=60,input_shape=(train_x.shape[1],train_x.shape[2]),return_sequences=True))
model.add(Dropout(0))
model.add(LSTM(units=60,return_sequences=False))
model.add(Dense(1))
model.add(Activation('tanh'))
start=time.time()
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(train_x,train_y,batch_size=1,epochs=1,validation_split=0.05)
def predict_point_by_point(model,data):
    predicted=model.predict(data)
    predicted=np.reshape(predicted,(predicted.size,))
    return predicted

predictions=predict_point_by_point(model,test_x)

truth=[]
forecast=[]

for i in range(len(test_y)):
    truth.append(test_y[i]*(data_max[train.shape[0]+i]-data_min[train.shape[0]+i])+data_min[train.shape[0]+i])
    forecast.append(predictions[i]*(data_max[train.shape[0]+i]-data_min[train.shape[0]+i])+data_min[train.shape[0]+i])
    
print(model.evaluate(test_x,test_y))

# windowlen=int(len(data)/3)
# serieslen=len(data)
# K=serieslen-windowlen+1
# X=np.zeros((windowlen,K))
#
# for i in range(K):
#     X[:,i]=data[i:i+windowlen]
#
# '''svd分解'''
# U,sigma,VT=np.linalg.svd(X,full_matrices=False)
# for i in range(VT.shape[0]):
#     VT[i,:]*=sigma[i]
# A=VT
#
# '''分组'''
# rec=np.zeros((windowlen,serieslen))
# for i in range(windowlen):
#     for j in range(windowlen-1):
#         for m in range(j+1):
#             rec[i,j]+=A[i,j-m]*U[m,i]
#         rec[i,j]/=(j+1)
#     for j in range(windowlen-1,serieslen-windowlen+1):
#         for m in range(windowlen):
#             rec[i,j]+=A[i,j-m]*U[m,i]
#         rec[i,j]/=windowlen
#     for j in range(serieslen-windowlen+1,serieslen):
#         for m in range(j-serieslen+windowlen,windowlen):
#             rec[i,j]+=A[i,j-m]*U[m,i]
#         rec[i,j]/=(serieslen-j)
#
# '''重构'''
# scf=sigma/sum(sigma)*100
# cscf=[]
# for i in range(len(scf)):
#     cscfi=sum(scf[0:i+1])/sum(scf)*100
#     cscf.append(cscfi)
# cscf=pd.Series(cscf)
# n1=cscf[cscf<=85].shape[0]
# data_trend=np.sum(rec[0:n1],axis=0)
# data_residual=np.sum(rec[n1:],axis=0)
# dec=pd.DataFrame()
# dec['data']=data
# dec['data_trend']=data_trend
# dec['data_residual']=data_residual
#
# dec.to_excel("svd_data.xlsx")
#
# pred_data=pd.DataFrame(forecast)
#
# pred_data.to_excel("LSTM_trend_PRED_data.xlsx")



