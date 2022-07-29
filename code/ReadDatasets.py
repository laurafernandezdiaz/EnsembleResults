# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
def fromObjecttoNumber(data,column):
    try:
         data_replace = data.copy()
         data_replace[column]=data_replace[column].astype(float)

    except:
        labels = data[column].astype('category').cat.categories.tolist()
        replace_map_comp = {column : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
        data_replace.replace(replace_map_comp, inplace=True)
    finally:
        return data_replace
    
class ReadDatasets():
     def load_automobile(seed): 
        porcentaje=0.33
        data = pd.read_csv("../datasets/Automobile/imports-85.data",sep=",",header=None,names=['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price'])
        
        data=data[data!='?'].dropna()
        
        data_copy = data.select_dtypes(include=['object']).copy()
        
        for col in data_copy:
            data=fromObjecttoNumber(data,col)
        
        data=data[data['price']!=5151]
        X=data[data.columns[0:25]]
        y=data['price']
        
        
        return X.to_numpy(), y.to_numpy()
        
     def load_fertility(seed):
        porcentaje=0.33
        data = pd.read_csv("../datasets/Fertility/fertility.csv",sep=",",names=['Season','Age','diseases','trauma','intervention','fevers','alcohol','Smoking','sitting','Output'])
        
        data=data[data!='?'].dropna()
        data_copy = data.select_dtypes(include=['object']).copy()
        for col in data_copy:
            data=fromObjecttoNumber(data,col)
        
        X=data[data.columns[0:9]]
        y=data['Output']
        
        
        return X.to_numpy(), y.to_numpy()
    
     def load_servo(seed):
        porcentaje=0.33
        data = pd.read_csv("../datasets/Servo/servo.data",sep=",",header=None,names=['motor','screw','pgain','vgain','class'])

        data_copy = data.select_dtypes(include=['object']).copy()
        for col in data_copy:
            data=fromObjecttoNumber(data,col)
        
        X=data[data.columns[0:4]]
    
        y=data['class']
    
        return X.to_numpy(), y.to_numpy()
    
     def load_wine_red(seed):
            porcentaje=0.33
            data = pd.read_csv("../datasets/WineQuality/winequality-red.csv",sep=";",header=None,names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'])
    
            data_copy = data.select_dtypes(include=['object']).copy()
            for col in data_copy:
                data=fromObjecttoNumber(data,col)
            
            X=data[data.columns[0:11]]
        
            y=data['quality']
        
            return X.to_numpy(), y.to_numpy()
        
     def load_wine_white(seed):
            porcentaje=0.33
            data = pd.read_csv("../datasets/WineQuality/winequality-white.csv",sep=";",header=None,names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'])
    
            data_copy = data.select_dtypes(include=['object']).copy()
            for col in data_copy:
                data=fromObjecttoNumber(data,col)
            
            X=data[data.columns[0:11]]
        
            y=data['quality']
            
            return X.to_numpy(), y.to_numpy()
  
     def load_forest(seed):
        porcentaje=0.33
        data = pd.read_csv("../datasets/Forest/forestfires.csv",sep=",",header=None,names=['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area'])
        
        data=data[data!='?'].dropna()
        data_copy = data.select_dtypes(include=['object']).copy()
        for col in data_copy:
            data=fromObjecttoNumber(data,col)
        X=data[data.columns[0:12]]
        y=data['area']
        return X.to_numpy(), y.to_numpy()
    
     def load_traffic(seed):
        porcentaje=0.33
        data = pd.read_csv("../datasets/Traffic/traffic.csv",sep=";",header=None,names=list(range(0,18,1)))
        data=data[data!='?'].dropna()
        data_copy = data.select_dtypes(include=['object']).copy()
        for col in data_copy:
            data=fromObjecttoNumber(data,col)
        X=data[data.columns[0:17]]
        y=data[17]

        return X.to_numpy(), y.to_numpy()

     def load_slump(seed):
        porcentaje=0.33
        data = pd.read_csv("../datasets/slump/slump1.csv",sep=";",header=None,names=list(range(0,8,1)))
        data=data[data!='?'].dropna()
        data_copy = data.select_dtypes(include=['object']).copy()
        for col in data_copy:
            data=fromObjecttoNumber(data,col)
        X=data[data.columns[0:7]]
        y=data[7]
        return X.to_numpy(), y.to_numpy()
    
     def load_flow(seed):
        porcentaje=0.33
        data = pd.read_csv("../datasets/flow/flow.csv",sep=";",header=None,names=list(range(0,8,1)))
        data=data[data!='?'].dropna()
        data_copy = data.select_dtypes(include=['object']).copy()
        for col in data_copy:
            data=fromObjecttoNumber(data,col)
        X=data[data.columns[0:7]]
        y=data[7]

        return X.to_numpy(), y.to_numpy()
  