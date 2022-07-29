# -*- coding: utf-8 -*-

from sklearn import svm,linear_model,datasets
from sklearn.model_selection import KFold
import random as random
from BEMRS.my_random_combine import my_random_combine
from sklearn.ensemble import RandomForestRegressor
from ReadDatasets import ReadDatasets
import pickle
from sklearn.preprocessing import StandardScaler
import math
import sys
import pathlib
from numpy import arange
from scipy.stats import uniform
import numpy as np
n_seed=2480 #seed selected

class ExpUniform:
    def __init__(self,loc,scale):
        self.uDis=uniform(loc=loc,scale=scale)
        
    def rvs(self,random_state=None):
        u=self.uDis.rvs(random_state=random_state)
        e=10**u
        return e
class ExpUniformRF:
    def __init__(self,loc,scale):
        self.uDis=uniform(loc=loc,scale=scale)
        
    def rvs(self,random_state=None):
        u=self.uDis.rvs(random_state=random_state)
        e=2**u
        return e   

RG=random.Random(n_seed) #Random generator with the selected seed
seed=RG.randint(0,2**31) #First seed generated with the random generator


"""
exp=700
MLSName="ridge"
dataname="automobile"
scoringName="neg_mean_squared_error"

"""
exp=sys.argv[1]
MLSName=sys.argv[3]
dataname=sys.argv[2] 
scoringName=sys.argv[4]

if dataname=='automobile':
    X,y = ReadDatasets.load_automobile(seed)

if dataname=='fertility':
    X,y  = ReadDatasets.load_fertility(seed)      

if dataname=='servo':
    X,y = ReadDatasets.load_servo(seed)
     
if dataname=='wine_red':
    X,y = ReadDatasets.load_wine_red(seed)
    
if dataname=='wine_white':
    X,y = ReadDatasets.load_wine_white(seed)
	
if dataname=='forest':
    X,y = ReadDatasets.load_forest(seed)     

if dataname=='traffic':
    X,y = ReadDatasets.load_traffic(seed)     
if dataname=='slump':
    X,y = ReadDatasets.load_slump(seed)     
if dataname=='flow':
    X,y = ReadDatasets.load_flow(seed)     


if MLSName=='ridge':
    MLS=linear_model.Ridge(random_state=seed) 
    GSparams = {'alpha': np.linspace(0.01, 1) , "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}# {'alpha':[math.exp(i) for i in range(-6,6+1)]}

if MLSName=='SVR':

    MLS=svm.SVR()
    GSparams = {'C':ExpUniform(loc=-3,scale=6),
                'kernel':['rbf','linear'],
                'gamma': np.linspace(0.01, 1)} 
    
if MLSName=='RandomForest':
    MLS=RandomForestRegressor(n_estimators=10, random_state=seed)
    GSparams = {'min_samples_leaf':ExpUniformRF(loc=-8,scale=6),
                 'max_features':np.linspace(0.2, 0.9999)}
      

#------------------------------------------CV 3----------------------------
cv=3
rep=1
random_combine=my_random_combine(MLS,GSparams,RG,scoringName,cv,exp,rep,X,y)


sumfit=0
sumpred=0
for i in range(0,rep):
	sumfit=sumfit+random_combine.random_combine_models[i].final_time_fit
	sumpred=sumpred+random_combine.random_combine_models[i].final_time_predict
print("Time fit per rep",sumfit/rep)
print("Time predict per rep",sumpred/rep)

path=pathlib.Path(__file__).parent.absolute()

# Open file to save random
with open(str(path)+'\\'+str(exp)+'\\random_regression_combine_'+str(dataname)+'_'+str(MLSName)+'_'+str(scoringName)+'_'+str(cv)+'_'+str(exp), 'wb') as file:
 
 
  # Save random in disk
  pickle.dump(random_combine, file)
