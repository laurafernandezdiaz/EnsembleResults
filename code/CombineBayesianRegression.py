# -*- coding: utf-8 -*-

from sklearn import svm,linear_model,datasets
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import random as random
from CombineBayesian.my_bayesian_combine import my_bayesian_combine
from ReadDatasets import ReadDatasets
import pickle
from sklearn.preprocessing import StandardScaler
import math
import sys
import pathlib
n_seed=2480 #seed selected

RG=random.Random(n_seed) #Random generator with the selected seed
seed=RG.randint(0,2**31) #First seed generated with the random generator

"""
exp=700
MLSName="ridge"
dataname="automobile"
scoringName="neg_mean_squared_error"
wrapper='RBST_ICM'
"""
exp=sys.argv[1]
MLSName=sys.argv[3]
dataname=sys.argv[2] 
scoringName=sys.argv[4]
wrapper=sys.argv[5]

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
    GSparams = [{'alpha': [1,0.1,0.01,0.001,0.0001,0] , "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}]# {'alpha':[math.exp(i) for i in range(-6,6+1)]}

if MLSName=='SVR':

    MLS=svm.SVR(kernel='linear',verbose=0)#svm.SVR()
    GSparams = [{'C':[10**i for i in range(-3,3+1)],
                'kernel':['rbf'],
                'gamma': [1,0.1,0.01,0.001]},{'C':[10**i for i in range(-3,3+1)],
                'kernel':['linear']}]
    
if MLSName=='RandomForest':
    MLS=RandomForestRegressor(n_estimators=10, random_state=seed)
    GSparams = [{'min_samples_leaf':[2**i for i in range(-8,-1)],
                 'max_features':[0.9999,0.8,0.6,0.4,0.2]}]
#------------------------------------------CV 3----------------------------
cv=3
rep=1
grid_combine=my_bayesian_combine(MLS,GSparams,RG,scoringName,cv,exp,rep,X,y,wrapper)


sumfit=0
sumpred=0
for i in range(0,rep):
	sumfit=sumfit+grid_combine.grid_combine_models[i].final_time_fit
	sumpred=sumpred+grid_combine.grid_combine_models[i].final_time_predict
print("Time fit per rep",sumfit/rep)
print("Time predict per rep",sumpred/rep)

path=pathlib.Path(__file__).parent.absolute()
# Open file to save grid
with open(str(path)+'\\'+str(wrapper)+'\\'+str(exp)+'\\grid_regression_combine_'+str(wrapper)+'_'+str(dataname)+'_'+str(MLSName)+'_'+str(scoringName)+'_'+str(cv)+'_'+str(exp), 'wb') as file:
 
 
  # Save grid in disk
  pickle.dump(grid_combine, file)
