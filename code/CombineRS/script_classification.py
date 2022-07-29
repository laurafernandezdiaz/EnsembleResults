# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 09:28:23 2020

@author: Y0633602
"""

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import random as random
from my_grid_combine import my_grid_combine
from Read_Datasets import ReadDatasets
import pickle

n_seed=2480 #seed selected

RG=random.Random(n_seed) #Random generator with the selected seed
seed=RG.randint(0,2**31) #First seed generated with the random generator

exp=8
withX=True
MLSName='KNeighbors'
dataname='breast_cancer' 
scoringName='accuracy'

if withX=='True':
    withX=True
if withX=='False':
    withX=False

    
if dataname=='iris':
    X_train, X_test, y_train, y_test = ReadDatasets.load_iris(seed)

if dataname=='breast_cancer':
    X_train, X_test, y_train, y_test = ReadDatasets.load_breast_cancer(seed)


if MLSName=='SVC':
    MLS=svm.SVC(kernel='linear')
    GSparams = {'C':[10**i for i in range(-3,3+1)]}
    
if MLSName=='KNeighbors':
    MLS=KNeighborsClassifier()
    GSparams = {'n_neighbors':[1,3,6,10,15]}

#------------------------------------------CV 10----------------------------
cv=10
rep=10
grid_combine=my_grid_combine(MLS,GSparams,RG,scoringName,withX,cv,exp,rep,X_train,y_train,X_test,y_test)

# Open file to save grid
with open(str(exp)+'/grid_classification_combine_'+str(dataname)+'_'+str(withX)+'_'+str(MLSName)+'_'+str(scoringName)+'_'+str(cv)+'_'+str(exp), 'wb') as file:
 
 
  # Save grid in disk
  pickle.dump(grid_combine, file)
