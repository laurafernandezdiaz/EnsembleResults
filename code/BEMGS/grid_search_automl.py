# -*- coding: utf-8 -*-

from .data_table import data_table
import numpy as np
from sklearn.base import clone
from time import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd
import os.path
import pickle
from sklearn.model_selection import ParameterGrid


def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.seed(1234)
    np.random.shuffle(arr)
    return arr

def concatena(X,P):
    XP=[]
    for i in range(len(X)):
        XP.append(X[i]+P[i])
    return XP
class grid_search_automl:

	
    def __init__(self,MLS,GSparams,RG,scoring,rep,cv_split=2,exp=1,fold=1):
        """
        Params:
            - MLS    : estimator type
            - GSparams     : params to use in the gridSearch
            - cv           : cross validation object
            - scoring      : score function to use
                       
                           
            Integer parameters:
            - rep          : Number of repetitions
            - cv_split     : Number os cross validation splits
            - exp          : Number of current experiment
            - fold         : Number of current fold

        """

        #Storing the parameters
        self.MLS=MLS #Estimator
        self.GSparams=GSparams #GridSearch params
        self.RG=RG #Cross validatin object
        self.scoring=scoring #Score function
        self.exp=exp
        self.rep=rep
        self.fold=fold
        self.cv_split=cv_split
        self.tol=0 #Tolerance to obtain coefs (stop criterium)
        #Creating the fields
        self.coef_grid_=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv.       
        self.best_estimator_grid_=[] #Attribute to save the best estimator of the normal gridsearchcv
       
        self.coef_grid_automl=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv
        self.estimators=[]
        
        
        params=[]
        for t in range(0,len(GSparams)):
            total_params=1
            for i in GSparams[t].keys():  
                total_params=total_params*len(GSparams[t][i])
            params.append(total_params)
        total_params=sum(params)        

        self.params=total_params   
         
    def fit(self,X,y):
        """
        Fit method
        
        Params:
                        Dataframe parameters:
            - X            : X data
            - y            : target data
        """
          # Checking parameters
        BaseErrMsg='Unsatisfied constraint in grid_search_automl.__fit(...)__: ';

        if type(X)!=np.ndarray or type(y)!=np.ndarray:
            raise grid_search_automlExcep(BaseErrMsg+'DataFrame attributes are not a numpy.ndarray');
        self.start_time_fit = time()
        self.X_train=X
        self.y_train=y
        
        self.exists=False
        os.chdir(os.path.dirname(__file__))
        self.path=str(os.getcwd())+'\\'+str(self.exp)+'\\'
        if os.path.isfile(self.path+'estimators_'+str(self.fold)+'.data'):
           
            pickle_file = open(self.path+'estimators_'+str(self.fold)+'.data','rb')
            self.estimators_fit = pickle.load(pickle_file)
            pickle_file.close()
            self.exists=True
        else:
                        
            self.estimators_fit=[]
               
            for g in ParameterGrid(self.GSparams):
                model=clone(self.MLS)
                #print("param",g)
                model.set_params(**g)
                model=model.fit(self.X_train,self.y_train)
                self.estimators_fit.append(model)

       
        if self.exists==False:
            pickle_file = open(self.path+'estimators_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.estimators_fit,pickle_file)
            pickle_file.close()
        
 
        self.final_time_fit = time() - self.start_time_fit
 
    def predict(self,X):
        """
        Predict method
        
        Params:
                        Dataframe parameters:
            - X            : X data
        Return:
            - Y_pred       : Predictions of Y (numpy.ndarray)
        """
        
        self.start_time_predict = time()
        n_fold=self.cv_split
        self.X_test=X
        self.y_pred = np.empty((X.shape[0], 1))  
        
        self.y_pred_test_params = np.empty((n_fold*self.X_test.shape[0], self.params)) 
        self.y_pred_test_params[:] = np.nan

        self.preds_test_grid=np.empty((1, self.params), dtype=object)

        self.preds_test=[]
        
        
      
        ids=np.array(list(range(0,self.X_test.shape[0]))).reshape(self.X_test.shape[0],1)
        self.X_test_ids=ids
      
                               
        self.preds_y=np.empty((self.X_test.shape[0],1), dtype=object) 
        

        self.preds_params=np.empty((self.X_test.shape[0], self.params), dtype=object)
        j=0
        for estimator in self.estimators_fit:

                self.preds_params[:,j]=estimator.predict(self.X_test)
                j=j+1

 
        self.final_time_predict = time() - self.start_time_predict
        return np.mean(self.preds_params,axis=1)

    def score(self,y_real,y_predict):
        if self.scoring=='r2_scorer':
            self.score=r2_score(y_real, y_predict)
        if self.scoring=='neg_mean_absolute_error':
            self.score=mean_absolute_error(y_real, y_predict)
        if self.scoring=='neg_mean_squared_error':
            self.score=mean_squared_error(y_real, y_predict)
        return self.score

#Pass exception        
class grid_search_automlExcep(Exception):
    pass