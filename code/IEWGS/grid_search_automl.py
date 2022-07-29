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
        
        if self.scoring=='r2_scorer':
            self.score_function=r2_score
        if self.scoring=='neg_mean_absolute_error':
            self.score_function=mean_absolute_error
        if self.scoring=='neg_mean_squared_error':
            self.score_function=mean_squared_error
            
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
        if os.path.isfile(self.path+'preds_'+str(self.fold)+'.csv'):
            self.X=pd.read_csv(self.path+'preds_'+str(self.fold)+'.csv') 
            self.y=pd.read_csv(self.path+'reals_'+str(self.fold)+'.csv')
            pickle_file = open(self.path+'estimators_'+str(self.fold)+'.data','rb')
            self.estimators_fit = pickle.load(pickle_file)
            pickle_file.close()
            self.exists=True
        else:
            
            self.table=data_table(self.MLS,self.X_train,self.y_train,self.GSparams,self.RG,self.scoring,self.params,self.rep,self.RG,self.exp) #Call to init the object to do the gridsearch
            
            self.table.fit() #Fit method to do the gridsearch
            
            #GRID
            scores=[] #Array to save the scores from the gridsearch
            scores.append(self.table.best_estimator_grid_.score(self.X_train,self.y_train))
    
            self.best_estimator_grid_=max(scores) #Best score
            self.fit_times=self.table.fit_times
            self.score_times=self.table.score_times
            
            print("fit times:")
            print(self.fit_times)        
            print("score times:")
            print(self.score_times)
            self.estimators=self.table.estimators
            
            self.X=self.table.evals_pred #The predicted evaluations will be the Xs
            self.y=self.table.evals_real #The real evaluations will be the target
            self.folds=self.table.folds            
            
            self.estimators_fit=[]
               
            for g in ParameterGrid(self.GSparams):
                model=clone(self.MLS)
                model.set_params(**g)
                model=model.fit(self.X_train,self.y_train)
                self.estimators_fit.append(model)
                  
        X_dataframe=pd.DataFrame(self.X)
        y_dataframe=pd.DataFrame(self.y)
        if self.exists==False:
            X_dataframe.to_csv(self.path+'preds_'+str(self.fold)+'.csv', index=False)
            y_dataframe=pd.DataFrame(np.asmatrix(self.y).transpose())
            y_dataframe.to_csv(self.path+'reals_'+str(self.fold)+'.csv', index=False) 
            pickle_file = open(self.path+'estimators_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.estimators_fit,pickle_file)
            pickle_file.close()
        
        self.inversos=np.empty((self.params), dtype=object)   
        
        self.X=np.array(self.X)
        for i in range(0,self.inversos.shape[0]):
            self.inversos[i]=self.score_function(self.X[:,i],self.y)
        

        if np.count_nonzero(self.inversos) <len(self.inversos):  
            self.inversos[self.inversos == 0] = 1
            self.inversos[self.inversos != 0] = 0
        
        self.inversos=(1/self.inversos)/sum(1/self.inversos)
        
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
        self.X_test=X
                               
        self.preds_params=np.empty((self.X_test.shape[0], self.params), dtype=object)
  
        j=0
        for estimator in self.estimators_fit:
            self.preds_params[:,j]=estimator.predict(self.X_test)
            j=j+1
                 
        self.preds_params_pond=np.empty((self.X_test.shape[0], self.params), dtype=object)

        for i in range(0,self.preds_params.shape[0]):
            for j in range(0,self.preds_params.shape[1]):
                self.preds_params_pond[i][j]=self.preds_params[i][j]*self.inversos[j]
        
        self.final_time_predict = time() - self.start_time_predict
        
        return np.sum(self.preds_params_pond,axis=1)

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