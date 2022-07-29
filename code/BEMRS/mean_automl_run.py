# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import ParameterGrid
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from time import time
from sklearn.base import clone
import pandas as pd
import os.path
import pickle

class mean_automl_run:
    
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
        self.rep=rep #Number of repetitions
        self.exp=exp
        self.fold=fold
        self.cv_split=cv_split
        #Creating the fields
        self.coef_grid_=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv.       
        self.best_estimator_grid_=[] #Attribute to save the best estimator of the normal gridsearchcv
        self.coef_grid_automl=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv
        self.estimators=[]
        
       
    def fit(self,X,y):
        """
        Fit method
        
        Params:
                        Dataframe parameters:
            - X            : X data
            - y            : target data
        """
          # Checking parameters
        BaseErrMsg='Unsatisfied constraint in mean_automl_run.__fit(...)__: ';

        if type(X)!=np.ndarray or type(y)!=np.ndarray:
            raise mean_automl_runExcep(BaseErrMsg+'DataFrame attributes are not a numpy.ndarray');
        self.start_time_fit = time()

        self.X_train=X
        self.y_train=y

        self.mean=np.mean(self.y_train)
        self.final_time_fit = time() - self.start_time_fit
    def predict(self,X,y):
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
        self.y_test=y
        
        
        contador=0

        self.y_pred_all = np.empty((self.y_test.shape[0]))
        for i in self.y_pred_all:
            self.y_pred_all[contador]=self.mean

            contador=contador+1
   
        self.final_time_predict = time() - self.start_time_predict
        return self.y_pred_all

    
    def _score(self,y_real,y_predict):
        if self.scoring=='r2_scorer':
            self.score=r2_score(y_real, y_predict)
        if self.scoring=='neg_mean_absolute_error':
            self.score=mean_absolute_error(y_real, y_predict)
        if self.scoring=='neg_mean_squared_error':
            self.score=mean_squared_error(y_real, y_predict)
        return self.score
#Pass exception        
class mean_automl_runExcep(Exception):
    pass