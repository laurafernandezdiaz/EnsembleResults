# -*- coding: utf-8 -*-

from hyperband import HyperbandSearchCV
import numpy as np
from sklearn.base import clone
from time import time
from sklearn import svm,linear_model,datasets
from sklearn.ensemble import RandomForestRegressor
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
class hyperband_automl:


    def __init__(self,MLS,GSparams,scoring,cv_split=3,rep=1,fold=1,exp=1):
        """
        Params:
            - MLS    : estimator type
            - GSparams     : params to use in the gridSearch
            - cv           : cross validation object
            - scoring      : score function to use
            
            Integer parameters:
            - rep          : Number of repetitions

        """
        #Storing the parameters
        self.MLS=MLS #Estimator
        self.GSparams=GSparams #GridSearch params
        #self.RG=RG #Cross validatin object
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
            pickle_file2 = open(self.path+'params_'+str(self.fold)+'.data','rb')
            pickle_file3 = open(self.path+'estimators_'+str(self.fold)+'.data','rb')
            self.params_fit = pickle.load(pickle_file2)
            self.estimators_fit = pickle.load(pickle_file3)
            pickle_file2.close()
            pickle_file3.close()
            self.exists=True
        else:                               
            if self.scoring=='neg_mean_squared_error':
                    if isinstance(self.MLS,svm.SVR):
                        points=35
                        self.search = HyperbandSearchCV(self.MLS, param_distributions=self.GSparams, pre_dispatch=1,
                        resource_param='max_iter',min_iter=int(100000),max_iter=3000000,verbose=1,
                        scoring=self.scoring)
                        self.search.fit(self.X_train,self.y_train)
                        self.params_fit=[]
                        self.estimators_fit=[]
                   
                        self.estimators_fit.append(self.search.best_estimator_)
                             

                                                
                    if isinstance(self.MLS,linear_model.Ridge):
                        points=36
                        self.search = HyperbandSearchCV(self.MLS, param_distributions=self.GSparams, pre_dispatch=1,
                        resource_param='max_iter',min_iter=int(100000),max_iter=3000000,verbose=1,
                        scoring=self.scoring)
                        self.search.fit(self.X_train,self.y_train)
                        self.params_fit=[]
                        self.estimators_fit=[]
                       
                        self.estimators_fit.append(self.search.best_estimator_)
                        
                         
                    if isinstance(self.MLS,RandomForestRegressor):
                        points=35
                        self.search = HyperbandSearchCV(self.MLS, param_distributions=self.GSparams, pre_dispatch=1,
                        resource_param='max_leaf_nodes',min_iter=int(100000),max_iter=3000000,verbose=1,
                        scoring=self.scoring)
                        self.search.fit(self.X_train,self.y_train)
                        self.params_fit=[]
                        self.estimators_fit=[]

                        self.estimators_fit.append(self.search.best_estimator_)

                        
        self.X=self.X_train
        self.y=self.y_train
        X_dataframe=pd.DataFrame(self.X)
        y_dataframe=pd.DataFrame(self.y)
        if self.exists==False:
            X_dataframe.to_csv(self.path+'preds_'+str(self.fold)+'.csv', index=False)
            y_dataframe=pd.DataFrame(np.asmatrix(self.y).transpose())
            y_dataframe.to_csv(self.path+'reals_'+str(self.fold)+'.csv', index=False) 
            pickle_file2 = open(self.path+'params_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.params_fit,pickle_file2)
            pickle_file3 = open(self.path+'estimators_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.estimators_fit,pickle_file3)
            pickle_file2.close()
            pickle_file3.close()
        
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
        
        self.y_pred_test_params = np.empty((n_fold*self.X_test.shape[0], len(self.params_fit))) 
        self.y_pred_test_params[:] = np.nan

        self.preds_test_grid=np.empty((1, len(self.params_fit)), dtype=object)
      
        self.preds_test=[]
     
        ids=np.array(list(range(0,self.X_test.shape[0]))).reshape(self.X_test.shape[0],1)
        self.X_test_ids=ids
          
        self.preds_y=np.empty((self.X_test.shape[0],1), dtype=object) 
      
        self.preds_params=np.empty((self.X_test.shape[0], len(self.params_fit)), dtype=object)
            

        self.y_pred_results=self.estimators_fit[0].predict(self.X_test)
     
        self.final_time_predict = time() - self.start_time_predict
        return self.y_pred_results

    def score(self,y_real,y_predict):
        if self.scoring=='r2_scorer':
            self.score=r2_score(y_real, y_predict)
        if self.scoring=='neg_mean_absolute_error':
            self.score=mean_absolute_error(y_real, y_predict)
        if self.scoring=='neg_mean_squared_error':
            self.score=mean_squared_error(y_real, y_predict)
        return self.score

#Pass exception        
class hyperband_automlExcep(Exception):
    pass