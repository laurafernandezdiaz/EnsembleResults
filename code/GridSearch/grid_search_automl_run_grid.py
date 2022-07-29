# -*- coding: utf-8 -*-

from .data_table import data_table
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from time import time
import os.path
import pickle

class grid_search_automl_run_grid:
    
    def __init__(self,MLS,GSparams,RG,scoring,rep,cv_split=10,exp=1,fold=1):
        """
        Params:
            - MLS    : estimator type
            - GSparams     : Params to use in the gridSearch
            - RG           : Cross validatin object
            - scoring      : Score function to use
            
                           
            Integer parameters:
            - rep          : Number of repetitions
            - cv_split     : Number os cross validation splits
            - exp          : Number of current experiment
            - fold         : Number of current fold


            
            On params errors in type or values a grid_search_automlExcep exception will be raised.
        """

        # Checking types      
        
        #Storing the parameters
        self.MLS=MLS #Estimator
        self.GSparams=GSparams #GridSearch params
        self.RG=RG #Cross validatin object
        self.scoring=scoring #Score function
        self.rep=rep
        self.fold=fold
        self.exp=exp
        self.cv_split=cv_split
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
            raise grid_search_automl_run_gridExcep(BaseErrMsg+'DataFrame attributes are not a numpy.ndarray');
        self.start_time_fit = time()
        self.X_train=X
        self.y_train=y

        os.chdir(os.path.dirname(__file__))
        self.path=str(os.getcwd())+'\\'+str(self.exp)+'\\'
        if os.path.isfile(self.path+'estimator_'+str(self.fold)+'.data'):

            pickle_file = open(self.path+'estimator_'+str(self.fold)+'.data','rb')
            self.best_estimators_grid_ = pickle.load(pickle_file)
            pickle_file.close()

        else:
            self.table=data_table(self.MLS,self.X_train,self.y_train,self.GSparams,self.RG,self.scoring,self.params,self.rep,self.cv_split,self.exp) #Call to init the object to do the gridsearch
            
            self.table.fit() #Fit method to do the gridsearch
            
            
            #GRID
            scores=[] #Array to save the scores from the gridsearch

            scores.append(self.table.best_estimator_grid_.score(self.X_train,self.y_train))
    
            self.best_estimators_grid_=self.table.best_estimator_grid_
        
            self.fit_times=self.table.fit_times
            self.score_times=self.table.score_times

            pickle_file = open(self.path+'estimator_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.best_estimators_grid_,pickle_file)
            pickle_file.close()
            print("fit times:")
            print(self.fit_times)        
            print("score times:")
            print(self.score_times)
        
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

        self.X_test=X
        self.start_time_predict = time()
        self.y_pred_all = np.empty((self.X_test.shape[0], 1))

        self.estimator_rep=self.best_estimators_grid_
        self.y_pred_all[:,0]=self.estimator_rep.predict(self.X_test).reshape(self.X_test.shape[0],)

     
        self.final_time_predict = time() - self.start_time_predict
        return self.y_pred_all
    


    
    def score(self,y_real,y_predict):
        if self.scoring=='r2_scorer':
            self.score=r2_score(y_real, y_predict)
        if self.scoring=='neg_mean_absolute_error':
            self.score=mean_absolute_error(y_real, y_predict)
        if self.scoring=='neg_mean_squared_error':
            self.score=mean_squared_error(y_real, y_predict)
        return self.score
    
#Pass exception        
class grid_search_automl_run_gridExcep(Exception):
    pass