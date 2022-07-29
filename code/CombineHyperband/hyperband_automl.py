# -*- coding: utf-8 -*-


from .data_table import data_table
import numpy as np
from sklearn import svm,linear_model,datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from time import time
import random
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd
import os.path
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn import svm,linear_model
from .OLS import OLS;
from .GEM import GEM
from .Caruana import Caruana
from .BST import BST
from .RBST import RBST
from .FSCriterium import FSCriterium
from .PLSCriterium import PLSCriterium
from .PCACriterium import PCACriterium

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

	
    def __init__(self,MLS,GSparams,RG,scoring,rep,cv_split=2,exp=1,wrapper='Mean',fold=1):
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
        self.wrapper=wrapper #Wrapper to use
        #Creating the fields
        self.coef_grid_=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv.       
        self.best_estimator_grid_=[] #Attribute to save the best estimator of the normal gridsearchcv 
        self.coef_grid_automl=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv
        self.estimators=[]

        #Number of total values for params 
        total_params=1
        for i in GSparams.keys():
            params=0        
            for j in GSparams[i]: 
                  params=params+1 
            total_params=total_params*params
        
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
        
        if isinstance(self.MLS,svm.SVR):
            points=35
                       
        if isinstance(self.MLS,linear_model.Ridge):
            points=36
        
        if isinstance(self.MLS,RandomForestRegressor):
            points=35  
            
        self.exists=False
        os.chdir(os.path.dirname(__file__))
        self.path=str(os.getcwd())+'\\'+str(self.wrapper)+'\\'+str(self.exp)+'\\'
     
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

            
            self.params_fit=[]
            self.estimators_fit=[]
            contador=0
            for i in self.table.clf.cv_results_['params']:
                if contador<points:
                    model=clone(self.MLS)
                    model.set_params(**i)
                    self.params_fit.append(i)        
                    model=model.fit(self.X_train,self.y_train)  
                    self.estimators_fit.append(model)
                    contador=contador+1
                else:
                    break 
           
        X_dataframe=pd.DataFrame(self.X)
        y_dataframe=pd.DataFrame(self.y)
        if self.exists==False:
            X_dataframe.to_csv(self.path+'preds_'+str(self.fold)+'.csv', index=False)
            y_dataframe=pd.DataFrame(np.asmatrix(self.y).transpose())
            y_dataframe.to_csv(self.path+'reals_'+str(self.fold)+'.csv', index=False) 
            pickle_file = open(self.path+'estimators_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.estimators_fit,pickle_file)
            pickle_file.close()
        
        
        self.xs_wrapper=np.zeros((self.X_train.shape[1],), dtype=int)
     
           
        if self.wrapper=='OLS':
            self.Mywrapper=OLS()
        
        if self.wrapper=='GEM':
            self.Mywrapper=GEM()
               
        if self.wrapper=='BST_ICM':
            expError=2
            self.Mywrapper=BST(expError=expError,readjustAtt=True,stop=1,maxIters=5000)  
               
        if self.wrapper=='BST_AICC':
            expError=2
            self.Mywrapper=BST(expError=expError,readjustAtt=True,stop=3)  
            
        if self.wrapper=='BST_AIC':
            expError=2
            self.Mywrapper=BST(expError=expError,readjustAtt=True,stop=4)  
            
        if self.wrapper=='BST_BIC':
            expError=2
            self.Mywrapper=BST(expError=expError,readjustAtt=True,stop=5)  
          
        if self.wrapper=='BST_HQIC':
            expError=2
            self.Mywrapper=BST(expError=expError,readjustAtt=True,stop=6)  
            
        if self.wrapper=='BST_GMDL':
            expError=2
            self.Mywrapper=BST(expError=expError,readjustAtt=True,stop=7)  
            
        if self.wrapper=='Caruana':
            RO=random.Random()
            RO.seed(200)
            if self.scoring=='r2_scorer':
                 self.score_function=r2_score
            if self.scoring=='neg_mean_absolute_error':
                self.score_function=mean_absolute_error
            if self.scoring=='neg_mean_squared_error':
                self.score_function=mean_squared_error
            self.Mywrapper=Caruana(randomObject=RO,score_function=self.score_function,s=None,rep=True,bags=1,r=1)  
               
 
        if self.wrapper=='FS_AICC':
            expError=2
            self.Mywrapper=FSCriterium(expError=expError,stop=3)      
        
        if self.wrapper=='FS_AIC':
            expError=2
            self.Mywrapper=FSCriterium(expError=expError,stop=4)
        
        if self.wrapper=='FS_BIC':
            expError=2
            self.Mywrapper=FSCriterium(expError=expError,stop=5)
            
        if self.wrapper=='FS_HQIC':
            expError=2
            self.Mywrapper=FSCriterium(expError=expError,stop=6)
        
        if self.wrapper=='FS_GMDL':
            expError=2
            self.Mywrapper=FSCriterium(expError=expError,stop=7)
        
        if self.wrapper=='PLS_AICC':
            expError=2
            self.Mywrapper=PLSCriterium(stop=3)  
            
        if self.wrapper=='PLS_AIC':
            expError=2
            self.Mywrapper=PLSCriterium(stop=4)
        
        if self.wrapper=='PLS_BIC':
            expError=2
            self.Mywrapper=PLSCriterium(stop=5)
         
        if self.wrapper=='PLS_HQIC':
            expError=2
            self.Mywrapper=PLSCriterium(stop=6)
        
        if self.wrapper=='PLS_GMDL':
            expError=2
            self.Mywrapper=PLSCriterium(stop=7)
        
        if self.wrapper=='PCR_AICC':
            expError=2
            self.Mywrapper=PCACriterium(stop=3) 
            
        if self.wrapper=='PCR_AIC':
            expError=2
            self.Mywrapper=PCACriterium(stop=4)
        
        if self.wrapper=='PCR_BIC':
            expError=2
            self.Mywrapper=PCACriterium(stop=5)
       
        if self.wrapper=='PCR_HQIC':
            expError=2
            self.Mywrapper=PCACriterium(stop=6)
        
        if self.wrapper=='PCR_GMDL':
            expError=2
            self.Mywrapper=PCACriterium(stop=7)    
            
            
        if self.wrapper=='RBST_ICM':
            expError=2
            self.Mywrapper=RBST(expError=expError,readjustAtt=True,stop=1,maxIters=5000,reg=True)          
       
        if self.wrapper=='RBST_AICC':
            expError=2
            self.Mywrapper=RBST(expError=expError,readjustAtt=True,stop=3,maxIters=5000,reg=True)  
        
        if self.wrapper=='RBST_AIC':
            expError=2
            self.Mywrapper=RBST(expError=expError,readjustAtt=True,stop=4,maxIters=5000,reg=True)  
        
        if self.wrapper=='RBST_BIC':
            expError=2
            self.Mywrapper=RBST(expError=expError,readjustAtt=True,stop=5,maxIters=5000,reg=True)  
        
        if self.wrapper=='RBST_HQIC':
            expError=2
            self.Mywrapper=RBST(expError=expError,readjustAtt=True,stop=6,maxIters=5000,reg=True)  
        
        if self.wrapper=='RBST_GMDL':
            expError=2
            self.Mywrapper=RBST(expError=expError,readjustAtt=True,stop=7,maxIters=5000,reg=True) 
        
         
        X_dataframe=X_dataframe.iloc[:,:points]              
        
        self.Mywrapper.fit(np.asarray(X_dataframe), np.asarray(y_dataframe))
        
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
        
        contador=0
        self.preds_params=[]
        
        j=0
       
        for estimator in self.estimators_fit:         
                self.preds_params.append(estimator.predict(self.X_test))

        self.preds_params=pd.DataFrame(self.preds_params).T.to_numpy()
        contador=contador+n_fold
                       
        self.preds_params=self.preds_params
  
        self.y_pred_results=self.Mywrapper.predict(np.asarray(self.preds_params))


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