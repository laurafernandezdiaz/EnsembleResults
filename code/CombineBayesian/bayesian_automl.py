# -*- coding: utf-8 -*-

from .BayesianHPO import BayesianOptForestParam,BayesianOptAlphaSolverParam,BayesianOptCGammaKernelParam
import numpy as np
from sklearn.base import clone
from time import time
import random
from sklearn import svm,linear_model,datasets
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd
import os.path
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn import svm,linear_model
from sklearn.ensemble import RandomForestRegressor
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
class bayesian_automl:


    def __init__(self,MLS,GSparams,scoring,cv_split=3,rep=1,fold=1,exp=1,wrapper='Mean'):
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
        self.path=str(os.getcwd())+'\\'+str(self.wrapper)+'\\'+str(self.exp)+'\\'
        if os.path.isfile(self.path+'preds_'+str(self.fold)+'.csv'):
            self.X=pd.read_csv(self.path+'preds_'+str(self.fold)+'.csv') 
            self.y=pd.read_csv(self.path+'reals_'+str(self.fold)+'.csv')
            pickle_file = open(self.path+'base_'+str(self.fold)+'.data','rb')
            pickle_file2 = open(self.path+'params_'+str(self.fold)+'.data','rb')
            pickle_file3 = open(self.path+'estimators_'+str(self.fold)+'.data','rb')
            self.base_fit = pickle.load(pickle_file)
            self.params_fit = pickle.load(pickle_file2)
            self.estimators_fit = pickle.load(pickle_file3)
            pickle_file.close()
            pickle_file2.close()
            pickle_file3.close()
            self.exists=True
        else:
            
            if self.scoring=='neg_mean_absolute_error':
                    if isinstance(self.MLS,svm.SVR):
                        points=35
                        self.grid_bayesian=BayesianOptCGammaKernelParam(self.MLS,(-3,3),(0.55,2),(0.001,1),self.X_train,self.y_train,points,mean_absolute_error,-1,self.cv_split,0,0,self.rep,self.fold)
                        
                    if isinstance(self.MLS,linear_model.Ridge):
                        points=36
                        self.grid_bayesian=BayesianOptAlphaSolverParam(self.MLS,(0.01, 1),(1,6),self.X_train,self.y_train,points,mean_absolute_error,-1,self.cv_split,0,0,self.rep,self.fold)
                        
                    
                    if isinstance(self.MLS,RandomForestRegressor):
                        points=35 
                        self.grid_bayesian=BayesianOptForestParam(self.MLS,(0.2,0.9999),(0.00390625,0.25),self.X_train,self.y_train,points,mean_absolute_error,-1,self.cv_split,0,0,self.rep,self.fold)

                                               
            if self.scoring=='neg_mean_squared_error':
                    if isinstance(self.MLS,svm.SVR):
                        points=35
                        self.grid_bayesian=BayesianOptCGammaKernelParam(self.MLS,(-3,3),(0.55,2),(0.001,1),self.X_train,self.y_train,points,mean_squared_error,-1,self.cv_split,0,0,self.rep,self.fold)
                        
                    if isinstance(self.MLS,linear_model.Ridge):
                        points=36
                        self.grid_bayesian=BayesianOptAlphaSolverParam(self.MLS,(0.01, 1),(1,6),self.X_train,self.y_train,points,mean_squared_error,-1,self.cv_split,0,0,self.rep,self.fold)
                        
                        
                    if isinstance(self.MLS,RandomForestRegressor):
                        points=35 
                        self.grid_bayesian=BayesianOptForestParam(self.MLS,(0.2,0.9999),(0.00390625,0.25),self.X_train,self.y_train,points,mean_squared_error,-1,self.cv_split,0,0,self.rep,self.fold)
 


            self.base_fit=self.grid_bayesian[0]
            self.params_fit =self.grid_bayesian[1]
            self.X=self.grid_bayesian[2]
            self.y=self.y_train
           
            
            self.estimators_fit=[]
            contador=0
            for g in self.params_fit:
                model=clone(self.base_fit)
                #print("param",g)
                #p=ParameterGrid({'C':g})
                if isinstance(self.MLS,svm.SVR):
                    if(self.grid_bayesian[3][contador]=='rbf'):
                        model.set_params(**{'C':g,'kernel':self.grid_bayesian[3][contador],'gamma':self.grid_bayesian[4][contador]})  
                    else:
                        model.set_params(**{'C':g,'kernel':self.grid_bayesian[3][contador]})  

                if isinstance(self.MLS,linear_model.Ridge):
                   model.set_params(**{'alpha':g,'solver':self.grid_bayesian[3][contador]})                        
                if isinstance(self.MLS,RandomForestRegressor):
                   model.set_params(**{'min_samples_leaf':self.grid_bayesian[3][contador],'max_features':g})
                   
                model=model.fit(self.X_train,self.y_train)
                self.estimators_fit.append(model)
                contador=contador+1
            
               
        X_dataframe=pd.DataFrame(self.X)
        y_dataframe=pd.DataFrame(self.y)
        if self.exists==False:
            X_dataframe.to_csv(self.path+'preds_'+str(self.fold)+'.csv', index=False)
            y_dataframe=pd.DataFrame(np.asmatrix(self.y).transpose())
            y_dataframe.to_csv(self.path+'reals_'+str(self.fold)+'.csv', index=False) 
            pickle_file = open(self.path+'base_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.base_fit,pickle_file)
            pickle_file2 = open(self.path+'params_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.params_fit,pickle_file2)
            pickle_file3 = open(self.path+'estimators_'+str(self.fold)+'.data', 'wb')
            pickle.dump(self.estimators_fit,pickle_file3)
            pickle_file.close()
            pickle_file2.close()
            pickle_file3.close()
       
                
        if isinstance(self.MLS,svm.SVR):
            points=35
                       
        if isinstance(self.MLS,linear_model.Ridge):
            points=36
                            
                
        if isinstance(self.MLS,RandomForestRegressor):
            points=35    
        
           
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
        pd.DataFrame(self.X_test).to_csv(self.path+'X_test_'+str(self.fold)+'.csv', index=False)
        self.y_pred = np.empty((X.shape[0], 1))  
        
        self.y_pred_test_params = np.empty((n_fold*self.X_test.shape[0], len(self.params_fit))) 
        self.y_pred_test_params[:] = np.nan
     
        self.preds_test_grid=np.empty((1, len(self.params_fit)), dtype=object)
       
        self.preds_test=[]
        
        ids=np.array(list(range(0,self.X_test.shape[0]))).reshape(self.X_test.shape[0],1)
        self.X_test_ids=ids
      
            
        self.preds_y=np.empty((self.X_test.shape[0],1), dtype=object) 
       
        contador=0
        self.preds_params=np.empty((self.X_test.shape[0], len(self.params_fit)), dtype=object)
       
        j=0
            
        for estimator in self.estimators_fit:
           
                self.preds_params[:,j]=estimator.predict(self.X_test)
                j=j+1
        
        contador=contador+n_fold
           
        pd.DataFrame(self.preds_params).to_csv(self.path+'pred_params_'+str(self.fold)+'.csv', index=False)    

        self.preds_params=self.preds_params
  
        
        self.y_pred_results=self.Mywrapper.predict(np.asarray(self.preds_params))

        self.final_time_predict = time() - self.start_time_predict
       																																
        pd.DataFrame(self.y_pred_results).to_csv(self.path+'y_pred_'+str(self.fold)+'.csv', index=False)
        return self.y_pred_results

    def score(self,y_real,y_predict):
        pd.DataFrame(y_predict).to_csv(self.path+'y_pred_'+str(self.fold)+'.csv', index=False)
        pd.DataFrame(y_real).to_csv(self.path+'y_real_'+str(self.fold)+'.csv', index=False)
        if self.scoring=='r2_scorer':
            self.score=r2_score(y_real, y_predict)
        if self.scoring=='neg_mean_absolute_error':
            self.score=mean_absolute_error(y_real, y_predict)
        if self.scoring=='neg_mean_squared_error':
            self.score=mean_squared_error(y_real, y_predict)
        return self.score

#Pass exception        
class bayesian_automlExcep(Exception):
    pass