# -*- coding: utf-8 -*-


from .optunityHPO import PSOOptForestParam,PSOOptAlphaSolverParam,PSOOptCGammaKernelParam
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
class optunity_automl:


    def __init__(self,MLS,GSparams,scoring,cv_split=3,rep=1,fold=1,exp=1):
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
        #Creating the fields
        self.coef_grid_=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv.       
        self.best_estimator_grid_=[] #Attribute to save the best estimator of the normal gridsearchcv
        self.coef_grid_automl=[] #Attribute to save the coef of the estimator fitted using noraml gridSearchcv
        self.estimators=[]
        
        if self.scoring=='r2_scorer':
            self.score_function=r2_score
        if self.scoring=='neg_mean_absolute_error':
            self.score_function=mean_absolute_error
        if self.scoring=='neg_mean_squared_error':
            self.score_function=mean_squared_error
         
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
            raise optunity_automlExcep(BaseErrMsg+'DataFrame attributes are not a numpy.ndarray');
        self.start_time_fit = time()
        self.X_train=X
        self.y_train=y
        
        self.exists=False
        os.chdir(os.path.dirname(__file__))
        self.path=str(os.getcwd())+'\\'+str(self.exp)+'\\'
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
                        self.grid_PSO=PSOOptCGammaKernelParam(self.MLS,(-3,3),(0.55,2),(0.001,1),self.X_train,self.y_train,points,mean_absolute_error,+1,self.cv_split,0,self.rep,self.fold)
                        
                    if isinstance(self.MLS,linear_model.Ridge):
                        points=36
                        self.grid_PSO=PSOOptAlphaSolverParam(self.MLS,(0.01, 1),(1,6),self.X_train,self.y_train,points,mean_absolute_error,+1,self.cv_split,0,self.rep,self.fold)
                                            
                    if isinstance(self.MLS,RandomForestRegressor):
                        points=35 
                        self.grid_PSO=PSOOptForestParam(self.MLS,(0.2,0.9999),(0.00390625,0.25),self.X_train,self.y_train,points,mean_absolute_error,+1,self.cv_split,0,self.rep,self.fold)
                                        
            if self.scoring=='neg_mean_squared_error':
                    if isinstance(self.MLS,svm.SVR):
                        points=35
                        self.grid_PSO=PSOOptCGammaKernelParam(self.MLS,(-3,3),(0.55,2),(0.001,1),self.X_train,self.y_train,points,mean_squared_error,+1,self.cv_split,0,self.rep,self.fold)
                        
                    if isinstance(self.MLS,linear_model.Ridge):
                        points=36
                        self.grid_PSO=PSOOptAlphaSolverParam(self.MLS,(0.01, 1),(1,6),self.X_train,self.y_train,points,mean_squared_error,+1,self.cv_split,0,self.rep,self.fold)
                                            
                    if isinstance(self.MLS,RandomForestRegressor):
                        points=35 
                        self.grid_PSO=PSOOptForestParam(self.MLS,(0.2,0.9999),(0.00390625,0.25),self.X_train,self.y_train,points,mean_squared_error,+1,self.cv_split,0,self.rep,self.fold)
                


            self.base_fit=self.grid_PSO[0]
            self.params_fit =self.grid_PSO[1]
            self.X=self.grid_PSO[2]
            self.y=self.y_train
           
            self.estimators_fit=[]
            contador=0
            for g in self.params_fit:
                model=clone(self.base_fit)
                if isinstance(self.MLS,svm.SVR):
                    if(self.grid_PSO[3][contador]=='rbf'):
                        model.set_params(**{'C':g,'kernel':self.grid_PSO[3][contador],'gamma':self.grid_PSO[4][contador]})  
                    else:
                        model.set_params(**{'C':g,'kernel':self.grid_PSO[3][contador]})  

                if isinstance(self.MLS,linear_model.Ridge):
                   model.set_params(**{'alpha':g,'solver':self.grid_PSO[3][contador]})                        
                if isinstance(self.MLS,RandomForestRegressor):
                   model.set_params(**{'min_samples_leaf':self.grid_PSO[3][contador],'max_features':g})
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
            
        self.inversos=np.empty((len(self.params_fit)), dtype=object)   
        
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
     
        contador=0
        self.preds_params=np.empty((self.X_test.shape[0], len(self.params_fit)), dtype=object)
    
        j=0
        for estimator in self.estimators_fit:
            self.preds_params[:,j]=estimator.predict(self.X_test)
            j=j+1
                 
        self.preds_params_pond=np.empty((self.X_test.shape[0], len(self.params_fit)), dtype=object)

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
class optunity_automlExcep(Exception):
    pass