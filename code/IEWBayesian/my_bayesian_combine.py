# -*- coding: utf-8 -*-

from .bayesian_automl import bayesian_automl
from sklearn.model_selection import KFold
from .mean_automl_run import mean_automl_run 
import statistics
from sklearn.metrics import mean_absolute_error,mean_squared_error
class my_bayesian_combine:
    
    def __init__(self,MLS,GSparams,RG,scoring,cv_split,exp,rep,X,y):
        
        self.y_test_predict_combine=[]
        self.score_combine=[]
        self.grid_combine_models=[]
        
        self.y_test_predict_mean=[]
        self.score_mean=[]
        self.grid_mean_models=[]
        for i in range(0,rep):
            self.score_combine_fold=[]
            self.score_mean_system_fold=[]
            self.score_fold=[]
            fold=1
            kf=KFold(n_splits=3, random_state=RG.randint(0,2**31), shuffle=True)
            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                cv=KFold(cv_split,shuffle=True,random_state=RG.randint(0,2**31))
        
                #GRID combine
                self.grid_combine=bayesian_automl(MLS,GSparams,scoring,cv_split,i,fold,exp)
               
                self.grid_combine.fit(X_train,y_train)
                self.grid_combine_models.append(self.grid_combine)
                y_test_predict_combine_fold=self.grid_combine.predict(X_test)
                self.y_test_predict_combine.append(y_test_predict_combine_fold)
                score_combine_fold_rep=self.grid_combine.score(y_test, y_test_predict_combine_fold)
                self.score_combine_fold.append(score_combine_fold_rep)
                  
                #GRID mean
                self.grid_mean=mean_automl_run(MLS,GSparams,cv,scoring,i,cv_split,exp,fold)            
                self.grid_mean.fit(X_train,y_train)
                self.grid_mean_models.append(self.grid_mean)
                y_test_predict_mean_fold=self.grid_mean.predict(X_test,y_test)
                self.y_test_predict_mean.append(y_test_predict_mean_fold)
                score_mean_fold_rep=self.grid_mean._score(y_test, y_test_predict_mean_fold)
                self.score_mean_system_fold.append(score_mean_fold_rep)
                
                self.score_fold.append((score_combine_fold_rep/score_mean_fold_rep)*100)
                
                fold=fold+1
                
            print('Scores in folds:',self.score_fold)
            self.score_mean_fold=statistics.mean(self.score_fold)
            self.score_combine.append(self.score_mean_fold)
            print('Scores in each rep:',self.score_combine)
        self.score_mean=statistics.mean(self.score_combine)
        print('Score final mean:',self.score_mean)
        
#Pass exception        
class my_grid_searchExcep(Exception):
    pass