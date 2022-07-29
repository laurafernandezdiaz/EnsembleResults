# -*- coding: utf-8 -*-

from .gridsearchcv_custom import GridSearchCV

class GridSearchCV_coef(GridSearchCV):
      def __init__(self, estimator, X, param_grid, scoring=None, fit_params=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score="warn",cv_split=2,exp=1,rep=1):
                    
        super(self.__class__, self).__init__(
            estimator=estimator, param_grid=param_grid, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv,  verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=False,cv_split=cv_split,exp=exp,rep=rep)
            
        self.coef_=[] # Necessary attribute to save the coef of the estimator fitted.
        
        self.best_estimator_grid_=[] # Necessary attribute to save the best estimator of the gridsearchcv
        self.best_parameters_=[]
        self.fit_times=[]
        self.score_times=[]
        self.estimators=[]

      def fit(self, X, y=None, groups=None, **fit_params):
         
        # Calling to the super estimator fit
        super(self.__class__, self).fit(X,y)
        

        self.best_estimator_grid_=self.best_estimator_
        self.best_parameters_=self.best_params_
        self.fit_times=self.fit_times
        self.score_times=self.score_times
        self.estimators=self.estimators
        

       
     
            
           
       
     

            
        