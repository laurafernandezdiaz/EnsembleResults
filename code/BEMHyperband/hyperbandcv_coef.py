# -*- coding: utf-8 -*-


from .hyperbandcv_custom import HyperbandSearchCV

class Hyperbandcv_coef(HyperbandSearchCV):
      def __init__(self, estimator, param_distributions,
                 resource_param='n_estimators', eta=3, min_iter=1,
                 max_iter=81, skip_last=0, scoring=None, n_jobs=1,
                 iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=0,
                 error_score='raise', return_train_score=False,cv_split=2,exp=1,rep=1):
                    
        super(self.__class__, self).__init__(
            estimator, param_distributions,
                 resource_param=resource_param, eta=eta, min_iter=min_iter,
                 max_iter=max_iter, skip_last=skip_last, scoring=scoring, n_jobs=n_jobs,
                 iid=iid, refit=refit, cv=cv,
                 verbose=verbose, pre_dispatch=pre_dispatch, random_state=random_state,
                 error_score=error_score, return_train_score=return_train_score,cv_split=cv_split,exp=exp,rep=rep)
     
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
        

       
     
            
           
       
     

            
        