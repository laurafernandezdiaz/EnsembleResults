# -*- coding: utf-8 -*-

from scipy.linalg import lstsq;
import numpy as np
class OLS():
  
        
    def __init__(self):
        """
        
        """
     
    def fit(self,X,y):
        """   
        Fit method (lstsq)
        Params:
            Dataset parameters: 
                - X            : X train data
                - y            : y train data
        """
        
        self.RMf=lstsq(X,y,cond=1e-12); # Regression
        self.Wf=self.RMf[0];        # Linear Model
        self.coef_=self.Wf
        self.used=str(np.count_nonzero(self.coef_!=0))+"/"+str(len(self.coef_))
        return [self.RMf,self.Wf]
        

    def predict(self,X): 
       [R,C]=X.shape;
       # Create the test Matrix
       # Eval Wf over XMf
       Rf=len(X);  
       preds=[]
      
       for r in range(Rf):
           Pred=0;
           preds.append(np.dot(X[r],self.Wf))
       self.Preds=preds
       return self.Preds

        