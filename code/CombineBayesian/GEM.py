# -*- coding: utf-8 -*-

def dot(V1,V2):
    return sum(map(lambda v1,v2:v1*v2,V1,V2))

def objective(w,X,y):
    MSE=0
    for e in range(len(X)):
        E=y[e]-dot(X[e],w)
        MSE=MSE+E**2
        #print(MSE)
    return MSE


def constraint1(x):
    cons=0
    for i in range(0,len(x)):
        cons=cons+x[i]
    return cons - 1.0        

import numpy as np
from scipy.optimize import minimize
    
class GEM():


    def __init__(self):
        """
        
        """
        
    def fit(self,X,y):
        """   
        Fit method (minimize_sum1_model)
        Params:
            Dataset parameters: 
                - X            : X train data
                - y            : y train data
        """
        X=X.tolist()
        y=y.tolist()
        num_variables=len(X[0])
        
        

        x0 = [0]*num_variables
        
        for i in range(0,num_variables):
            x0[i] = 1 / num_variables

        b = (0, 1.0)
        bnds = ((b, ) * num_variables)
        
        con1 = {'type': 'eq', 'fun': constraint1}
        cons = [con1]
        
        solution = minimize(objective, x0, args=(X,y),method='SLSQP',
                            options={'disp': True, 'maxiter': 3000, 'eps': 1e-3},
                            bounds=bnds,
                            constraints=cons)
        
        print(solution)
        print('sum(w)={}'.format(sum(solution.x)))
        self.coef_=solution.x
        self.used=str(np.count_nonzero(self.coef_!=0))+"/"+str(len(self.coef_))
        return self.coef_
        

    def predict(self,X): 
       [R,C]=X.shape;
       # Create the test Matrix
       Rf=len(X);  
       preds=[]
    
       for r in range(Rf):   
           preds.append(np.dot(X[r],self.coef_))
       self.Preds=preds
       return self.Preds

        