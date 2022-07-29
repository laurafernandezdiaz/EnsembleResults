#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: quevedo
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import numpy as np
import math
from .Criteriums import Criteriums

class PLSCriterium : 
    """
    Uses sklearn.cross_decomposition.PLSRegression
    if stop:
        0=> n_components>1
        1=> FEV self.n_components>0 and self.n_components<1
    if n_components >0 and <1 it is used as Fraction of Explained Variance (FEV)
    Returns the regression with the minimum number of components that get that FEV
      or the maximun number if it is not possible to get that value
    """
    
    def __init__(self,stop=0,n_components=1,expError=2):
        # Params
        self.n_components=n_components
        self.stop=stop
        # Model
        self.PLSModel=[]
        self.expError=expError
    def fit(self,X,Y):
        Y_Original=list(Y.ravel())
        
        if self.stop==0:
            self.PLSModel=PLSRegression(n_components=self.n_components)
            self.RMf=self.PLSModel.fit(X,Y)
        elif self.stop==1:
            Ex=len(X)
            At=len(X[0])
            max_comp=min(Ex,At)
            for nc in range(1,max_comp+1):
                self.PLSModel=PLSRegression(n_components=nc)
                self.RMf=self.PLSModel.fit(X,Y)
                OW=self.PLSModel.predict(X)
                FEV=r2_score(Y,OW)
#                print('nc={} FEV={:6.4f} n_components={:6.4f}'.format(nc,FEV,self.n_components))
                if FEV>=self.n_components:
                    break 
        elif self.stop>=3: #AIC
            Ex=len(X)
            At=len(X[0])

            AICc=math.inf
            AIC=math.inf
            BIC=math.inf
            HQIC=math.inf
            GMDL=math.inf
            
            max_comp=min(Ex,At)
            for nc in range(1,max_comp+1):
                prevaicc=AICc
                prevaic=AIC
                prevbic=BIC
                prevhqic=HQIC
                prevgmdl=GMDL
                
                PLSModel=PLSRegression(n_components=nc)
                RMf=PLSModel.fit(X,Y)
                P=PLSModel.predict(X)
                W=RMf.coef_[:nc]
                res=sum(map(lambda p,y:abs(p-y)**self.expError,P,Y))
                
#                print('nc={} FEV={:6.4f} n_components={:6.4f}'.format(nc,FEV,self.n_components))
                print('stop={}'.format(self.stop))
            
             
                if type(Y_Original[0])==type([]):
                    SumYSq=(sum(map(lambda x:x[0]**2,Y_Original)))
                else:
                    SumYSq=(sum(map(lambda x:x**2,Y_Original)))
                    
                k=sum(map(lambda x:0 if x==0 else 1,W)) + 1 # Number of no-zero coefs
                
                
                [AIC,AICc,BIC,HQIC,GMDL]=Criteriums(res,SumYSq,At,nc+1)
                
                print('AIC={} ,AICc={} ,BIC={} ,HQIC={} ,GMDL={} '.format(AIC,AICc,BIC,HQIC,GMDL))
                
                if self.stop==3:
                    print("\nAICc: "+str(AICc),end='')
                    if (prevaicc<=AICc):
                        print('  ** STOP ** -> Increment in AICc')
                        break
                    else:
                        print()
                        
                if self.stop==4:
                   print("\nAIC: "+str(AIC),end='')
                   if (prevaic<=AIC):
                       print('  ** STOP ** -> Increment in AIC')
                       break
                   else:
                       print()
                       
                if self.stop==5:
                   print("\nBIC: "+str(BIC),end='')
                   if (prevbic<=BIC):
                       print('  ** STOP ** -> Increment in BIC')
                       break
                   else:
                       print()
                       
                if self.stop==6:
                   print("\nHQIC: "+str(HQIC),end='')
                   if (prevhqic<=HQIC):
                       print('  ** STOP ** -> Increment in HQIC')
                       break
                   else:
                       print()
                       
                if self.stop==7:
                   print("\nGMDL: "+str(GMDL),end='')
                   if (prevgmdl<=GMDL):
                       print('  ** STOP ** -> Increment in GMDL')
                       break
                   else:
                       print()
                       
                self.PLSModel=PLSModel 
                self.Wf=W;
        else:
            raise Exception('PLSRegressionFEV.n_components={}, not valid value'.format(self.n_components))
        
        self.coef_=self.Wf
        self.used=str(np.count_nonzero(self.coef_))+"/"+str(len(self.coef_))
        print('PLS({})'.format(self.PLSModel.n_components))
        
    def predict(self,X):
        return self.PLSModel.predict(X)


