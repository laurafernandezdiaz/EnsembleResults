#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ver 0.9

@author: quevedo
"""

from sklearn.decomposition import PCA
import scipy.linalg
import numpy as np
import math
from .Criteriums import Criteriums

class PCACriterium:
    """
    Makes a PCA decomposition and then applies linear regression with intercept
    """
    def __init__(self,stop=0, n_components=1, expError=2):
        """
        n_components :number of components of PCA. 
                      Must be in [1;NAttributes] 
        """
        self.n_components=n_components
        self.stop=stop
        self.expError=expError
        self.lc=None # Lineal Coefficients

    def _addColum1(self,X):
        """
        Add a constant 1 column and the end of the matrix
        Used for calculating the intercept
        """
        X1=[]
        for r in X:
            X1.append(r+[1])
        return X1

    def fit(self,X,Y):
        Y_Original=list(Y.ravel())
        if self.stop==0:
            X=list(X)
            Y=list(Y)
            self.PCAModel=PCA(self.n_components)
            
            self.PCAModel=self.PCAModel.fit(X)
            XPCA=self.PCAModel.transform(X)
            c=scipy.linalg.lstsq(self._addColum1(XPCA),Y)
            self.lc=c[0] # Coefficients
            print('PCA({})'.format(len(self.PCAModel.components_)))
        elif self.stop>=3:
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

                PCAModel=PCA(nc)
                PCAModel=PCAModel.fit(X)
                XPCA=PCAModel.transform(X)
                c=scipy.linalg.lstsq(self._addColum1(XPCA),Y)
                lc=c[0] # Coefficients
                W=lc
                P=self._linEval(self._addColum1(XPCA),lc)

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
            
                self.lc=W
                self.PCAModel=PCAModel   
        print('PCA({})'.format(len(self.PCAModel.components_)))
    def _linEval(self,X,c):
        c=c.flatten()
        P=[0]*len(X)
        for r in range(len(X)):
            P[r]=sum(X[r]*c)
        return P
    
    def predict(self,X):
        X=list(X)
        XPCA=self.PCAModel.transform(X)
        Eval=self._linEval(self._addColum1(XPCA),self.lc)
        self.coef_=self.lc
        self.used=str(np.count_nonzero(self.coef_))+"/"+str(len(self.coef_))
        return Eval

