#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@version 0.1

@author: quevedo
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import math
from .Criteriums import Criteriums


class FSCriterium:
    """
    Forward Search by gradient
    """
    
    def __init__(self,expError=2,eps=0,fit_intercept=True,max_iters=None,stop=0, min_FEV=0.95):
        """
        Params:
            eps       : The algorithm ends when the sum of the residuals will be less or equal to eps
                         or all the attributes are used
            expError  : the minimized error is: abs(Pred-Real)**expError 
            max_iters : If None dont use the iterations as stop criterium (only eps)
            min_FEV   : Fraction of Explained Variance
            stop: 
                    0=> FEV
                    4=> AIC
                    5=> BIC
        """
        
        # Params
        self.eps=eps
        self.expError=expError
        self.fit_intercept=fit_intercept 
        self.max_iters=max_iters
        self.min_FEV=min_FEV
        self.stop=stop
        
        # Model
        self.std=[]
        self.coef_=[]
        self.intercept_=None
        
    def calculateStd(self,X_Original):
        X=X_Original.copy()
        Xt=np.array(X).transpose()
        for A in Xt:
            self.std.append(np.std(A))
            
    def _filter(self,XT,NoUsed,a):
        fX=[]
#        pdb.set_trace()
#        print('Filter:',end='')
        for i in range(len(NoUsed)):
            if (not NoUsed[i] or i==a):
                fX.append(XT[i])
#                print(' {}'.format(i),end='')
#        print('')
        fX=np.array(fX).transpose()
        return fX.tolist()
    
    def _copyFilter(self,NoUsed,MinW):
        iMin=0
        W=[0]*len(NoUsed)
        for i in range(len(NoUsed)):
            if not NoUsed[i]:
                W[i]=MinW[iMin]
                iMin=iMin+1
        return W
    
    def _SStot(self,Y):
        m=sum(Y)/len(Y)
        Var=0
        for v in Y:
            Var=Var+(v-m)**2
        return Var
    
    def fit(self,X_Original,Y_Original):
        """
        Calculates a forward Stepwise regression 

        """  
        X_Original=list(X_Original)
        Y_Original=list(Y_Original)
        Y=Y_Original.copy()
        X=X_Original.copy()
        print('E={}'.format(len(X)))
        XT=np.array(X).transpose().tolist()
        self.std=[]
        for A in XT:
            self.std.append(np.std(A))
        
#        NA=len(XT)        
#        # Adapt to regression
#        for i in range(NA):
#            for j in range(len(XT[0])):
#                XT[i][j]=[XT[i][j]]
                
        if type(Y[0])==type([0]):
            for i in range(len(Y)):
                Y[i]=Y[i][0]
                
        A=len(XT)
        if self.max_iters==None:
            max_iters=A
        else:
            max_iters=self.max_iters
            
        SStot=self._SStot(Y)
        FEV=0
        W=[0]*A
        b=0
        NoUsed=[1]*A
        res=self.eps+1
        AICc=math.inf
        AIC=math.inf
        BIC=math.inf
        HQIC=math.inf
        GMDL=math.inf
        
        if self.stop==0:
            condition=sum(NoUsed)>(A-max_iters) and res>self.eps and FEV<=self.min_FEV
        elif self.stop>=3:
            condition=sum(NoUsed)>(A-max_iters) and res>self.eps
        
        while condition:
            prevaicc=AICc
            prevaic=AIC
            prevbic=BIC
            prevhqic=HQIC
            prevgmdl=GMDL
            
            # Search for the best attribute
            MinRes=None
            MinW=0
            MinB=0
            MinA=0
            for a in range(A):
                if NoUsed[a]>0:
                     [CRes,CW,Cb]=self._regression(self._filter(XT,NoUsed,a),Y)
                     if MinRes==None or MinRes>CRes:
                         MinRes=CRes
                         MinW=CW
                         MinB=Cb
                         MinA=a
                         FEV=1-CRes/SStot
#                         print('Local Min: a={} CRes={} FEV={:6.4f} W={}'.format(MinA,MinRes,FEV,MinW[0]))
            
            NoUsed[MinA]=0
            W=self._copyFilter(NoUsed,MinW)
            b=MinB
            #print('Global Min: a={} CRes={} FEV={:6.4f} W={}'.format(MinA,MinRes,FEV,MinW[0]))
            print('stop={}'.format(self.stop))
            
            if self.stop>=3:
                prevaicc=AICc
                prevaic=AIC
                prevbic=BIC
                prevhqic=HQIC
                prevgmdl=GMDL
                if type(Y_Original[0])==type([]):
                    SumYSq=(sum(map(lambda x:x[0]**2,Y_Original)))
                else:
                    SumYSq=(sum(map(lambda x:x**2,Y_Original)))
                    
                k=sum(map(lambda x:0 if x==0 else 1,W))+1 # Number of no-zero coefs
                [AIC,AICc,BIC,HQIC,GMDL]=Criteriums(res,SumYSq,A,k)
                
                print('AIC={} ,AICc={} ,BIC={} ,HQIC={} ,GMDL={} '.format(AIC,AICc,BIC,HQIC,GMDL))
                
                if self.stop==3:
                    print("\AICc: "+str(AICc),end='')
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
                       
                self.coef_=W
                self.intercept_=b 
        print('FS W=',end='')
        for w in self.coef_ :
            print(' {:+.5g}'.format(w),end='')
        print(' b={:+.5g}'.format(b))
                
       
        self.used=str(np.count_nonzero(self.coef_))+"/"+str(len(self.coef_))
        #print(self.coef_)
        print("intercept_ " + str(self.intercept_))

        
    def _regression(self,X,Y):
        Reg=LinearRegression()
        Reg.fit_intercept=self.fit_intercept
#        pdb.set_trace()
        Y= [item for sublist in Y for item in sublist]

        Reg.fit(X,Y)
        P=Reg.predict(X)
        res=sum(map(lambda p,y:abs(p-y)**self.expError,P,Y))
        return [res,Reg.coef_,Reg.intercept_]
    

    
    def predict(self,X):
        X=list(X)
        P=[0]*len(X)
        for i in range(len(X)):
            P[i]=sum(map(lambda x,w:x*w,X[i],self.coef_))+self.intercept_
        return P

