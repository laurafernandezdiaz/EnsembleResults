#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.linear_model import LinearRegression
import math
import pdb


class BST:
    """
    Boosting Linear Regression. Step Wise but allowing to repeat attribute
    Bühlmann, P., & Hothorn, T. (2007). Boosting algorithms: Regularization, prediction and model fitting. Statistical science, 22(4), 477-505.


    
    Fields:
        eps : Tolerance criterium. The algorithm finish when the residual is less than eps
    """
    def AIC_BIC_AICc(self,data,res,coefs,Y):
        #AIC
        k=sum(map(lambda x:0 if x==0 else 1,coefs))+1 # Number of no-zero coefs
        n=len(data)

        sigma2=res/n
        if (n-k)>0:
            S=(n*sigma2)/(n-k)
        else:
            S=math.inf

        if type(Y[0])==type([]):
            F=(sum(map(lambda x:x**2,Y[0])))/(k*S)
        else:
            F=(sum(map(lambda x:x**2,Y)))/(k*S)

        if F<0:
            pdb.set_trace()
        if res>0:
            l=-n/2*(1+math.log(2*math.pi)+math.log(1/n*res))
        else:
            l=-math.inf
        aic_calculada=-2*l/n+2*k/n

        print("l:"+str(l))
        print("k:"+str(k))
        print("n:"+str(n))
        #BIC
        if res>0:
            bic_calculada=-2*l/n+k*np.log(n)/n
        else:
            
            bic_calculada=math.inf

        
        #AICc
        if (n-k-1)>0:
            aic_corrected=-2*l+2*k+2*(k*k+1)/(n-k-1)
        else:
            aic_corrected=math.inf
            
        #HQIC
        if res>0:
            hqic_calculada=-2*l/n+2*k*np.log(np.log(n))/n
        else:
            
            hqic_calculada=-math.inf
        
        #GMDL
        if res>0 and (n-k)>0:
                        
            gmdl_calculada=np.log(S)+k/n*np.log(F)
        else:
            
            gmdl_calculada=math.inf


        self.AIC.append(aic_calculada), self.BIC.append(bic_calculada), self.AICc.append(aic_corrected), self.HQIC.append(hqic_calculada),self.GMDL.append(gmdl_calculada)

        return aic_calculada,bic_calculada,aic_corrected,hqic_calculada,gmdl_calculada#self.AIC, self.BIC, self.AICc
    
    def __init__(self,expError=2,eps=1e-15,fit_intercept=True,readjustAtt=True,stop=0,maxIters=100):
        """
        Params:
           
            expError: the minimized error is: abs(Pred-Real)**expError 
            eps     : The algorithm ends when the sum of the residuals will be less or equal to eps
            fit_intercept  : Whether fit intercept or not
            readjustAtt : If True runs classical boosting where in each iteration all attributes are available.
                          If False once that an attribute is choosed it will not ever be choosed again.
            stop    : If 0: not top criterium (run until maxIters)
                      If 1: stopts where found a cofficient with higer abs value than the previous
                      If 2: stops where found a change in the sign of the second derivate of the train error
                      If 3: stops where an increment in AIC corrected where found
                      If 4: stops where an increment in AIC where found
                      If 5: stops where an increment in BIC where found
                      If 6: stops where an increment in HQIC where found
                      If 7: stops where an increment in GMDL where found
                      
                      
            maxITers: If None then the maximum iterations willl be= max(Examples,Attributes)
        """
        
        # Params
        self.eps=eps
        self.expError=expError
        self.fit_intercept=fit_intercept 
        self.readjustAtt=readjustAtt
        self.stop=stop
        self.maxIters=maxIters
        
        # Test, for zoom in only
        self.Xte=[]
        self.Yte=[]
        self.TeErr=[]
        self.HErr=None
    
        
        # Model
        self.OWErr=[]
        self.std=[]
        self.coef_=[]
        self.intercept_=None
        self.nIters=0
        self.nAtts=0
        
        self.AIC=[]
        self.BIC=[]
        self.AICc=[]
        self.HQIC=[]
        self.GMDL=[]
        
    def setTest(self,Xte,Yte):
        self.Xte=Xte
        self.Yte=Yte
        
    def calculateStd(self,X_Original):
        X=X_Original.copy()
        Xt=np.array(X).transpose()
        for A in Xt:
            self.std.append(np.std(A))
    
    def fit(self,X_Original,Y_Original):
   
        self.nIters=0
        self.nAtts=0
        E=len(X_Original)
        Y_Original=list(Y_Original.ravel())			   
        Y=Y_Original.copy()
        X=X_Original.copy()
        XT=np.array(X).transpose().tolist()
        self.std=[]
        for A in XT:
            self.std.append(np.std(A))
        
        NA=len(XT)        
        # Adapt to regression
        for i in range(NA):
            for j in range(len(XT[0])):
                XT[i][j]=[XT[i][j]]
                
        if type(Y[0])==type([0]):
            for i in range(len(Y)):
                Y[i]=Y[i][0]
                
        A=len(XT)
        maxIters=self.maxIters
        if(maxIters==None):
            maxIters=max(E,A)
        W=[0]*A
        b=0
        res=self.eps+1
        PrevW=None
        PrevW_Std=None
        it=0
        Error=0
        prevaic=math.inf
        aic=math.inf
        aicc=math.inf
        bic=math.inf
        hqic=math.inf
        gmdl=math.inf
        while  res>self.eps and it<maxIters and not (self.readjustAtt==False and it>=A):
            it=it+1
            
            # Search for the best attribute
            MinRes=None
            MinW=0
            MinB=0
            MinA=0
            MinP=0
            MinW_Std=0
            for a in range(A):
                if self.readjustAtt or W[a]==0:
                     [CRes,CW,Cb,P]=self._regression(XT[a],Y)

                     if MinRes==None or MinRes>CRes:
                         MinRes=CRes
                         MinW=CW
                         MinB=Cb
                         MinA=a
                         MinP=P
                         MinW_Std=CW*self.std[a]
            print('it:{:4d} A:{} W={:+10.5g} W*std={:+8.15g} b={:+8.5g} Res={}'.format(it,MinA,MinW,MinW_Std,MinB,MinRes))
            if MinRes!=None:
                res=MinRes
                PRes=MinP

            if self.stop==1 and PrevW!=None and abs(PrevW_Std)<=abs(MinW_Std) or\
               self.stop==2 and MinRes!=None and self._changeD2Sign(self.OWErr+[MinRes]):
                print(' STOP({})'.format(self.stop))
                if MinRes!=None:
                    print('H1={} H2={}'.format(abs(PrevW_Std)<abs(MinW_Std),self._changeD2Sign(self.OWErr+[MinRes])))

                if self.HErr==None:
                    self.HErr=Error
                break
            PrevW=MinW
            PrevW_Std=MinW_Std
            
            if self.stop>=3:
                prevaicc=aicc
                prevaic=aic
                prevbic=bic
                prevhqic=hqic
                prevgmdl=gmdl

                aic,bic,aicc,hqic,gmdl=self.AIC_BIC_AICc(XT,res,W,Y_Original)
                
                if self.stop==3:
                    print("\nAICC: "+str(aicc),end='')
                    if (prevaicc<=aicc):
                        print('  ** STOP ** -> Increment in AICC')
                        break
                    else:
                        print()
                        
                if self.stop==4:
                   print("\nAIC: "+str(aic),end='')
                   if (prevaic<=aic):
                       print('  ** STOP ** -> Increment in AIC')
                       break
                   else:
                       print()
                       
                if self.stop==5:
                   print("\nBIC: "+str(bic),end='')
                   if (prevbic<=bic):
                       print('  ** STOP ** -> Increment in BIC')
                       break
                   else:
                       print()
                       
                if self.stop==6:
                   print("\nHQIC: "+str(hqic),end='')
                   if (prevhqic<=hqic):
                       print('  ** STOP ** -> Increment in HQIC')
                       break
                   else:
                       print()
                       
                if self.stop==7:
                   print("\nGMDL: "+str(gmdl),end='')
                   if (prevgmdl<=gmdl):
                       print('  ** STOP ** -> Increment in GMDL')
                       break
                   else:
                       print()
            
            # Count iterations and attributes
            self.nIters=self.nIters+1
            if W[MinA]==0:
                self.nAtts=self.nAtts+1
            W[MinA]=W[MinA]+MinW
            
            b=b+MinB
            
            # For zoom in
            self.OWErr.append(MinRes)
            if len(self.Xte)>0:
                self.coef_=W
                self.intercept_=b
                P=self.predict(self.Xte)
                Error=(sum(map(lambda r,p:abs(r-p)**2,self.Yte,P))/len(P))[0]
                print(' Error={}'.format(Error))
                self.TeErr.append(Error)
            else:
                print('')
            
            # Take the rest as the new Y
            Reg=LinearRegression()
            Reg.fit_intercept=self.fit_intercept
            Reg.fit(XT[MinA],Y)
            P=Reg.predict(XT[MinA])
            for i in range(len(Y)):
                Y[i]=Y[i]-P[i]

        print('SWR W=',end='')
        for w in W :
            print(' {:+.5g}'.format(w),end='')
        print(' b={:+.5g}'.format(b))
                
        self.coef_=W
        self.used=str(np.count_nonzero(self.coef_))+"/"+str(len(self.coef_))
        self.chosen_features=np.count_nonzero(self.coef_!=0)																																
        self.intercept_=b       
        if(self.nIters!=0):
            self.reemplazo=(self.nIters-self.nAtts)/self.nIters
        else:
            self.reemplazo=0
        print("nIters:" + str(self.nIters))
        print("nAtts:" + str(self.nAtts))
        print("reemplazo:" + str(self.reemplazo))                                                   

     
        
    def _derive(self,Y):
        n=len(Y)
        dY=[0]*n
        dY[0]=Y[1]-Y[0]
        dY[n-1]=Y[n-1]-Y[n-2]
        for i in range(1,n-1):
            dY[i]=(Y[i+1]-Y[i-1])/2

        return dY
    
    def _changeD2Sign(self,Y):
        if len(Y)<=2:
            return False
        d2=self._derive(self._derive(Y))
        print(d2)
        for i in range(1,len(d2)):
            if d2[i]>=0 and d2[i-1]<=0 or d2[i-1]>=0 and d2[i]<=0:
                return True
        return False
        
        
    def _regression(self,X,Y):
        Reg=LinearRegression()
        Reg.fit_intercept=self.fit_intercept
        Reg.fit(X,Y)
        P=Reg.predict(X)
        res=sum(map(lambda p,y:abs(p-y)**self.expError,P,Y))/len(Y)
        return [res,Reg.coef_[0],Reg.intercept_,P]
    

    
    def predict(self,X):
        P=[0]*X.shape[0]
        for i in range(X.shape[0]):
            P[i]=sum(X[i,:]*self.coef_)+self.intercept_
        return P
