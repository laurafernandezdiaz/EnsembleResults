#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold

class SysWrapCGammaKernel:
    def __init__(self,base,X,Y,loss,loss_sign=+1,cv=3,cv_state=0):
        """
        SVMBase  : SVM Base with C Param
        X,Y      : Data set for error estimating
        loss     : loss function to optimize
        loss_sign: multiply the loss function in order to be better as higher
        cv       : number of folds for Cross Validation stimation
        cv_state : Integer that represents a Random state for Cross Validation
        """
        #Params
        self.base=base
        self.X=np.array(X)
        self.Y=np.array(Y)
        self.loss=loss
        self.loss_sign=loss_sign
        
        # Folds
        self.FoldTr=[]
        self.FoldTe=[]
        kf = KFold(n_splits=cv,shuffle=True,random_state=cv_state)
        ind=kf.split(X)
        for k,(tr,te) in enumerate(ind):
            self.FoldTr.append(tr)
            self.FoldTe.append(te)
            
        # Model
        self.Cs=[]
        self.gammas=[]
        self.kernels=[]
        self.Evals=[]
        
    def __call__(self,C,gamma,kernel):
        """
        Function that estimates the error of SVMBase with C, gamma and kernel params using CrossValidation(cv,cv_state)
        """
        # Setting parameters
        self.base.C=10**C
        kernel=int(kernel)
        if(kernel==1):
            self.base.kernel='rbf'
        if(kernel==2):
            self.base.kernel='linear'

        if(self.base.kernel=='rbf'):
           self.base.gamma=gamma
        
        # Loss estimating using Cross Validation
        AllEval=[None]*len(self.X)
        AllLoss=[]
        for k in range(len(self.FoldTr)):
            self.base.fit(self.X[self.FoldTr[k]],self.Y[self.FoldTr[k]])
            ev=self.base.predict(self.X[self.FoldTe[k]])
            for i in range(len(self.FoldTe[k])):
                AllEval[self.FoldTe[k][i]]=ev[i]
            l=self.loss(self.Y[self.FoldTe[k]],ev)
            AllLoss.append(l*self.loss_sign)
  
        # Store
        self.Cs.append(self.base.C)
        self.gammas.append(self.base.gamma)
        self.kernels.append(self.base.kernel)
        self.Evals.append(AllEval)
        
        l=sum(AllLoss)/len(AllLoss)
        
        return l