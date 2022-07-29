#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold

class SysWrapAlphaSolver:
    def __init__(self,base,X,Y,loss,loss_sign=+1,cv=3,cv_state=0):
        """
        base     : Ridge Base
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
        self.alphas=[]
        self.solvers=[]
        self.Evals=[]
        
    def __call__(self,alpha,solver):
        """
        Function that estimates the error of RidgeBase with alpha and solver param using CrossValidation(cv,cv_state)
        """
        # Setting parameters
        self.base.alpha=alpha
        solver=int(solver)
        if(solver==1):
            self.base.solver='svd'
        if(solver==2):
            self.base.solver='cholesky'
        if(solver==3):
            self.base.solver='lsqr'
        if(solver==4):
            self.base.solver='sparse_cg'
        if(solver==5):
            self.base.solver='sag'
        if(solver==6):
            self.base.solver='saga'
       
        
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
        self.alphas.append(self.base.alpha)
        self.solvers.append(self.base.solver)
        self.Evals.append(AllEval)
        
        l=sum(AllLoss)/len(AllLoss)
        
        return l