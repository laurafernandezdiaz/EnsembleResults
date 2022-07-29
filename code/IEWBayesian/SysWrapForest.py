#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import KFold

class SysWrapForest:
    def __init__(self,base,X,Y,loss,loss_sign=+1,cv=3,cv_state=0):
        """
        base     : RandomForest base
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
        self.max_featuress=[]
        self.min_samples_leafs=[]
        self.Evals=[]
        
    def __call__(self,max_features,min_samples_leaf):
        """
        Function that estimates the error of RandomForestBase with max_features and min_samples_leaf params using CrossValidation(cv,cv_state)
        """
        # Setting parameters
        self.base.max_features=max_features
        self.base.min_samples_leaf=min_samples_leaf
       
        
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
        self.max_featuress.append(self.base.max_features)
        self.min_samples_leafs.append(self.base.min_samples_leaf)
        self.Evals.append(AllEval)
        
        l=sum(AllLoss)/len(AllLoss)
        
        return l