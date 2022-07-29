# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import Counter
class Caruana:
    
    def __init__(self,randomObject,score_function,s=None,rep=True,bags=20,r=0.5,ver=0):
        """
        Parameters:
            randomObject   : random.Random object
            score_function : function that returns the score from 2 vectors (better as lower)
            s              : number attributes in each aggregation 
                             if None then same number al attributes
            rep            : if True with replacement else without replacement
            bags           : number of bags. Each bag creates a aggregation of s attributes
            r              : proportion of attributes (without replacement) in each bag
            ver            : verbosity {0,1,2}
            
        """
        self.randomObject=randomObject
        self.score_function=score_function
        self.s=s
        self.rep=rep
        self.bags=bags
        self.r=r
        self.ver=ver
       

    def getCandidates(self,rO,n,r):
        po=int(n*r)
        ne=n-po
        V=[1]*po+[0]*ne
        rO.shuffle(V)
        index=[]
        for i in range(n):
            if V[i]==1:
                index.append(i)
        return index
    
    def meanCols(self,X,atts):
        Ms=[]
        for e in range(len(X)):
            M=0
            for a in atts:
                M=M+X[e][a]
            M=M/len(atts)
            Ms.append(M)
        return Ms
    
    def fit(self,X,Y):
        """
        Returns a weight vector for each X attribute using caruana strategy.
        Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004, July). Ensemble selection from libraries of models. In Proceedings of the twenty-first international conference on Machine learning (p. 18).
        
        
        Parameters:
            X,Y            : dataset
                       
        Returns:
            W : weight vector for each attribute
        """

        p=len(X[0])
        if self.s==None:
            self.s=p
        
        allSelModels=[]
        
        for b in range(0,self.bags):
            candidates=self.getCandidates(self.randomObject,p,self.r)
            if self.ver>=1:
                print('Bag {}, candidates:'.format(b),candidates)
            global_best_model=[]
            global_best_score=math.inf
            
            if self.rep==False:
                self.s=min(self.s,len(candidates))
            
            selected_models=[]
            for j in range(self.s):
                best=math.inf
                Selected_index=-1
                for c in candidates: 
                    
                    if self.rep or c not in selected_models:
                    
                        try_model=selected_models.copy()
                        try_model.append(c)
                        score_ensemble=self.score_function(self.meanCols(X,try_model),Y)
                        if self.ver>=2:
                            print('try={} score={}'.format(try_model,score_ensemble))
                        
                        if score_ensemble<best:
                            
                            best=score_ensemble
                            Selected_index=c
                                     
                if self.ver>=1:
                    print('Best candidate: s={}. When {} selected {}'.format(j,selected_models,Selected_index))
                selected_models.append(Selected_index) 
                
                if global_best_score>best:
                    if self.ver>=1:
                        print('Global Best, score={}'.format(global_best_score))
                    global_best_score=best
                    global_best_model=selected_models.copy()
            
                 
            allSelModels=allSelModels+global_best_model
            if self.ver>=1:
                print('Global best for this bag:',end='')
                print(global_best_model)
                print('Current Model',allSelModels)
            
            
        self.nIters=len(allSelModels)
        self.nAtts=len(Counter(allSelModels).keys())

        if(self.nIters!=0):
            self.reemplazo=(self.nIters-self.nAtts)/self.nIters
        else:
            self.reemplazo=0
                
        print("nIters:" + str(self.nIters))
        print("nAtts:" + str(self.nAtts))
        print("reemplazo:" + str(self.reemplazo)) 
        
        weights=[0]*p
        for selP in allSelModels:
            weights[selP]=weights[selP]+1
        for p in range(p):
            weights[p]=weights[p]/len(allSelModels)
         
        self.coef_=weights
        self.used=str(np.count_nonzero(self.coef_))+"/"+str(len(self.coef_))
        self.chosen_features=np.count_nonzero(self.coef_!=0)																																  

    def predict(self,X):
        P=[0]*X.shape[0]
        for i in range(X.shape[0]):
            P[i]=sum(X[i,:]*self.coef_)
        return P
