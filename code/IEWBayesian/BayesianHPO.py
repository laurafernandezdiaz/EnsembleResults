#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bayes_opt import BayesianOptimization
from .SysWrapAlphaSolver import SysWrapAlphaSolver
from .SysWrapCGammaKernel import SysWrapCGammaKernel
from .SysWrapForest import SysWrapForest
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np


# Documentation's example
#def black_box_function(x,y):
#    return -x ** 2 - (y - 1) ** 2 + 1
#
#pbounds = {'x': (2, 4), 'y': (-3, 3)}
#
#optimizer = BayesianOptimization(
#    f=black_box_function,
#    pbounds=pbounds,
#    random_state=1,
#)
#
#optimizer.maximize(init_points=2,n_iter=3)

def BayesianOptAlphaSolverParam(base,Alphabounds,Solverbounds,X,Y,points,loss,loss_sign,cv,cv_state,bayes_state,init_prop=0.3,rep=1,fold=1):
    """
    Bayesian optimization software from: 
        https://github.com/fmfn/BayesianOptimization
    
    Search for the best loss (higher loss*loss_sign) in C=10**Cbounds
    The loss is calculated using Cross Validation with cv folds for the (X,Y) dataset
    
    Params:
        base        : scikit Machine Learing System with C param
        Cbounds     : 2 element tuple (min_expC, max_expC) where 
                       min_C=10**min_expC, and max_C=10**max_expC
        X,Y         : dataset target to the search
        points      : number of points in the space search (min_c,max_C)
                       30% as initial points and 70% as iterations for bayesian optimization
        loss        : loss function
        loss_sign   : loss*loss_sign must be better as higher
        cv          : number of folds in inner Cross Validation (CV)
        cv_state    : integer for random_state for inner CV
        bayes_state : integer for random_state for bayesian optimization
        init_prop   : proportion of initial random points.
                       If =1 only random search is made
        
    Returns: [base,Cs,Evals]
        base : trained base with the best C
        Cs   : Array of points C values tested
        Evals: Array of points Evaluations. Each evaluation store the predicction 
                for each example in the inner CV
    """
    
    SysWrap=SysWrapAlphaSolver(base,X,Y,loss,loss_sign,cv=cv,cv_state=cv_state)
    init_points=int(points*init_prop)
    n_iter=points-init_points
    optimizer = BayesianOptimization(f=SysWrap,pbounds={'alpha':Alphabounds,'solver':Solverbounds},random_state=bayes_state)
    optimizer.maximize(init_points=init_points,n_iter=n_iter-1)
    base.alpha=float(optimizer.max['params']['alpha'])
    solver=int(optimizer.max['params']['solver'])
    
    solver=int(solver)
    if(solver==1):
        base.solver='svd'
    if(solver==2):
        base.solver='cholesky'
    if(solver==3):
        base.solver='lsqr'
    if(solver==4):
        base.solver='sparse_cg'
    if(solver==5):
       base.solver='sag'
    if(solver==6):
        base.solver='saga'
    base.fit(X,Y)
    return [base,SysWrap.alphas,pd.DataFrame(SysWrap.Evals).T,SysWrap.solvers]

def BayesianOptForestParam(base,max_featuresbounds,min_samples_leafbounds,X,Y,points,loss,loss_sign,cv,cv_state,bayes_state,init_prop=0.3,rep=1,fold=1):
    """
    Bayesian optimization software from: 
        https://github.com/fmfn/BayesianOptimization
    
    Search for the best loss (higher loss*loss_sign) in C=10**Cbounds
    The loss is calculated using Cross Validation with cv folds for the (X,Y) dataset
    
    Params:
        base        : scikit Machine Learing System with C param
        Cbounds     : 2 element tuple (min_expC, max_expC) where 
                       min_C=10**min_expC, and max_C=10**max_expC
        X,Y         : dataset target to the search
        points      : number of points in the space search (min_c,max_C)
                       30% as initial points and 70% as iterations for bayesian optimization
        loss        : loss function
        loss_sign   : loss*loss_sign must be better as higher
        cv          : number of folds in inner Cross Validation (CV)
        cv_state    : integer for random_state for inner CV
        bayes_state : integer for random_state for bayesian optimization
        init_prop   : proportion of initial random points.
                       If =1 only random search is made
        
    Returns: [base,Cs,Evals]
        base : trained base with the best C
        Cs   : Array of points C values tested
        Evals: Array of points Evaluations. Each evaluation store the predicction 
                for each example in the inner CV
    """
    
    SysWrap=SysWrapForest(base,X,Y,loss,loss_sign,cv=cv,cv_state=cv_state)
    init_points=int(points*init_prop)
    n_iter=points-init_points
    optimizer = BayesianOptimization(f=SysWrap,pbounds={'max_features':max_featuresbounds,'min_samples_leaf':min_samples_leafbounds},random_state=bayes_state)
    optimizer.maximize(init_points=init_points,n_iter=n_iter-1)
    base.max_features=float(optimizer.max['params']['max_features'])
    base.min_samples_leaf=float(optimizer.max['params']['min_samples_leaf'])

    
    base.fit(X,Y)
    return [base,SysWrap.max_featuress,pd.DataFrame(SysWrap.Evals).T,SysWrap.min_samples_leafs]


def BayesianOptCGammaKernelParam(base,Cbounds,Kernelbounds,Gammabounds,X,Y,points,loss,loss_sign,cv,cv_state,bayes_state,init_prop=0.3,rep=1,fold=1):
    """
    Bayesian optimization software from: 
        https://github.com/fmfn/BayesianOptimization
    
    Search for the best loss (higher loss*loss_sign) in C=10**Cbounds
    The loss is calculated using Cross Validation with cv folds for the (X,Y) dataset
    
    Params:
        base        : scikit Machine Learing System with C param
        Cbounds     : 2 element tuple (min_expC, max_expC) where 
                       min_C=10**min_expC, and max_C=10**max_expC
        X,Y         : dataset target to the search
        points      : number of points in the space search (min_c,max_C)
                       30% as initial points and 70% as iterations for bayesian optimization
        loss        : loss function
        loss_sign   : loss*loss_sign must be better as higher
        cv          : number of folds in inner Cross Validation (CV)
        cv_state    : integer for random_state for inner CV
        bayes_state : integer for random_state for bayesian optimization
        init_prop   : proportion of initial random points.
                       If =1 only random search is made
        
    Returns: [base,Cs,Evals]
        base : trained base with the best C
        Cs   : Array of points C values tested
        Evals: Array of points Evaluations. Each evaluation store the predicction 
                for each example in the inner CV
    """
    
    SysWrap=SysWrapCGammaKernel(base,X,Y,loss,loss_sign,cv=cv,cv_state=cv_state)
    init_points=int(points*init_prop)
    n_iter=points-init_points
    optimizer = BayesianOptimization(f=SysWrap,pbounds={'C':Cbounds,'gamma':Gammabounds,'kernel':Kernelbounds},random_state=bayes_state)
    optimizer.maximize(init_points=init_points,n_iter=n_iter-1)
    base.C=10**float(optimizer.max['params']['C'])
    base.gamma=float(optimizer.max['params']['gamma'])
    kernel=int(optimizer.max['params']['kernel'])

    
    kernel=int(kernel)
    if(kernel==1):
        base.kernel='rbf'
    if(kernel==2):
        base.kernel='linear'

    if(base.kernel=='rbf'):
         base.gamma=float(optimizer.max['params']['gamma'])
         
    base.fit(X,Y)
    return [base,SysWrap.Cs,pd.DataFrame(SysWrap.Evals).T,SysWrap.kernels,SysWrap.gammas]

