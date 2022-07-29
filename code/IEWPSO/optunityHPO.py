#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import optunity #https://optunity.readthedocs.io/en/latest/index.html

from sklearn.svm import SVR
from .SysWrapAlphaSolver import SysWrapAlphaSolver
from .SysWrapCGammaKernel import SysWrapCGammaKernel
from .SysWrapForest import SysWrapForest
import pandas as pd


def PSOOptForestParam(base,max_featuresbounds,min_samples_leafbounds,X,Y,num_evals,loss,loss_sign,cv,cv_state,rep=1,fold=1):
    SysWrap=SysWrapForest(base,X,Y,loss,loss_sign,cv=cv,cv_state=cv_state)
    
    if loss_sign==-1:
        pars, details, _ = optunity.minimize(SysWrap, num_evals=num_evals, max_features=max_featuresbounds, min_samples_leaf=min_samples_leafbounds, solver_name='particle swarm')
    else:
        pars, details, _ = optunity.maximize(SysWrap, num_evals=num_evals, max_features=max_featuresbounds, min_samples_leaf=min_samples_leafbounds, solver_name='particle swarm')
    print("pars",pars)
    print("details",details)
    base.max_features=float(pars['max_features'])
    
    base.min_samples_leaf=float(pars['min_samples_leaf'])
         
    base.fit(X,Y)
    
    return [base,SysWrap.max_featuress,pd.DataFrame(SysWrap.Evals).T,SysWrap.min_samples_leafs]

def PSOOptAlphaSolverParam(base,Alphabounds,Solverbounds,X,Y,num_evals,loss,loss_sign,cv,cv_state,rep=1,fold=1):
   
    
    SysWrap=SysWrapAlphaSolver(base,X,Y,loss,loss_sign,cv=cv,cv_state=cv_state)
    
    if loss_sign==-1:
        pars, details, _ = optunity.minimize(SysWrap, num_evals=num_evals, alpha=Alphabounds, solver=Solverbounds, solver_name='particle swarm')
    else:
        pars, details, _ = optunity.maximize(SysWrap, num_evals=num_evals, alpha=Alphabounds, solver=Solverbounds, solver_name='particle swarm')
    print("pars",pars)
    print("details",details)
    base.alpha=pars['alpha']
    solver=pars['solver']
    
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
   

def PSOOptCGammaKernelParam(base,Cbounds,Kernelbounds,Gammabounds,X,Y,num_evals,loss,loss_sign,cv,cv_state,rep=1,fold=1):
    

    SysWrap=SysWrapCGammaKernel(base,X,Y,loss,loss_sign,cv=cv,cv_state=cv_state)
    
    if loss_sign==-1:        
        pars, details, _ = optunity.minimize(SysWrap, num_evals=num_evals, C=Cbounds, gamma=Gammabounds, kernel=Kernelbounds, solver_name='particle swarm')
    else:
        pars, details, _ = optunity.maximize(SysWrap, num_evals=num_evals, C=Cbounds, gamma=Gammabounds, kernel=Kernelbounds, solver_name='particle swarm')
    print("pars",pars)
    print("details",details)
    base.C=10**float(pars['C'])
    
    base.gamma=float(pars['gamma'])
    kernel=int(pars['kernel'])

    if(kernel==1):
        base.kernel='rbf'
    if(kernel==2):
        base.kernel='linear'

    if(base.kernel=='rbf'):
        base.gamma=float(pars['gamma'])
         
         
    base.fit(X,Y)
    
    return [base,SysWrap.Cs,pd.DataFrame(SysWrap.Evals).T,SysWrap.kernels,SysWrap.gammas]