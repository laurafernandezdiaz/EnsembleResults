#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

def Criteriums(Res,SumYSq,n,k):
    """
    Calculate several criteiums for linear regression 
    Params:
        - Res : residual : sum of the squares of the difference between prediction and real values
        - SumYSq : sum of the squares of Y
        - n   : number of attributes (variables)
        - k   : number of parameters
        
    Returs:
        [AIC,AICc,BIC,HQIC,GMDL]
    """
    

    sigma2=Res/n
    if (n-k)>0:
        S=(n*sigma2)/(n-k)
    else:
        S=math.inf
    
    F=SumYSq/(k*S)
    assert F>=0

    if Res>0:
        l=-n/2*(1+math.log(2*math.pi)+math.log(1/n*Res))
    else:
        l=-math.inf
    AIC=-2*l/n+2*k/n
    
    #BIC
    if Res>0:
        BIC=-2*l/n+k*math.log(n)/n
    else:
        BIC=math.inf

    
    #AICc
    if (n-k-1)>0:
        AICc=-2*l+2*k+2*(k*k+1)/(n-k-1)
    else:
        AICc=math.inf
        
    #HQIC
    if Res>0:
        HQIC=-2*l/n+2*k*math.log(math.log(n))/n
    else:
        HQIC=-math.inf
    
    #GMDL
    if Res>0 and S>0 and F>0: 
#        print('S={} F={}'.format(S,F))
        GMDL=math.log(S)+k/n*math.log(F)
    else:
        GMDL=math.inf
        
    return [AIC,AICc,BIC,HQIC,GMDL]
    