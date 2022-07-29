# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:58:41 2019

@author: Y0633602
"""

"""
The :mod:`sklearn.model_selection._validation` module includes classes and
functions to validate the model.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause


import warnings
import numbers
import time
from traceback import format_exception_only
import scipy.sparse as sp
from sklearn.utils import safe_indexing
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils._joblib import logger
from sklearn.exceptions import FitFailedWarning

import sqlite3
from sqlite3 import Error

def sql_connection(exp,rep):

    try:


        con = sqlite3.connect('mydatabase_combineC_'+str(exp)+'_'+str(rep)+'.db')

        return con

    except Error:

        print(Error)
        
def sql_table_data(con):

    cursorObj = con.cursor()

    cursorObj.execute("CREATE TABLE if not exists Data (example INTEGER,rep INTEGER,params TEXT,xs TEXT,y_pred REAL,score REAL,y_real REAL)")

    con.commit()
    
    
def sql_insert_data(con, data):

    cursorObj = con.cursor()
    cursorObj.execute('INSERT INTO Data(example,rep,params,xs,y_pred,score,y_real) VALUES(?, ?, ?, ?, ?,?,?)', data)



def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters,cv_split,rep, exp,fit_params,return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=True, return_estimator=True,
                   error_score='raise-deprecating'):
    
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' | 'raise-deprecating' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If set to 'raise-deprecating', a FutureWarning is printed before the
        error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        Default is 'raise-deprecating' but from version 0.22 it will change
        to np.nan.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    return_estimator : boolean, optional, default: False
        Whether to return the fitted estimator.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.

    estimator : estimator object
        The fitted estimator
    """

    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = {k: _index_param_value(X, v, train)
                  for k, v in fit_params.items()}

    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
 

    X_train_part=X_train[:,1:]
    #X_test_part=X_test[:,1:]
    is_multimetric = not callable(scorer)
   
    n_scorers = len(scorer.keys()) if is_multimetric else 1
    try:
        if y_train is None:
            estimator.fit(X_train_part, **fit_params)

        else:
            estimator.fit(X_train_part, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif error_score == 'raise-deprecating':
            warnings.warn("From version 0.22, errors during fit will result "
                          "in a cross validation score of NaN by default. Use "
                          "error_score='raise' if you want an exception "
                          "raised or error_score=np.nan to adopt the "
                          "behavior from version 0.22.",
                          FutureWarning)
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                
                test_scores = dict(zip(scorer.keys(),
                                   [error_score, ] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(),
                                        [error_score, ] * n_scorers))
            else:
                test_scores = error_score
            
            
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exception_only(type(e), e)[0]),
                          FitFailedWarning)
           
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        #sm=_score_metric(X_test, y_test,estimator)

        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric,parameters,cv_split,exp,rep)
       # test_scores = _score(sm, estimator, X_test, y_test, scorer, is_multimetric)
        
       
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            
           # sm=_score_metric(X_train, y_train,estimator)
            #train_scores = _score(sm,estimator, X_train, y_train, scorer,
             #                     is_multimetric)
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric,parameters,cv_split,exp,rep)
    if verbose > 2:
        if is_multimetric:
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += "(train=%.3f," % train_scores[scorer_name]
                    msg += " test=%.3f)" % test_scores[scorer_name]
                else:
                    msg += "%.3f" % test_scores[scorer_name]
        else:
            msg += ", score="
            msg += ("%.3f" % test_scores if not return_train_score else
                    "(train=%.3f, test=%.3f)" % (train_scores, test_scores))

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:

        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
        

    return ret

def _score(estimator, X_test, y_test, scorer, is_multimetric=False,parameters=None,cv_split=2,exp=1,rep=1):
    """Compute the score(s) of an estimator on a given test set.

    Will return a single float if is_multimetric is False and a dict of floats,
    if is_multimetric is True
    """
    if is_multimetric:

        
        return _multimetric_score(estimator, X_test, y_test, scorer,parameters,cv_split,exp,rep)
    else:
        print("no")
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            print("no")
                   
            score = scorer(X_test,y_test,estimator)
        
        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%r)"
                             % (str(score), type(score), scorer))


    return score


def _multimetric_score(estimator, X_test, y_test, scorers,parameters,cv_split,exp,rep):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    con=sql_connection(exp,rep)
    sql_table_data(con)

    for name, scorer in scorers.items():
        X_test_part=X_test[:,1:]

        
        if y_test is None:
            
            score = scorer(estimator, X_test_part)
            
        else:
                
                
                
                result = scorer(estimator,X_test_part,y_test)
                #Begin Laura 
                #score=result
                score=result[0]
                y_pred=result[1]
                
          
                example=0

                #f = open(str(exp)+'/eval_grid_'+str(cv_split)+'.csv','a')
      
                for i in X_test:
                    
                    xs=""
                    pos=0
                    for j in i:

                        j=str(j).replace("[", "").replace("]", "")
                        
                        if pos==0:
                            xs=str(j)
                        else:
                            xs=str(xs)+';'+str(j)
                        
                        pos=pos+1
                  
                    position=0
                    for e in parameters.keys():
                          
                           for p in estimator.get_params().keys():
                               if e==p:
                                  
                                   if position==0:
                                       estimator_params=str(estimator.get_params()[e])
                                   else:
                                       estimator_params=estimator_params+';'+str(estimator.get_params()[e])
                                    
                           position=position+1
                   

                    data=(example,rep,str(estimator_params),str(xs).replace("[", "").replace("]", ""),float(y_pred[example]),score,float(y_test[example]))             

                    sql_insert_data(con, data)
                    #f.write(str(example)+';'+str(rep)+';'+str(estimator_params)+';'+str(xs).replace("[", "").replace("]", "")+';'+str(y_pred[example])+';'+str(score)+';'+str(y_test[example])+'\n') #Saved data in eval_grid.csv.
                    example=example+1
                con.commit()    
              #End Laura 
            
        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score
        
        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))

    return scores



def _index_param_value(X, v, indices):
    """Private helper function for parameter value indexing."""
    if not _is_arraylike(v) or _num_samples(v) != _num_samples(X):
        # pass through: skip indexing
        return v
    if sp.issparse(v):
        v = v.tocsr()
    return safe_indexing(v, indices)


