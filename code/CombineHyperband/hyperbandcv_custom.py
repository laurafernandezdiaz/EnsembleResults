
"""
The :mod:`sklearn.model_selection._search` includes utilities to fine-tune the
parameters of an estimator.
"""
from __future__ import print_function
from __future__ import division

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict
from functools import partial, reduce
from itertools import product
import operator
import time
import warnings

import numpy as np
from scipy.stats import rankdata

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection._split import check_cv
#Begin Laura
from .validation_custom import _fit_and_score
#End Laura
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._validation import Parallel, delayed
from sklearn.externals import six
from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version
from sklearn.utils.fixes import _Mapping as Mapping, _Sequence as Sequence
from sklearn.utils.fixes import _Iterable as Iterable
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.deprecation import DeprecationDict
#Begin Laura
from .scorer_custom import _check_multimetric_scoring
#End Laura

import copy

import numpy as np
from scipy.stats import rankdata

from sklearn.utils import check_random_state


__all__ = ['HyperbandSearchCV']


class ParameterGrid(object):
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    See also
    --------
    :class:`GridSearchCV`:
        Uses :class:`ParameterGrid` to perform a full parallelized parameter
        search.
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            The iteration index

        Returns
        -------
        params : dict of string to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


class ParameterSampler(object):
    """Generator on parameters sampled from given distributions.

    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Note that before SciPy 0.16, the ``scipy.stats.distributions`` do not
    accept a custom RNG instance and always use the singleton RNG from
    ``numpy.random``. Hence setting ``random_state`` will not guarantee a
    deterministic iteration whenever ``scipy.stats`` distributions are used to
    define the parameter search space. Deterministic behavior is however
    guaranteed from SciPy 0.16 onwards.

    Read more in the :ref:`User Guide <search>`.

    Parameters
    ----------
    param_distributions : dict
        Dictionary where the keys are parameters and values
        are distributions from which a parameter is to be sampled.
        Distributions either have to provide a ``rvs`` function
        to sample from them, or can be given as a list of values,
        where a uniform distribution is assumed.

    n_iter : integer
        Number of parameter settings that are produced.

    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    params : dict of string to any
        **Yields** dictionaries mapping each estimator parameter to
        as sampled value.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterSampler
    >>> from scipy.stats.distributions import expon
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> param_grid = {'a':[1, 2], 'b': expon()}
    >>> param_list = list(ParameterSampler(param_grid, n_iter=4))
    >>> rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
    ...                 for d in param_list]
    >>> rounded_list == [{'b': 0.89856, 'a': 1},
    ...                  {'b': 0.923223, 'a': 1},
    ...                  {'b': 1.878964, 'a': 2},
    ...                  {'b': 1.038159, 'a': 2}]
    True
    """
    def __init__(self, param_distributions, n_iter, random_state=None):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def __iter__(self):
        # check if all distributions are given as lists
        # in this case we want to sample without replacement
        all_lists = np.all([not hasattr(v, "rvs")
                            for v in self.param_distributions.values()])
        rnd = check_random_state(self.random_state)

        if all_lists:
            # look up sampled parameter settings in parameter grid
            param_grid = ParameterGrid(self.param_distributions)
            grid_size = len(param_grid)
            n_iter = self.n_iter

            if grid_size < n_iter:
                warnings.warn(
                    'The total space of parameters %d is smaller '
                    'than n_iter=%d. Running %d iterations. For exhaustive '
                    'searches, use GridSearchCV.'
                    % (grid_size, self.n_iter, grid_size), UserWarning)
                n_iter = grid_size
            for i in sample_without_replacement(grid_size, n_iter,
                                                random_state=rnd):
                yield param_grid[i]

        else:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(self.param_distributions.items())
            for _ in six.moves.range(self.n_iter):
                params = dict()
                for k, v in items:
                    if hasattr(v, "rvs"):
                        if sp_version < (0, 16):
                            params[k] = v.rvs()
                        else:
                            params[k] = v.rvs(random_state=rnd)
                    else:
                        params[k] = v[rnd.randint(len(v))]
                yield params

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter


def fit_grid_point(X, y, estimator, parameters, train, test, scorer,
                   verbose, error_score='raise-deprecating', **fit_params):
    
    """Run fit on one set of parameters.

    Parameters
    ----------
    X : array-like, sparse matrix or list
        Input data.

    y : array-like or None
        Targets for input data.

    estimator : estimator object
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    parameters : dict
        Parameters to be set on estimator for this grid point.

    train : ndarray, dtype int or bool
        Boolean mask or indices for training set.

    test : ndarray, dtype int or bool
        Boolean mask or indices for test set.

    scorer : callable or None
        The scorer callable object / function must have its signature as
        ``scorer(estimator, X, y)``.

        If ``None`` the estimator's default scorer is used.

    verbose : int
        Verbosity level.

    **fit_params : kwargs
        Additional parameter passed to the fit function of the estimator.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.

    Returns
    -------
    score : float
         Score of this parameter setting on given training / test split.

    parameters : dict
        The parameters that have been evaluated.

    n_samples_test : int
        Number of test samples in this split.
    """
    # NOTE we are not using the return value as the scorer by itself should be
    # validated before. We use check_scoring only to reject multimetric scorer

    scores, n_samples_test = _fit_and_score(estimator, X, y,
                                            scorer, train,
                                            test, verbose, parameters,
                                            fit_params=fit_params,
                                            return_n_test_samples=True,
                                            error_score=error_score)

    return scores, parameters, n_samples_test


def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, six.string_types) or
                    not isinstance(v, (np.ndarray, Sequence))):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))


# XXX Remove in 0.20
class _CVScoreTuple (namedtuple('_CVScoreTuple',
                                ('parameters',
                                 'mean_validation_score',
                                 'cv_validation_scores'))):
    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __repr__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __repr__(self):
        """Simple custom repr to summarize the main info"""
        return "mean: {0:.5f}, std: {1:.5f}, params: {2}".format(
            self.mean_validation_score,
            np.std(self.cv_validation_scores),
            self.parameters)


class BaseSearchCV(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):
    """Abstract base class for hyper parameter search with cross-validation.
    """

    @abstractmethod
    def __init__(self, estimator, scoring=None,
                 fit_params=None, n_jobs=None, iid='warn',
                 refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise-deprecating', return_train_score=True,cv_split=2,exp=1,rep=1):

        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.fit_params = fit_params
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.rep=rep
        self.exp=exp
        self.cv_split=cv_split
      

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        score = self.scorer_[self.refit] if self.multimetric_ else self.scorer_
        return score(self.best_estimator_, X, y)

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_params_`` '
                                 'attribute'
                                 % (type(self).__name__, method_name))
        else:
            check_is_fitted(self, 'best_estimator_')

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        -----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    def _run_search(self, evaluate_candidates):
        """Repeatedly calls `evaluate_candidates` to conduct a search.

        This method, implemented in sub-classes, makes it possible to
        customize the the scheduling of evaluations: GridSearchCV and
        RandomizedSearchCV schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearchCV abstract base class.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts a list of candidates, where each candidate is
            a dict of parameter settings. It returns a dict of all results so
            far, formatted like ``cv_results_``.

        Examples
        --------

        ::

            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
        raise NotImplementedError("_run_search not implemented.")

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        print("fitBase")
        X_part=X[:,1:]
     
        
        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
         # Standard regression scores

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)
        
        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X_part, y, groups = indexable(X_part, y, groups)
        n_splits = cv.get_n_splits(X_part, y, groups)
        
        
        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=True,                                   
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results_container = [{}]
        with parallel:
            all_candidate_params = []
            all_out = []
            
            def evaluate_candidates(candidate_params):
               
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))
               
                out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                       X, y,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       cv_split=self.cv_split,
                                                       exp=self.exp,
                                                       rep=self.rep,
                                                       **fit_and_score_kwargs)
                               for parameters, (train, test)
                               in product(candidate_params,
                                          cv.split(X_part, y, groups)))
                                           
                
              
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                
                # XXX: When we drop Python 2 support, we can use nonlocal
                # instead of results_container
                results_container[0] = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
              #  print(results_container[0]["params"])
                return results_container[0]
            
            self._run_search(evaluate_candidates)
           
        results = results_container[0]
        
        
        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = results["params"][self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X_part, y, **fit_params)
            else:
                self.best_estimator_.fit(X_part, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits
        self.fit_times=results['mean_fit_time']
        self.score_times=results['mean_score_time']
        self.estimators=results['estimator']
        
        return self

    def _format_results(self, candidate_params, scorers, n_splits, out):
        n_candidates = len(candidate_params)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
             score_time,estimator,_) = zip(*out)
        else:
            (test_score_dicts, test_sample_counts, fit_time,
             score_time,estimator,_) = zip(*out)
        
        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        # TODO: replace by a dict in 0.21
        results = (DeprecationDict() if self.return_train_score == 'warn'
                   else {})

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)
        iid = self.iid
        if self.iid == 'warn':
            warn = False
            for scorer_name in scorers.keys():
                scores = test_scores[scorer_name].reshape(n_candidates,
                                                          n_splits)
                means_weighted = np.average(scores, axis=1,
                                            weights=test_sample_counts)
                means_unweighted = np.average(scores, axis=1)
                if not np.allclose(means_weighted, means_unweighted,
                                   rtol=1e-4, atol=1e-4):
                    warn = True
                    break

            if warn:
                warnings.warn("The default of the `iid` parameter will change "
                              "from True to False in version 0.22 and will be"
                              " removed in 0.24. This will change numeric"
                              " results when test-set sizes are unequal.",
                              DeprecationWarning)
            iid = True

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if iid else None)
            if self.return_train_score:
                prev_keys = set(results.keys())
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)
                if self.return_train_score == 'warn':
                    for key in set(results.keys()) - prev_keys:
                        message = (
                            'You are accessing a training score ({!r}), '
                            'which will not be available by default '
                            'any more in 0.21. If you need training scores, '
                            'please set return_train_score=True').format(key)
                        # warn on key access
                        results.add_warning(key, message, FutureWarning)

        results.update( {'estimator' : estimator} )

        return results

class HyperbandSearchCV(BaseSearchCV):
    """Hyperband search on hyper parameters.

    HyperbandSearchCV implements a ``fit`` and a ``score`` method.
    It also implements ``predict``, ``predict_proba``, ``decision_function``,
    ``transform`` and ``inverse_transform`` if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings using the hyperband
    algorithm [1]_ .

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the scikit-learn `User Guide
    <http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-search>`_.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    resource_param : str, default='n_estimators'
        The name of the cost parameter for the estimator ``estimator``
        to be fitted. Typically, this is the number of decision trees
        ``n_estimators`` in an ensemble or the number of iterations
        for estimators trained with stochastic gradient descent.

    eta : float, default=3
        The inverse of the proportion of configurations that are discarded
        in each round of hyperband.

    min_iter : int, default=1
        The minimum amount of resource that should be allocated to the cost
        parameter ``resource_param`` for a single configuration of the
        hyperparameters.

    max_iter : int, default=81
        The maximum amount of resource that can be allocated to the cost
        parameter ``resource_param`` for a single configuration of the
        hyperparameters.

    skip_last : int, default=0
        The number of last rounds to skip. For example, this can be used
        to skip the last round of hyperband, which is standard randomized
        search. It can also be used to inspect intermediate results,
        although warm-starting HyperbandSearchCV is not supported.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`sklearn.model_selection.StratifiedKFold`
        is used. In all other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer `User Guide <http://scikit-learn.org/stable/modules/cross_validation.html>`_
        for the various cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``HyperbandSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, optional, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |        0.8      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |        0.7      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    References
    ----------

    .. [1] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A.,
           2017. Hyperband: A novel bandit-based approach to hyperparameter
           optimization. The Journal of Machine Learning Research, 18(1),
           pp.6765-6816.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`sklearn.model_selection.GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`sklearn.model_selection.ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    """
    def __init__(self, estimator, param_distributions,
                 resource_param='n_estimators', eta=3, min_iter=1,
                 max_iter=81, skip_last=0, scoring=None, n_jobs=1,
                 iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=0,
                 error_score='raise', return_train_score=False,cv_split=0,exp=1,rep=1):
        self.param_distributions = param_distributions
        self.resource_param = resource_param
        self.eta = eta
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.skip_last = skip_last
        self.random_state = random_state

        super(HyperbandSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=None,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score,cv_split=cv_split,exp=exp,rep=rep)

        
    def _run_search(self, evaluate_candidates):
        self._validate_input()

        s_max = int(np.floor(np.log(self.max_iter / self.min_iter) / np.log(self.eta)))
        B = (s_max + 1) * self.max_iter

        refit_metric = self.refit if self.multimetric_ else 'score'
        random_state = check_random_state(self.random_state)

        if self.skip_last > s_max:
            raise ValueError('skip_last is higher than the total number of rounds')

        for round_index, s in enumerate(reversed(range(s_max + 1))):
            n = int(np.ceil(int(B / self.max_iter / (s + 1)) * np.power(self.eta, s)))

            # initial number of iterations per config
            r = self.max_iter / np.power(self.eta, s)
            configurations = list(ParameterSampler(param_distributions=self.param_distributions,
                                                   n_iter=n,
                                                   random_state=random_state))

            if self.verbose > 0:
                print('Starting bracket {0} (out of {1}) of hyperband'
                      .format(round_index + 1, s_max + 1))

            for i in range((s + 1) - self.skip_last):

                n_configs = np.floor(n / np.power(self.eta, i))  # n_i
                n_iterations = int(r * np.power(self.eta, i))  # r_i
                n_to_keep = int(np.floor(n_configs / self.eta))

                if self.verbose > 0:
                    msg = ('Starting successive halving iteration {0} out of'
                           ' {1}. Fitting {2} configurations, with'
                           ' resource_param {3} set to {4}')

                    if n_to_keep > 0:
                        msg += ', and keeping the best {5} configurations.'

                    msg = msg.format(i + 1, s + 1, len(configurations),
                                     self.resource_param, n_iterations,
                                     n_to_keep)
                    print(msg)

                # Set the cost parameter for every configuration
                parameters = copy.deepcopy(configurations)
                for configuration in parameters:
                    configuration[self.resource_param] = n_iterations

                results = evaluate_candidates(parameters)

                if n_to_keep > 0:
                    top_configurations = [x for _, x in sorted(zip(results['rank_test_%s' % refit_metric],
                                                                   results['params']),
                                                               key=lambda x: x[0])]

                    configurations = top_configurations[:n_to_keep]

            if self.skip_last > 0:
                print('Skipping the last {0} successive halving iterations'
                      .format(self.skip_last))

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        
        super().fit(X, y, groups, **fit_params)
        
        s_max = int(np.floor(np.log(self.max_iter / self.min_iter) / np.log(self.eta)))
        B = (s_max + 1) * self.max_iter

        brackets = []
        for round_index, s in enumerate(reversed(range(s_max + 1))):
            n = int(np.ceil(int(B / self.max_iter / (s + 1)) * np.power(self.eta, s)))
            n_configs = int(sum([np.floor(n / np.power(self.eta, i))
                                 for i in range((s + 1) - self.skip_last)]))
            bracket = (round_index + 1) * np.ones(n_configs)
            brackets.append(bracket)

        self.cv_results_['hyperband_bracket'] = np.hstack(brackets)

        return self

    def _validate_input(self):
        if not isinstance(self.min_iter, int) or self.min_iter <= 0:
            raise ValueError('min_iter should be a positive integer, got %s' %
                             self.min_iter)

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError('max_iter should be a positive integer, got %s' %
                             self.max_iter)

        if self.max_iter < self.min_iter:
            raise ValueError('max_iter should be bigger than min_iter, got'
                             'max_iter=%d and min_iter=%d' % (self.max_iter,
                                                              self.min_iter))

        if not isinstance(self.skip_last, int) or self.skip_last < 0:
            raise ValueError('skip_last should be an integer, got %s' %
                             self.skip_last)

        if not isinstance(self.eta, int) or not self.eta > 1:
            raise ValueError('eta should be a positive integer, got %s' %
                             self.eta)

        if self.resource_param not in self.estimator.get_params().keys():
            raise ValueError('resource_param is set to %s, but base_estimator %s '
                             'does not have a parameter with that name' %
                             (self.resource_param,
                              self.estimator.__class__.__name__))
