from IPython import embed as shell
from core.internal_repr.predicate import Predicate
from core.util_classes.matrix import Vector2d
from core.util_classes.openrave_body import OpenRAVEBody
from errors_exceptions import PredicateException
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
import numpy as np
from openravepy import Environment
import ctrajoptpy

"""
This file implements the classes for commonly used predicates that are useful in a wide variety of
typical domains.
"""

DEFAULT_TOL=1e-4

class ExprPredicate(Predicate):

    """
    Predicates which are defined by a target value for a set expression.
    """

    def __init__(self, name, expr, attr_inds, tol, params, expected_param_types):
        """
        attr2inds is a dictionary that maps each parameter name to a
        list of (attr, active_inds) pairs. This defines the mapping
        from the primitive predicates of the params to the inputs to
        expr
        """
        super(ExprPredicate, self).__init__(name, params, expected_param_types)
        self.expr = expr
        self.attr_inds = attr_inds
        self.tol = tol

        self.x_dim = sum(len(active_inds)
                         for p_attrs in attr_inds.values()
                         for (_, active_inds) in p_attrs)
        self.x = np.zeros(self.x_dim)


    def get_expr(self, pred_dict, action_preds):
        """
        Returns an expr or None

        pred_dict is a dictionary containing
        - the Predicate object (self)
        - negated (Boolean): whether the predicated is negated
        - hl_info (string) which is "pre", "post" and "hl_state" if the
          predicate is a precondition, postcondition, or neither and part of the
          high level state respectively
        - active_timesteps (tuple of (start_time, end_time))

        action_preds is a list containing all the predicate dictionaries for
            the action get_expr is being called from.
        """
        raise NotImplementedError

    def get_param_vector(self, t):
        i = 0
        for p in self.params:
            for attr, ind_arr in self.attr_inds[p.name]:
                n_vals = len(ind_arr)

                if p.is_symbol():
                    self.x[i:i+n_vals] = getattr(p, attr)[ind_arr, 0]
                else:
                    self.x[i:i+n_vals] = getattr(p, attr)[ind_arr, t]
                i += n_vals
        return self.x

    def test(self, time):
        if not self.is_concrete():
            return False
        if time < 0:
            raise PredicateException("Out of range time for predicate '%s'."%self)
        try:
            return self.expr.eval(self.get_param_vector(time), tol=self.tol)
        except IndexError:
            ## this happens with an invalid time
            raise PredicateException("Out of range time for predicate '%s'."%self)

    def unpack(self, y):
        """
        Gradient returned in a similar form to attr_inds
        {param_name: [(attr, (g1,...gi,...gn)]}
        gi are in the same order as the attr_inds list
        """
        res = {}
        i = 0
        for p in self.params:
            res[p.name] = []
            for attr, ind_arr in self.attr_inds[p.name]:
                n_vals = len(ind_arr)
                res[p.name].append((attr, y[i:i+n_vals]))
                i += n_vals
        return res

    def _grad(self, t):
        return self.expr.grad(self.get_param_vector(t))
