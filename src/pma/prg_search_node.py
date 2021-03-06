from IPython import embed as shell

class SearchNode(object):
    """
    There are two types of nodes in the plan refinement graph (PRGraph). High-level search
    nodes store abstract and concrete representations of the problem (concrete is an instance
    of the Problem class), and they interface to planning with the chosen HLSolver. Low-level search
    nodes store the Plan object for refinement, and they interface to planning with the chosen LLSolver.
    """
    def __init__(self, *args):
        raise NotImplementedError("Must instantiate either HL or LL search node.")

    def heuristic(self):
        """
        The node with the highest heuristic value is selected at each iteration of p_mod_abs.
        """
        return 0

    def is_hl_node(self):
        return False
    
    def is_ll_node(self):
        return False

    def plan(self):
        raise NotImplementedError("Override this.")

class HLSearchNode(SearchNode):
    def __init__(self, abs_prob, domain, concr_prob, prefix=None):
        self.abs_prob = abs_prob
        self.domain = domain
        self.concr_prob = concr_prob
        self.prefix = prefix if prefix else []

    def is_hl_node(self):
        return True

    def plan(self, solver):
        plan_obj = solver.solve(self.abs_prob, self.domain, self.concr_prob)
        if self.prefix:
            return self.prefix + plan_obj
        else:
            return plan_obj

    def heuristic(self):
        return -1

class LLSearchNode(SearchNode):
    def __init__(self, plan):
        self.curr_plan = plan

    def get_problem(self, i, failed_pred):
        """
        Returns a representation of the search problem which starts from the end state of step i and goes to the same goal.
        """
        raise NotImplementedError

    def solved(self):
        raise NotImplementedError

    def is_ll_node(self):
        return True

    def plan(self, solver):
        solver.solve(self.curr_plan)

    def get_failed_pred(self):
        return self.curr_plan.get_failed_pred()
