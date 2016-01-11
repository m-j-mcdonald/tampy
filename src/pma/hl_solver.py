from IPython import embed as shell
import subprocess
import os

class HLSolver:
    """
    HLSolver deals with interfacing to the chosen task planner, e.g. methods for
    translating to PDDL if using FF/FD.
    """
    def translate(self, concr_prob, config):
        """
        Translates concrete (instantiated) problem to representation required for task planner.
        Also has access to the configuration file in case it's necessary. Initial state should be based
        on concr_prob initial state, NOT initial state from config (which may be outdated).
        E.g. for an FFsolver this would return a PDDL domain (only generated once) and problem file.
        """
        raise NotImplementedError("Override this.")

    def solve(self, abs_prob, concr_prob):
        """
        Solves the problem and returns a skeleton Plan object, which is instantiated in LLSearchNode's init.
        abs_prob is what was returned by self.translate().
        An FFSolver would only need to use abs_prob here, but in
        general a task planner may make use of the geometry, so we
        pass in the concrete problem as well.
        """
        raise NotImplementedError("Override this.")

class FFSolver(HLSolver):
    """
    For an FFSolver, the abstract problem returned by translate() and used by solve() is a tuple of
    (domain PDDL string, problem PDDL string).
    """
    FF_EXEC = "../task_planners/FF-v2.3/ff"
    FILE_PREFIX = "temp_"

    def translate(self, concr_prob, config):
        return (self._construct_domain_str(config), self._construct_problem_str(concr_prob))

    def solve(self, abs_prob, concr_prob):
        plan = self._run_planner(abs_prob)

    def _construct_domain_str(self, config):
        dom_str = "; AUTOGENERATED. DO NOT EDIT.\n\n(define (domain robotics)\n(:requirements :strips :equality :typing)\n(:types "
        for t in config["Types"].split(","):
            dom_str += t.strip() + " "
        dom_str += ")\n\n(:predicates\n"
        for p_defn in config["Predicates"].split(";"):
            p_name, p_params = map(str.strip, p_defn.split(",", 1))
            p_params = [s.strip() for s in p_params.split(",")]
            dom_str += "(%s "%p_name
            for i, param in enumerate(p_params):
                dom_str += "?var%d - %s "%(i, param)
            dom_str += ")\n"
        dom_str += ")\n\n"
        for key in config.keys():
            if key.startswith("Action"):
                count, inds = 0, [0]
                for i, token in enumerate(config[key]):
                    if token == "(":
                        count += 1
                    if token == ")":
                        count -= 1
                        if count == 0:
                            inds.append(i+1)
                assert len(inds) == 4
                params = config[key][inds[0]:inds[1]].strip()
                pre = config[key][inds[1]:inds[2]].strip()
                eff = config[key][inds[2]:inds[3]].strip()
                dom_str += "(:action %s\n:parameters %s\n:precondition %s\n:effect %s\n)\n\n"%(key.split()[1], params, pre, eff)
        dom_str += ")"
        return dom_str

    def _construct_problem_str(self, concr_prob):
        prob_str = "; AUTOGENERATED. DO NOT EDIT.\n\n(define (problem ff_prob) (:domain robotics)\n(:objects\n"
        for param in concr_prob.init_state.params:
            prob_str += "%s - %s\n"%(param.name, param.get_type())
        prob_str += ")\n\n(:init\n"
        for pred in concr_prob.init_state.preds:
            prob_str += "(%s "%pred.get_type()
            for param in pred.params:
                prob_str += "%s "%param.name
            prob_str += ")\n"
        prob_str += ")\n\n(:goal\n(and "
        for pred in concr_prob.goal_preds:
            prob_str += "(%s "%pred.get_type()
            for param in pred.params:
                prob_str += "%s "%param.name
            prob_str += ") "
        prob_str += ")\n)\n)"
        return prob_str

    def _run_planner(self, abs_prob):
        dom, prob = abs_prob
        with open("%sdom.pddl"%FFSolver.FILE_PREFIX, "w") as f:
            f.write(dom)
        with open("%sprob.pddl"%FFSolver.FILE_PREFIX, "w") as f:
            f.write(prob)
        with open(os.devnull, "w") as devnull:
            subprocess.call([FFSolver.FF_EXEC, "-o", "%sdom.pddl"%FFSolver.FILE_PREFIX, "-f", "%sprob.pddl"%FFSolver.FILE_PREFIX], stdout=devnull)
        with open("%sprob.pddl.soln"%FFSolver.FILE_PREFIX, "r") as f:
            plan = [s.strip() for s in f.readlines()[1:]]
        subprocess.call(["rm", "%sdom.pddl"%FFSolver.FILE_PREFIX, "%sprob.pddl"%FFSolver.FILE_PREFIX, "%sprob.pddl.soln"%FFSolver.FILE_PREFIX])
        return plan
