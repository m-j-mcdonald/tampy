import numpy as np
from gps.agent.agent_utils import generate_noise
from gps.algorithm.algorithm_utils import IterationData
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION
from gps.sample.sample import Sample, SampleList
from pma.ll_solver import NAMOSolver

# TODO: Don't just use END_EFFECTOR_POINTS to measuer trajectories

class DummyAgent(object):
    def __init__(self, plan):
        self.T = plan.horizon
        self.dU = 2
        self.dX = 2
        self.dO = 0
        self.dM = 0

class DummyAlgorithm(object):
    def __init__(self, hyperparams):
        self.M = hyperparams['conditions']
        self.cur = [IterationData() for _ in range(self.M)]

class NAMOPolicySolver(NAMOSolver):

    def get_policy(self, plan, n_samples=5, iterations=5, callback=None, n_resamples=0, verbose=False):
        '''
        Use the PI2 trajectory optimizer from the GPS code base to generate
        policies for the plan, using the NAMOSolver to create samples
        '''
        dummy_hyperparams = {
            'conditions': 1,
            'x0': plan.params['pr2'].pose[:, 0]
        }
        alg = DummyAlgorithm(dummy_hyperparams)
        agent = DummyAgent(plan)
        traj_opt = TrajOptPI2(dummy_hyperparams)
        alg.cur[0].traj_distr = init_pd(dummy_hyperparams)
        samples = []
        sample_costs = np.ndarray((n_samples, plan.horizon))
        for i in range(n_samples):
            self.solve(plan, callback, n_resamples, None, verbose, force_init=True)
            samples.append(self._traj_to_sample(plan))
            sample_costs[i] = self._get_traj_cost(self, plan, agent)
        alg.cur[0].sample_list = sample_list(samples)
        alg.cur[0].traj_distr = traj_opt.update(0, alg, sample_costs)
        policy = alg.cur[0].traj_distr
        for _ in range(1, iterations):
            samples = []
            sample_costs = np.ndarray((n_samples, plan.horizon))
            for i in range(n_samples):
                plan.params['pr2'].pose = self._sample_to_traj(self._sample_policy(policy), agent)
                self.solve(plan, callback, n_resamples, None)
                samples.append(self._traj_to_sample(plan))
                sample_costs[i] = self._get_traj_cost(self, plan, agent)
            alg.cur[0].sample_list = sample_list(samples)
            alg.cur[0].traj_distr = traj_opt.update(0, alg, sample_costs)
            policy = alg.cur[0].traj_distr
        return policy, plan.satisfied()

    def _traj_to_sample(self, plan, agent):
        sample = Sample(agent)
        for t in range(plan.horizon - 1):
            sample.set(END_EFFECTOR_POINTS, plan.params[pr2].pose[:, t], t)
            sample.set(ACTION, plan.params[pr2].pose[:, t+1] - plan.params[pr2].pose[:, t], t)
        return sample

    def _sample_to_traj(self, sample):
        traj = sample.get(END_EFFECTOR_POINTS)
        return traj

    def _get_traj_cost(self, plan):
        '''
        Get a vector of the costs for each timestep in the plan
        '''
        cur_action = plan.actions.filter(lambda a: a.active_ts[0] <= 0)[0]
        costs = np.ndarray((plan.horizon))
        for ts in range(plan.horizon):
            timestep_cost = 0
            if ts > cur_action.active_ts[1]:
                cur_action = plan.actions.filter(lambda a: a.active_ts[0] == ts)[0]
            for p in cur_actions.preds:
                if pred_d['hl_info'] == 'hl_state': continue
                pred = p['pred']
                negated = p['negated']
                start, end = pred_d['active_timesteps']
                if ts < start or ts > end: continue
                if not pred.test(ts, negated=negated):
                    param_vector = pred.get_param_vector(ts)
                    penalty_expr = pred.expr.convexify(param_vector)
                    timestep_cost += penalty_expr.eval(param_vector, tol=pred.tol, negated=negated)
            costs[ts] = timestep_cost
        return costs

    def _sample_policy(self, policy, agent, plan):
        new_sample = Sample(agent)
        new_sample.set(END_EFFECTOR_POINTS, plan.params['pr2'].pose[0, :])
        noise = generate_noise(agent.T, agent.dU, dummy_hyperparams)
        U = np.zeros([agent.T, agent.dU])
        for t in range(self.T - 1):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            cur_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = cur_U
            new_sample.set(END_EFFECTOR_POINTS, X_t + cur_U, t+1)
        return new_sample
