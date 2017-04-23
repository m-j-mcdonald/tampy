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


class DummyAgent(object):
    def __init__(self, plan):
        self.T = plan.horizon
        self.dU = 2
        self.dX = 2
        self.dO = 0
        self.dM = 0

def dummy_hyperparams = {
    'conditions': 1
}

class DummyAlgorithm(object):
    def __init__(self, hyperparams):
        self.M = hyperparams['conditions']
        self.cur = [IterationData() for _ in range(self.M)]

class NAMOPolicySolver(NAMOSolver):

    def get_policy(self, plan, n_samples=5, iterations=5, callback=None, n_resamples=5, verbose=False):
        '''
        Use the PI2 trajectory optimizer from the GPS code base to generate
        policies for the plan, using the NAMOSolver to create samples
        '''
        alg = DummyAlgorithm(dummy_hyperparams)
        agent = DummyAgent(plan)
        traj_opt = TrajOptPI2(dummy_hyperparams)
        samples = []
        sample_costs = np.ndarray((n_samples, plan.horizon))
        for i in range(n_samples):
            self.solve(plan, callback, n_resamples, a.active_ts, verbose, force_init=True)
            samples.append(self._traj_to_sample(plan))
            sample_costs[i] = self._get_traj_cost(self, plan, agent)
        alg.cur[0].sample_list = sample_list(samples)
        alg.cur[0].traj_distr = init_pd(dummy_hyperparams)
        alg.cur[0].traj_distr = traj_opt.update(0, alg, sample_costs)
        policy = alg.cur[0].traj_distr
        
        for _ in range(1, iterations):
            for i in range(n_samples):
                new_sample = self._init_sample(condition, feature_fn=feature_fn)
                noise = generate_noise(agent.T, agent.dU, dummy_hyperparams)
                U = np.zeros([self.T, self.dU])
                for t in range(self.T):
                    X_t = new_sample.get_X(t=t)
                    obs_t = new_sample.get_obs(t=t)
                    cur_U = policy.act(X_t, obs_t, t, noise[t, :])
                    U[t, :] = cur_U

            self.solve(plan, callback, n_resamples, a.active_ts, verbose)
            policy = alg.cur[0].traj_distr


    def _traj_to_sample(self, plan, agent):
        sample = Sample(agent)
        for t in range(plan.horizon):
            sample.set(END_EFFECTOR_POINTS, plan.params[pr2].pose[:, t], t)
        return sample

    def _sample_to_traj(self, sample, plan):
        traj = sample.get(END_EFFECTOR_POINTS)

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

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features
        new_sample = self._init_sample(condition, feature_fn=feature_fn)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            if verbose:
                self._world[condition].plot(mj_X)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition, feature_fn=feature_fn)
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample
