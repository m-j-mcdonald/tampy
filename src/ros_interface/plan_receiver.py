import sys

import numpy as np
import rospy

class PlanReceiver(object):
	def listen_for_plans(self):
		rospy.init_node('plan_receiver')
		rospy.Subscriber('Plan', Plan, _execute_plan)
		rospy.spin()

		def _read_plan(data):
			plan = self._build_plan(data)

	def _build_plan(self, data):
		params = []
		for param in data.params:
			params.append(_build_param(param))

		actions = []
		for action in data.actions:
			actions.append(_build_action(action))

		return Plan(params, actions, data.horizon, None)

	def _build_action(self, data):
		params = []
		for param in data.params:
			params.append(_build_param(param))

		preds = []
		for pred in data.preds:
			# Right now no params can be rebuilt that require env
			if _pred_is_expr(pred):
				preds.append(_build_pred(pred))

		return Action(data.step_num, data.name, data.active_timesteps, params, preds)

	def _pred_is_expr(self, pred):
		pred_class = getattr(sys.modules[__name__], pred.type)
		return issubclass(pred_class, ExprPredicate)

	def _build_pred(self, data):
		pred_class = getattr(sys.modules[__name__], data.type)
		params = []
		for param in data.params:
			params.append(_build_param(param))
		return pred_class(data.name, params, data.param_types)

	def _build_param(self, data):
		if data.is_symbol:
			new_param = Object(attr_types=data.attr_types)
		else:
			new_param = Symbol(attr_types=data.attr_types)

		if hasattr(data, 'lArmPose'):
			new_param.lArmPose = self._float_array_to_numpy(data.lArmPose)
		if hasattr(data, 'rArmPose'):
			new_param.rArmPose = self._float_array_to_numpy(data.rArmPose)
		if hasattr(data, 'lGripper'):
			new_param.lGripper = self._float_array_to_numpy(data.lGripper)
		if hasattr(data, 'rGripper'):
			new_param.rGripper = self._float_array_to_numpy(data.rGripper)
		if hasattr(data, 'pose'):
			new_param.pose = self._float_array_to_numpy(data.pose)
		if hasattr(data, 'value'):
			new_param.value = self._float_array_to_numpy(data.value)
		if hasattr(data, 'rotation'):
			new_param.rotation = self._float_array_to_numpy(data.rotation)

	def _float_array_to_numpy(self, float_array_msg):
		new_array = []
		for row in float_array_msg:
			new_array.append(row.data)
		return np.array(new_array)
