import numpy as np

import rospy
from std_msgs.msg import String

from baxter_plan.msg import ActionMSG
from baxter_plan.msg import ActionPredMSG
from baxter_plan.msg import FloatArrayMSG
from baxter_plan.msg import GeomMSG
from baxter_plan.msg import ParameterMSG
from baxter_plan.msg import PlanMSG
from baxter_plan.msg import PredicateMSG

class PlanPublisher(object):
	def publish_plan(self, plan):
		pub = rospy.Publisher('Plan', PlanMSG, queue_size=10)
		pub2 = rospy.Publisher('test', ActionMSG, queue_size=10)
		rospy.init_node('planner', anonymous=True)
		msg = self.create_plan_msg(plan)
		pub2.publish(self.create_action_msg(plan.actions[0]))
		pub.publish(msg)

	def create_floatarray_msg(self, array):
		float_array_msg = FloatArrayMSG()
		float_array_msg.data = array.tolist()
		return float_array_msg

	def create_geom_msg(self, geom):
		geom_msg = GeomMSG()
		geom_msg.type_name = str(type(geom)).split("'")[1].split(".")[-1]
		attr_dict = {k:v for k,v in geom.__dict__.iteritems() if type(v) is float or type(v) is str}
		geom_msg.attrs = str(attr_dict)[1:-1]
		return geom_msg

	def create_parameter_msg(self, param):
		param_msg = ParameterMSG()
		param_msg.type_name = param.get_type()
		param_msg.is_symbol = param.is_symbol()

		if hasattr(param, 'name') and type(param.name) is str:
			param_msg.name = param.name

		if hasattr(param, 'lArmPose'):
			if type(param.lArmPose) is np.ndarray:
				for joint in param.lArmPose:
					param_msg.lArmPose.append(self.create_floatarray_msg(joint))
			else:
				param_msg.undefined_attrs.append('lArmPose')

		if hasattr(param, 'rArmPose'):
			if type(param.rArmPose) is np.ndarray:
				for joint in param.rArmPose:
					param_msg.rArmPose.append(self.create_floatarray_msg(joint))
			else:
				param_msg.undefined_attrs.append('rArmPose')

		if hasattr(param, 'lGripper'):
			if type(param.lGripper) is np.ndarray:
				for joint in param.lGripper:
					param_msg.lGripper.append(self.create_floatarray_msg(joint))
			else:
				param_msg.undefined_attrs.append('lGripper')

		if hasattr(param, 'rGripper'):
			if type(param.rGripper) is np.ndarray:
				for joint in param.rGripper:
					param_msg.rGripper.append(self.create_floatarray_msg(joint))
			else:
				param_msg.undefined_attrs.append('rGripper')

		if hasattr(param, 'pose'):
			if type(param.pose) is np.ndarray:
				for pose in param.pose:
					param_msg.pose.append(self.create_floatarray_msg(pose))
			else:
				param_msg.undefined_attrs.append('pose')

		if hasattr(param, 'value'):
			if type(param.value) is np.ndarray:
				for val in param.value:
					param_msg.value.append(self.create_floatarray_msg(val))
			else:
				param_msg.undefined_attrs.append('value')

		if hasattr(param, 'rotation'):
			if type(param.rotation) is np.ndarray:
				for rotation in param.rotation:
					param_msg.rotation.append(self.create_floatarray_msg(rotation))
			else:
				param_msg.undefined_attrs.append('rotation')

		if hasattr(param, 'geom'):
			param_msg.geom = self.create_geom_msg(param.geom)

		return param_msg

	def create_predicate_msg(self, pred):
		pred_msg = PredicateMSG()
		pred_msg.type_name = pred.get_type()
		pred_msg.name = pred.name
		for param in pred.params:
			pred_msg.params.append(self.create_parameter_msg(param))
			pred_msg.param_types.append(param.get_type())
		return pred_msg

	def create_actionpred_msg(self, actionpred):
		actionpred_msg = ActionPredMSG()
		actionpred_msg.negated = actionpred['negated']
		actionpred_msg.hl_info = actionpred['hl_info']
		actionpred_msg.pred = self.create_predicate_msg(actionpred['pred'])
		actionpred_msg.active_timesteps = actionpred['active_timesteps']
		return actionpred_msg

	def create_action_msg(self, action):
		action_msg = ActionMSG()
		action_msg.name = action.name
		action_msg.active_timesteps = action.active_timesteps
		for param in action.params:
			action_msg.parameters.append(param.name)
		for pred in action.preds:
			action_msg.predicates.append(self.create_actionpred_msg(pred))
		return action_msg

	def create_plan_msg(self, plan):
		plan_msg = PlanMSG()
		plan_msg.horizon = plan.horizon
		for parameter in plan.params.values():
			plan_msg.parameters.append(self.create_parameter_msg(parameter))
		for action in plan.actions:
			plan_msg.actions.append(self.create_action_msg(action))

		return plan_msg

if __name__ == "__main__":
	pb = PlanPublisher()
