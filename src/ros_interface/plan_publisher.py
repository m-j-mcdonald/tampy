import numpy as np

import rospy
from std_msgs.msg import String

from _ActionMSG import ActionMSG
from _FloatArrayMSG import FloatArrayMSG
from _GeomMSG import GeomMSG
from _ParameterMSG import ParameterMSG
from _PlanMSG import PlanMSG
from _PredicateMSG import PredicateMSG

class PlanPublisher(object):
	def publish_plan(self, plan):
		pub = rospy.Publisher('Plan', PlanMSG, queue_size=10)
		pub2 = rospy.Publisher('test', String, queue_size=10)
		rospy.init_node('planner', anonymous=True)
		msg = self.create_plan_msg(plan)
		pub2.publish("Once upon a midnight dreary")
		pub.publish(msg)

	def create_floatarray_msg(self, array):
		float_array_msg = FloatArrayMSG()
		float_array_msg.data = array.tolist()
		return float_array_msg

	def create_geom_msg(self, geom):
		geom_msg = GeomMSG()
		geom_msg.type_name = str(type(geom)).split("'")[1].split(".")[-1]
		geom_msg.attrs = str(geom.__dict__)
		return geom_msg

	def create_parameter_msg(self, param):
		param_msg = ParameterMSG()
		param_msg.type_name = param.get_type()
		param_msg.is_symbol = param.is_symbol()
		if hasattr(param, 'lArmPose') and type(param.lArmPose) is np.ndarray:
			for joint in param.lArmPose:
				param_msg.lArmPose.append(self.create_floatarray_msg(joint))
		if hasattr(param, 'rArmPose') and type(param.rArmPose) is np.ndarray:
			for joint in param.rArmPose:
				param_msg.lArmPose.append(self.create_floatarray_msg(joint))
		if hasattr(param, 'lGripper') and type(param.lGripper) is np.ndarray:
			for grip in param.lGripper:
				param_msg.lArmPose.append(self.create_floatarray_msg(grip))
		if hasattr(param, 'rGripper') and type(param.rGripper) is np.ndarray:
			for grip in param.rGripper:
				param_msg.rGripper.append(self.create_floatarray_msg(grip))
		if hasattr(param, 'pose') and type(param.pose) is np.ndarray:
			for pose in param.pose:
				param_msg.pose.append(self.create_floatarray_msg(pose))
		if hasattr(param, 'value') and type(param.value) is np.ndarray:
			for value in param.value:
				param_msg.pose.append(self.create_floatarray_msg(value))
		if hasattr(param, 'rotation') and type(param.rotation) is np.ndarray:
			for rotation in param.rotation:
				param_msg.pose.append(self.create_floatarray_msg(rotation))
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

	def create_action_msg(self, action):
		action_msg = ActionMSG()
		action_msg.name = action.name
		action_msg.active_timesteps = action.active_timesteps
		for param in action.params:
			action_msg.parameters.append(self.create_parameter_msg(param))
		for pred in action.preds:
			action_msg.predicates.append(self.create_predicate_msg(pred['pred']))
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
