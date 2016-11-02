class PlanPublisher(object):
	def publish_plan(self, plan):
		pub = rospy.Publisher('Plan', Plan, queue_size=10)
		rospy.init_node('planner', anonymous=True)
		msg = generate_plan_msg(Plan)
		pub.publish(msg)

	def create_floatarray_msg(self, array):
		float_array_msg = FloatArray()
		float_array_msg.data = array[:]
		return float_array_msg

	def create_parameter_msg(self, param):
		param_msg = Parameter()
		param_msg.type = param.get_type()
		param_msg.is_symbol = param.is_symbol()
		if hasattr(param, 'lArmPose'):
			for joint in param.lArmPose:
				param_msg.lArmPose.append(create_floatarray_msg(joint))
		if hasattr(param, 'rArmPose'):
			for joint in param.rArmPose:
				param_msg.lArmPose.append(create_floatarray_msg(joint))
		if hasattr(param, 'lGripper'):
			for grip in param.lGripper:
				param_msg.lArmPose.append(create_floatarray_msg(grip))
		if hasattr(param, 'rGripper'):
			for grip in param.rGripper:
				param_msg.rGripper.append(create_floatarray_msg(grip))
		if hasattr(param, 'pose'):
			for pose in param.pose:
				param_msg.pose.append(create_floatarray_msg(pose))
		if hasattr(param, 'value'):
			for value in param.value:
				param_msg.pose.append(create_floatarray_msg(value))
		if hasattr(param, 'rotation'):
			for rotation in param.rotation:
				param_msg.pose.append(create_floatarray_msg(rotation))
		return param_msg

	def create_predicate_msg(self, pred):
		pred_msg = Predicate()
		pred_msg.type = pred.get_type()
		pred_msg.name = pred.name
		for param in pred.params:
			pred_msg.params.append(create_parameter_msg(param))
			pred_msg.param_types.append(param.get_type())
		return pred_msg

	def create_action_msg(self, action):
		action_msg = Action()
		action_msg.name = action.name
		action_msg.active_timesteps = action.active_timesteps
		for param in action.params:
			action_msg.parameters.append(create_parameter_msg(param))
		for pred in action.preds:
			action_msg.pred.append(create_predicate_msg(pred))
		return action_msg

	def create_plan_msg(self, plan):
		plan_msg = Plan()
		plan_msg.horizon = plan.horizon
		for action in plan.actions:
			plan_msg.actions.append(create_action_msg(action))
		for parameter in plan.params:
			plan_msg.parameters.append(create_parameter_msg(parameter))
		for predicate in plan.preds:
			plan_msg.predicates.append(create_predicate_msg(predicates))
