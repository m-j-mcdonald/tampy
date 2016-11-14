import argparse
import sys
import threading
import Queue

import numpy as np
import rospy

import actionlib

import baxter_dataflow
import baxter_interface

from baxter_interface import CHECK_VERSION

from core.internal_repr.action import Action
from core.internal_repr.parameter import Object, Symbol
from core.internal_repr.plan import Plan
from core.internal_repr.predicate import Predicate
from core.util_classes.baxter_predicates import *
from core.util_classes.box import Box
from core.util_classes.can import Can, BlueCan, RedCan, GreenCan
from core.util_classes.circle import Circle
from core.util_classes.common_predicates import *
from core.util_classes.robots import Baxter
from core.util_classes.robot_predicates import *
from core.util_classes.table import Table
from core.util_classes.wall import Wall

class PlanReceiver(object):
	def listen_for_plans(self):
		rospy.init_node('plan_receiver')
		rospy.Subscriber('Plan', PlanMsg, _execute_plan)
		pub = rospy.publisher('Failed Predicates', String)
		rospy.spin()

		def _execute_plan(data):
			plan = self._build_plan(data)
			failed_preds = self._check_preds(plan.preds)
			if failed_preds:
				pub.publish("Failed plan predicates: {0}", str(failed_preds))
				return
			for action in plan.actions:
				failed_action_preds = self._check_preds(action.preds)
				if failed_action_preds:
					pub.publishstr("Failed action {0}. Failed preds: {1}", action.name, failed_action_preds)
					return
				self._execute_action(action)


	def _build_plan(self, data):
		env = Environment()
		params = {}
		for param in data.params:
			new_param = _build_param(param)
			params[new_param.name] = new_param

		actions = []
		for action in data.actions:
			actions.append(_build_action(action, params, env))

		return Plan(params.values(), actions, data.horizon, env)


	def _build_action(self, data, params, env):
		params = []
		for param in data.params:
			params.append(params[param])

		preds = []
		for pred in data.preds:
			preds.append(_build_pred(pred, env))

		return Action(data.step_num, data.name, data.active_timesteps, params, preds)


	def _build_pred(self, data, env):
		pred_class = eval(data.type_name)
		params = []
		for param in data.params:
			params.append(_build_param(param))
		return pred_class(data.name, params, data.param_types, env)


	def _build_param(self, data):
		if data.is_symbol:
			new_param = Object()
		else:
			new_param = Symbol()

		new_param.type = data.type_name

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
		if hasattr(data, 'geom'):
			new_param.geom = self_build_geom(data.geom)

		return new_param


	def _build_geom(self, data):
		geom_class = eval(data.type_name)
		attrs = map(lambda attr: {attr.split(":")[0] : attr.split(":")[1]}, data.attrs.split(","))

		if issubclass(geom_class, Can):
			radius = int(attrs['radius'])
			height = int(attrs['height'])
			return geom_class(radius, height)
		elif geom_class is BaxterGeom:
			return geom_class()
		elif geom_class is Box or geom_class is Table:
			dim = int(attrs['dim'])
			return geom_class(dim)
		elif issubclass(geom_class, Circle):
			radius = int(attrs['radius'])
			return geom_class(radius)
		elif geom_class is Wall:
			wall_type = attrs['wall_type']
			return geom_class(wall_type)

		return None



	def _float_array_to_numpy(self, float_array_msg):
		new_array = []
		for row in float_array_msg:
			new_array.append(row.data)
		return np.array(new_array)


	def _check_preds(self, preds):
		failed_preds = []
		for pred in preds:
			if not pred.test():
				failed_preds.append(pred)
		return failed_preds


	def _execute_action(self, action):
		def get_joint_positions(limb, pos, i):
			return {limb + "_s0": pos[0][i], limb + "_s1": pos[1][i], limb + "_e0": pos[2][i], limb + "_e1": pos[3][i], limb + "_w0": pos[4][i], limb + "_w1": pos[5][i], limb + "_w2": pos[6][i]}

		baxter = None
		for param in action.params:
			if param.name == 'baxter':
				baxter = param

		if not baxter:
			raise Exception("Baxter not found for action: %s" % action.name)

		l_arm_pos = baxter.lArmPose
		l_gripper = baxter.lGripper[0]
		r_arm_pos = baxter.rArmPose
		r_gripper = baxter.rGripper[0]

		print("Initializing node... ")
		rospy.init_node("rsdk_joint_trajectory_client")
		print("Getting robot state... ")
		rs = baxter_interface.RobotEnable(CHECK_VERSION)
		init_state = rs.state().enabled

		def clean_shutdown():
			print("\nExiting example...")
			if not init_state:
				print("Disabling robot...")
				rs.disable()
		rospy.on_shutdown(clean_shutdown)

		print("Enabling robot... ")
		rs.enable()
		print("Running. Ctrl-c to quit")

		left = baxter_interface.limb.Limb("left")
		right = baxter_interface.limb.Limb("right")
		grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
		grip_right = baxter_interface.Gripper('right', CHECK_VERSION)

		left_queue = Queue.Queue()
		right_queue = Queue.Queue()
		rate = rospy.Rate(1000)

		if grip_left.error():
			grip_left.reset()
		if grip_right.error():
			grip_right.reset()
		if (not grip_left.calibrated() and
			grip_left.type() != 'custom'):
			grip_left.calibrate()
		if (not grip_right.calibrated() and
			grip_right.type() != 'custom'):
			grip_right.calibrate()

		left.move_to_joint_positions(get_joint_positions("left", l_arm_pos, 0))
		right.move_to_joint_positions(get_joint_positions("right", r_arm_pos, 0))


		def move_thread(limb, gripper, angle, grip, queue, timeout=15.0):
				"""
				Threaded joint movement allowing for simultaneous joint moves.
		        """
				try:
					limb.move_to_joint_positions(angle, timeout)
					gripper.command_position(grip)
					queue.put(None)
				except Exception, exception:
					queue.put(traceback.format_exc())
					queue.put(exception)

		for i in range(0, len(l_gripper)):

			left_thread = threading.Thread(
				target=move_thread,
				args=(left,
					grip_left,
					get_joint_positions("left", l_arm_pos, i),
					l_gripper[i],
					left_queue
					)
			)
			right_thread = threading.Thread(
				target=move_thread,
				args=(right,
				grip_right,
				get_joint_positions("right", r_arm_pos, i),
				r_gripper[i],
				right_queue
				)
			)

			left_thread.daemon = True
			right_thread.daemon = True
			left_thread.start()
			right_thread.start()
			baxter_dataflow.wait_for(
				lambda: not (left_thread.is_alive() or right_thread.is_alive()),
				timeout=20.0,
				timeout_msg=("Timeout while waiting for arm move threads to finish"),
				rate=10,
			)
			left_thread.join()
			right_thread.join()
			result = left_queue.get()
			if not result is None:
				raise left_queue.get()
			result = right_queue.get()
			if not result is None:
				raise right_queue.get()

def receive_plan():
	pr = PlanReceiver()
	pr.listen_for_plans()

if __name__ == "__main__":
	receive_plan()
