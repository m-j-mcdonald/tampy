#!/usr/bin/env python

import argparse
import sys
import threading
import Queue

import numpy as np

import rospy
from std_msgs.msg import String

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
from core.util_classes.circle import Circle, BlueCircle, RedCircle, GreenCircle
from core.util_classes.common_predicates import *
from core.util_classes.matrix import *
from core.util_classes.robots import Baxter
from core.util_classes.robot_predicates import *
from core.util_classes.table import Table
from core.util_classes.wall import Wall

from baxter_plan.msg import ActionMSG, FloatArrayMSG, GeomMSG, ParameterMSG, PlanMSG, PredicateMSG

from baxter_plan import execute_action

class PlanReceiver(object):
	def listen_for_plans(self):
		rospy.init_node('plan_receiver')
		rospy.Subscriber('Plan', PlanMSG, self._execute_plan)
		rospy.spin()

	def _execute_plan(self, data):
			plan = self._build_plan(data)
			pub = rospy.Publisher('Failed_Predicates', String, queue_size=10)
			for action in plan.actions:
				failed_action_preds = action.get_failed_preds()
				if failed_action_preds:
					pub.publish("Failed action {0}. Failed preds: {1}".format(action.name, str(failed_action_preds)))
					# return
				execute_action(action)
	
	def _build_plan(self, data):
		print "Building plan."
		env = Environment()
		params = {}
		for param in data.parameters:
			new_param = self._build_param(param)
			params[new_param.name] = new_param

		actions = []
		for action in data.actions:
			actions.append(self._build_action(action, params, env))

		return Plan(params, actions, data.horizon, env)


	def _build_action(self, data, plan_params, env):
		params = []
		for param in data.parameters:
			params.append(plan_params[param])

		preds = []
		for pred in data.predicates:
			preds.append(self._build_actionpred(pred, plan_params, env))

		return Action(data.step_num, data.name, data.active_timesteps, params, preds)


	def _build_actionpred(self, data, plan_params, env):
		actionpred = {}
		actionpred['negated'] = data.negated
		actionpred['hl_info'] = data.hl_info
		actionpred['pred'] = self._build_pred(data.pred, plan_params, env)
		actionpred['active_timesteps'] = (data.active_timesteps[0], data.active_timesteps[1])
		return actionpred


	def _build_pred(self, data, plan_params, env):
		pred_class = eval(data.type_name)
		params = []
		for param in data.params:
			if param.name in plan_params.keys():
				params.append(plan_params[param.name])
			else:
				params.append(self._build_param(param))
		return pred_class(data.name, params, data.param_types, env)


	def _build_param(self, data):
		attr_types = {}

		attr_types['_type'] = str
		attr_types['name'] = str

		if data.geom.type_name != '':
			attr_types['geom'] = eval(data.geom.type_name)

		if data.type_name == 'Robot':
			attr_types['pose'] = Vector1d
			attr_types['lArmPose'] = ArmPose7d
			attr_types['lGripper'] = Vector1d
			attr_types['rArmPose'] = ArmPose7d
			attr_types['rGripper'] = Vector1d
		elif data.type_name == "RobotPose":
			attr_types['value'] = Vector1d
			attr_types['lArmPose'] = ArmPose7d
			attr_types['lGripper'] = Vector1d
			attr_types['rArmPose'] = ArmPose7d
			attr_types['rGripper'] = Vector1d
		elif data.type_name == 'Can':
			attr_types['pose'] = Vector3d
			attr_types['rotation'] = Vector3d
		elif data.type_name == 'Obstacle':
			attr_types['pose'] = Vector3d
			attr_types['rotation'] = Vector3d
		elif data.type_name == 'EEPose':
			attr_types['value'] = Vector3d
			attr_types['rotation'] = Vector3d
		elif data.type_name == 'Target':
			attr_types['value'] = Vector3d
			attr_types['rotation'] = Vector3d
		else:
			raise Exception("Missing {0} in _build_param.".format(data.type_name))

		if data.is_symbol:
			new_param = Symbol(attr_types=attr_types)
		else:
			new_param = Object(attr_types=attr_types)

		new_param._type = data.type_name
		new_param.name = data.name

		if data.geom.type_name != '':
			new_param.geom = self._build_geom(data.geom)

		if data.type_name == 'Robot':
			new_param.lArmPose = self._build_2D_numpy_array(data.lArmPose)
			new_param.lGripper = self._build_2D_numpy_array(data.lGripper)
			new_param.rArmPose = self._build_2D_numpy_array(data.rArmPose)
			new_param.rGripper = self._build_2D_numpy_array(data.rGripper)
			new_param.pose = self._build_2D_numpy_array(data.pose)
		elif data.type_name == "RobotPose":
			new_param.lArmPose = self._build_2D_numpy_array(data.lArmPose)
			new_param.lGripper = self._build_2D_numpy_array(data.lGripper)
			new_param.rArmPose = self._build_2D_numpy_array(data.rArmPose)
			new_param.rGripper = self._build_2D_numpy_array(data.rGripper)
			new_param.value = self._build_2D_numpy_array(data.value)
		elif data.type_name == 'Can':
			new_param.rotation = self._build_2D_numpy_array(data.rotation)
			new_param.pose = self._build_2D_numpy_array(data.pose)
		elif data.type_name == 'Obstacle':
			new_param.rotation = self._build_2D_numpy_array(data.rotation)
			new_param.pose = self._build_2D_numpy_array(data.pose)
		elif data.type_name == 'EEPose':
			new_param.rotation = self._build_2D_numpy_array(data.rotation)
			new_param.value = self._build_2D_numpy_array(data.value)
		elif data.type_name == 'Target':
			new_param.rotation = self._build_2D_numpy_array(data.rotation)
			new_param.value = self._build_2D_numpy_array(data.value)

		for attr in data.undefined_attrs:
			setattr(new_param, attr, 'undefined')

		return new_param


	def _build_geom(self, data):
		geom_class = eval(data.type_name)
		attrs = {}
		for attr in data.attrs.split(", "):
			attr = attr.split(": ")
			attrs[attr[0]] = attr[1]

		if issubclass(geom_class, Can):
			radius = float(attrs["'radius'"])
			height = float(attrs["'height'"])
			return geom_class(radius, height)
		elif geom_class is Baxter:
			geometry = geom_class()
			# Fix this
			geometry.shape = "/home/michael/robot_work/tampy/models/baxter/baxter.xml"
			return geometry
		elif geom_class is Box:
			dim = [float(attrs["'length'"]), float(attrs["'height'"]), float(attrs["'width'"])]
			return geom_class(dim)
		elif geom_class is Table:
			raise Exception("Yeah, fix this. It's in _build_geom of plan_receiver.")
		elif issubclass(geom_class, Circle):
			radius = float(attrs["'radius'"])
			return geom_class(radius)
		elif geom_class is Wall:
			wall_type = attrs["'wall_type'"]
			return geom_class(wall_type)

		raise Exception('Geometry {0} not implemented yet.'.format(data.type_name))


	def _build_2D_numpy_array(self, float_array_msgs):
		new_array = []
		for row in float_array_msgs:
			new_array.append(row.data)
		return np.array(new_array)


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
