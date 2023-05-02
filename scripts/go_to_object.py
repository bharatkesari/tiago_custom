#!/usr/bin/env python3

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
from robot_custom.srv import ModelPose, StringBool
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


import tf.transformations as tft
import math
import numpy as np
import time

class GoToObject(object):

    def __init__(self) -> None:
        
        rospy.init_node('go_to_object')

        # Wait for get_model_pose service to become available
        rospy.wait_for_service('/get_model_pose')
        self.get_model_pose = rospy.ServiceProxy('get_model_pose', ModelPose)

        # Wait for move_base to become available
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base.wait_for_server()

        # Define go_to_object service
        self.go_to_object = rospy.Service('go_to_object/', StringBool, self.handler)

        t = int(time.time())
        np.random.seed(t)

        while not rospy.is_shutdown():
            rospy.spin()

    def handler(self, req):
        
        try:
            model_info = rospy.get_param(f'objects/{req.msg}')
        except KeyError:
            rospy.logerr(f'Model named "{req.msg}" not found')
            return False
        
        model_pose = self.get_model_pose(req.msg)

        if not model_pose.success:
            return False
        else:
            goal = self.build_goal(model_pose, model_info)

            print(str(goal))

        return self.dispatch_goal(goal)

    def build_goal(self, model_pose, model_info):
        # Build goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()

        # If object is symettric, build a target point at a random point around the object
        if model_info['symetrical']:
            # Generate a random vector with three components
            vec = np.random.uniform(-1, 1, size=(2,))

            # Normalize the vector to obtain a unit vector
            unit_vec = vec / np.linalg.norm(vec)

            # Multiply by radius
            displacement_vec = unit_vec * model_info['radius']

            # Set position
            goal.target_pose.pose.position.x = displacement_vec[0] + model_pose.position.x
            goal.target_pose.pose.position.y = displacement_vec[0] + model_pose.position.y
            goal.target_pose.pose.position.z = 0

            # Set orientation
            # Compute the angle between the XY vector and the X-axis
            theta = math.atan2(-displacement_vec[1], -displacement_vec[0])

            # Compute the quaternion from the angle
            quat = tft.quaternion_from_euler(0, 0, theta)

            goal.target_pose.pose.orientation.x = quat[0]
            goal.target_pose.pose.orientation.y = quat[1]
            goal.target_pose.pose.orientation.z = quat[2]
            goal.target_pose.pose.orientation.w = quat[3]

        return goal

    
    def dispatch_goal(self, goal):
        # Send goal
        print(str(goal))
        self.move_base.send_goal(goal)
        rospy.loginfo('Sending goal')
        self.move_base.wait_for_result()

        if (self.move_base.get_state() == GoalStatus.SUCCEEDED):
            return True
            
        return False

if __name__ == '__main__':
    try:
        GoToObject()
    except rospy.ROSException as e:
        rospy.logerr(e)