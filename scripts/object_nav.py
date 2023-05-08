#!/usr/bin/env python3

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
from robot_custom.srv import ObjPose, StringBool
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


import tf.transformations as tft
import math
import numpy as np

class ObjectNav(object):

    def __init__(self) -> None:
        
        rospy.init_node('object_nav')

        # Wait for get_object_pose service to become available
        rospy.wait_for_service('/get_object_pose')
        self.get_object_pose = rospy.ServiceProxy('get_object_pose', ObjPose)

        # Wait for move_base to become available
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base.wait_for_server()

        # Define go_to_object service
        self.object_nav = rospy.Service('object_nav/', StringBool, self.handler)

        rospy.loginfo("/object_nav service running")

        while not rospy.is_shutdown():
            rospy.spin()

    def handler(self, req):
        
        rospy.loginfo(f"recieved request for {req.msg}")

        # Get object info
        try:
            object_info = rospy.get_param(f'objects/{req.msg}')
        except KeyError:
            rospy.logerr(f'object named "{req.msg}" not found')
            return False
        
        # Get object position in map frame
        object_pose = self.get_object_pose(req.msg)

        if not object_pose.success:
            return False
        else:
            goal = self.build_goal(object_pose, object_info)

        return self.dispatch_goal(goal)

    def build_goal(self, model_pose, model_info):
        # Build goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()

        # Get facing and displacement angles
        f_angle = model_info['direction']
        d_angle = (f_angle + math.pi) if (f_angle < math.pi) else (f_angle - math.pi)

        # Compute displacement vector
        offset = model_info['offset']
        d_vec = offset * np.array([math.cos(d_angle), math.sin(d_angle)])

        # Set cartesian coords of goal
        goal.target_pose.pose.position.x = model_pose.position.x + d_vec[0]
        goal.target_pose.pose.position.y = model_pose.position.y + d_vec[1]

        # Set orientation of goal
        quat = tft.quaternion_from_euler(0, 0, f_angle)
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]  
        
        return goal

    
    def dispatch_goal(self, goal):
        # Send goal
        self.move_base.send_goal(goal)
        self.move_base.wait_for_result()

        rospy.loginfo(f"attempting to navigate to:\n {str(goal.target_pose.pose)}")

        if (self.move_base.get_state() == GoalStatus.SUCCEEDED):
            rospy.loginfo("navigation successful")
            return True
            
        return False

if __name__ == '__main__':
    try:
        ObjectNav()
    except rospy.ROSException as e:
        rospy.logerr(e)