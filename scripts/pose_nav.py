#!/usr/bin/env python

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from robot_custom.srv import StringBool

import tf.transformations as tft

class WaypointNav(object):

    def __init__(self) -> None:
        
        rospy.init_node('pose_nav')

        # Wait for move_base to become available
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base.wait_for_server()

        # Define go_to_object service
        self.object_nav = rospy.Service('pose_nav/', StringBool, self.handler)

        rospy.loginfo("/pose_nav service running")

        while not rospy.is_shutdown():
            rospy.spin()
    
    def handler(self, req):

        rospy.loginfo(f"recieved request")

        goal = self.build_goal(req.pose)

        return self.dispatch_goal(goal)


    def build_goal(self, pose):

        # Build goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()

        # Set cartesian coords
        goal.target_pose.pose.position.x = pose[0]
        goal.target_pose.pose.position.y = pose[1]
        goal.target_pose.pose.position.z = pose[2]

        quat = tft.quaternion_from_euler(pose[3], pose[4], pose[5])

        # Set orientation as quaternion
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