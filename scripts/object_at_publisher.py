#!/usr/bin/env python3

import rospy
from shapely import Polygon, Point
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

import json

class AtPublisher(object):

    def __init__(self) -> None:
        
        rospy.init_node('object_at_publisher')
        self.rate = rospy.Rate(10)

        # Read in boundaries
        self.boundaries = {}
        
        # Default at_predicate value
        self.at_pred = 'none'

        # Get 'at_boundaries' parameter
        while not rospy.search_param('at_boundaries'):
            rospy.logerr_once("'at_boundaries' parameter is not set")
        
        self.boundaries = rospy.get_param('at_boundaries')

        if type(self.boundaries) != dict:
            rospy.logerr("'at_boundaries' parameter is not a dict, setting to empty dict")
            self.boundaries = {}

        # create polygons
        self.polygons = {}
        for boundry_name, boundry_points in self.boundaries.items():
            self.polygons[boundry_name] = Polygon(boundry_points)

        # Create publisher
        self.at_pub = rospy.Publisher('/object_at_predicate', String, queue_size=10)
        rospy.loginfo('at_publisher is running')

        while not rospy.is_shutdown():
            self.at_pub.publish(String(self.at_pred))
            self.rate.sleep()
        
if __name__ == '__main__':
    try:
        AtPublisher()
    except rospy.ROSException as e:
        rospy.logerr(e)