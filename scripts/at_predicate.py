#!/usr/bin/env python3

import rospy
from shapely import Polygon, Point
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

class AtPublisher(object):

    def __init__(self) -> None:
        
        rospy.init_node('at_publisher')
        self.rate = rospy.Rate(10)

        # Read in boundaries
        self.boundaries = {}
        
        # Default at_predicate value
        self.at_pred = 'none'

        try:
            self.boundaries = rospy.get_param('at_boundaries')
        except:
            rospy.logerr('Boundaries parameter does not exist')
            rospy.signal_shutdown()

        # create polygons
        self.polygons = {}
        for boundry_name, boundry_points in self.boundaries.items():
            self.polygons[boundry_name] = Polygon(boundry_points)

        # Subscribe to pose topic
        self.odom_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.handler)

        # Create publisher
        self.at_pub = rospy.Publisher('/at_predicate', String, queue_size=10)

        rospy.loginfo('at_publisher is running')

        while not rospy.is_shutdown():
            self.at_pub.publish(String(self.at_pred))
            self.rate.sleep()

    def handler(self, odom):
        pose = odom.pose.pose.position

        # Build point
        point = Point(pose.x, pose.y)

        # Check if any polygon contains point
        for boundry_name, poly in self.polygons.items():
            if poly.contains(point):
                self.at_pred = boundry_name
                return
        
        self.at_pred = 'none'
        
if __name__ == '__main__':
    try:
        AtPublisher()
    except rospy.ROSException as e:
        rospy.logerr(e)