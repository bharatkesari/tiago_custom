#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped


if __name__ == '__main__':
    rospy.init_node('initpose_pub')

    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = rospy.Time.now()
    msg.pose.pose.position.x = 2
    msg.pose.pose.orientation.w = 1
    msg.pose.covariance[0] = 0.05
    msg.pose.covariance[7] = 0.05
    msg.pose.covariance[35] = 0.1

    rospy.sleep(5)

    for i in range(5):
        pub.publish(msg)
        rospy.Rate(3).sleep()
