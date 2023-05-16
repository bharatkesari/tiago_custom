#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import tf.transformations as tft

if __name__ == '__main__':

    rospy.init_node('map_to_gazebo_broadcaster')

    transform = []

    try:
        transform = rospy.get_param('transforms/gazebo_transform')
    except:
        rospy.logerr('gazebo_transform param does not exist')
        rospy.signal_shutdown()

    broadcaster = tf2_ros.TransformBroadcaster()
    transform_stamped = geometry_msgs.msg.TransformStamped()
    transform_stamped.header.frame_id = 'map'
    transform_stamped.child_frame_id = 'gazebo'
    transform_stamped.transform.translation.x = transform[0]
    transform_stamped.transform.translation.y = transform[1]
    transform_stamped.transform.translation.z = transform[2]
    
    quaternion = tft.quaternion_from_euler(transform[3], transform[4], transform[5])

    print(quaternion)

    transform_stamped.transform.rotation.x = quaternion[0]
    transform_stamped.transform.rotation.y = quaternion[1]
    transform_stamped.transform.rotation.z = quaternion[2]
    transform_stamped.transform.rotation.w = quaternion[3]
    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        transform_stamped.header.stamp = rospy.Time.now()
        broadcaster.sendTransform(transform_stamped)
        rate.sleep()
