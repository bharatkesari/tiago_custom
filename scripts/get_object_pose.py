#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import GetModelState
from robot_custom.srv import ModelPose, ModelPoseResponse
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs

class GetModelPose(object):

    def __init__(self) -> None:
        
        rospy.init_node('get_object_pose')

        # Wait for get_model_state service to become available
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Define get_model_pose service
        self.get_model_pose = rospy.Service('get_object_pose/', ModelPose, self.handler)

        rospy.loginfo("/get_object_pose service is running")

        # Create buffer to hold most recent map-gazebo transform
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        while not rospy.is_shutdown():
            rospy.spin()

    def handler(self, req):
        rospy.loginfo(f"recieved request for {req.name}")

        # Get coordinates of object from gazebo
        model_state = model_state = self.get_model_state(req.name, 'world')

        # Transform to map frame
        point_transformed = self.transform(model_state.pose.position)

        rospy.loginfo(f"{req.name} position:")
        rospy.loginfo(str(model_state.pose.position))

        return ModelPoseResponse(point_transformed, model_state.success)

    def transform(self, point):
        point_stamped = geometry_msgs.msg.PointStamped()
        point_stamped.header.frame_id = 'gazebo'
        point_stamped.point.x = point.x
        point_stamped.point.y = point.y
        point_stamped.point.z = point.z

        # Transform to map frame
        try:
            transform_stamped = self.buffer.lookup_transform('map', 'gazebo', rospy.Time())
            point_transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform_stamped)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Transform lookup failed: {e}")

        return point_transformed.point
    

if __name__ == '__main__':
    try:
        GetModelPose()
    except rospy.ROSException as e:
        rospy.logerr(e)
