#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import GetModelState
from robot_custom.srv import ObjPose, ObjPoseResponse
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs

class GetObjPose(object):

    def __init__(self) -> None:
        
        rospy.init_node('get_object_pose')

        # Wait for get_model_state service to become available
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Define get_obj_pose service
        self.get_obj_pose = rospy.Service('get_object_pose/', ObjPose, self.handler)

        rospy.loginfo("/get_object_pose service is running")

        # Create buffer to hold most recent map-gazebo transform
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        rospy.sleep(2)

        while not rospy.is_shutdown():
            rospy.spin()

    def handler(self, req):
        rospy.loginfo(f"recieved request for {req.name}")

        # Get coordinates of object from gazebo
        object_state = self.get_model_state(req.name, 'world')

        # Transform to map frame
        pose_transformed = self.transform(object_state.pose)

        return ObjPoseResponse(pose_transformed, object_state.success)

    def transform(self, pose):
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = 'gazebo'
        pose_stamped.header.stamp = rospy.Time.now()

        pose_stamped.pose = pose

        # Transform to map frame
        try:
            transform_stamped = self.buffer.lookup_transform('map', 'gazebo', rospy.Time())
            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform_stamped)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Transform lookup failed: {e}")

        return pose_transformed.pose
    
if __name__ == '__main__':
    try:
        GetObjPose()
    except rospy.ROSException as e:
        rospy.logerr(e)
