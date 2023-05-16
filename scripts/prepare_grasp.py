#! /usr/bin/env python3

import rospy
import subprocess
from robot_custom.srv import StringBool, ObjPose

import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs

class PrepareGrasp(object):

    def __init__(self) -> None:
        

        rospy.init_node('prepare_grasp_srv')

        # Wait for get_object_pose service to become available
        rospy.wait_for_service('/get_object_pose')
        self.get_object_pose = rospy.ServiceProxy('get_object_pose', ObjPose)

        # Create buffer to hold most recent map-gazebo transform
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        rospy.sleep(2)

        self.srv = rospy.Service('/prepare_grasp', StringBool, self.handler)

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
        
        point_transformed = self.transform(object_pose.position)
        point_transformed.z += object_info['height']

        return self.move_arm(point_transformed)
            
    def move_arm(self, point):

        cmd = f'rosrun tiago_moveit_tutorial plan_arm_torso_ik {point.x} {point.y} {point.z} 0 1.57 -1.57'
        code = subprocess.run(cmd, shell=True)

        if code.returncode == 0:
            return True
        
        return False

    def transform(self, point):
        point_stamped = geometry_msgs.msg.PointStamped()
        point_stamped.header.frame_id = 'map'
        point_stamped.point.x = point.x
        point_stamped.point.y = point.y
        point_stamped.point.z = point.z

        # Transform to map frame
        try:
            transform_stamped = self.buffer.lookup_transform('base_footprint', 'map', rospy.Time())
            point_transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform_stamped)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Transform lookup failed: {e}")

        return point_transformed.point
    
if __name__ == '__main__':
    try:
        PrepareGrasp()
    except rospy.ROSException as e:
        rospy.logerr(e)