#!/usr/bin/env python3

import rospy 
from robot_custom.srv import ObjPose
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, PointStamped
import tf2_ros
import tf2_geometry_msgs
import math
import tf.transformations as tft

class FacingPublisher(object): 

    def __init__(self) -> None:
        
        rospy.init_node('facing_publisher')
        self.rate = rospy.Rate(10)

        # Wait for get_object_pose service to become available
        rospy.wait_for_service('/get_object_pose')
        self.get_object_pose = rospy.ServiceProxy('get_object_pose', ObjPose)

        # Create buffer to hold most recent map-gazebo transform
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        rospy.sleep(2)

        # Default at_predicate value
        self.facing_pred = 'none'

        # Load in objects
        self.obj = {}

        try:
            self.obj = rospy.get_param('facing_boundaries')
        except:
            rospy.logerr('Boundaries parameter does not exist')
            rospy.signal_shutdown()

        # Subscribe to pose topic
        self.odom_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.handler)

        # Create publisher
        self.at_pub = rospy.Publisher('/facing_predicate', String, queue_size=10)

        rospy.loginfo('facing_publisher is running')

        while not rospy.is_shutdown():
            self.at_pub.publish(String(self.facing_pred))
            self.rate.sleep()
    
    def handler(self, pose):
        point_stamped = PointStamped()
        point_stamped.header.frame_id = 'map'

        facing = 'none'

        # Check if robot is facing each obj
        for obj, value in self.obj.items():
            rospy.loginfo(obj)

            object_point = None
            # Obj is a Gazebo obj
            if value['type'] == 'object': 
            
                object_pose = self.get_object_pose(obj)

                if object_pose.success == False:
                    rospy.logerr(f'No facing boundary entry for {obj}')
                    continue
                
                object_point = object_pose.position

            # Obj is a coordinate 
            else:
                object_coords = value['location']
                object_point = Point(object_coords[0], object_coords[1], object_coords[2])
            
            point_stamped.point = object_point

            # Transform point to base_link frame
            try:
                transform_stamped = self.buffer.lookup_transform('base_link', 'map', rospy.Time())
                point_transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform_stamped)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"Transform lookup failed: {e}")

            # Get distance and angle to point
            dist = math.sqrt(point_transformed.point.x**2 + point_transformed.point.y**2)
            z_angle = math.atan2(point_transformed.point.y, point_transformed.point.x)

            # Check if robot is facing object
            if dist < value['radius'] and abs(z_angle) < value['angle']:
                facing = obj
                break

        self.facing_pred = facing

if __name__ == '__main__':
    try:
        FacingPublisher()
    except rospy.ROSException as e:
        rospy.logerr(e)

