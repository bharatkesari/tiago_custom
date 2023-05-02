#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import GetModelState
from robot_custom.srv import ModelPose, ModelPoseResponse

class GetModelPose(object):

    def __init__(self) -> None:
        
        rospy.init_node('get_model_pose')

        # Wait for get_model_state service to become available
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Define get_model_pose service
        self.get_model_pose = rospy.Service('get_model_pose/', ModelPose, self.handler)

        while not rospy.is_shutdown():
            rospy.spin()

    def handler(self, req):
        model_state = model_state = self.get_model_state(req.name, 'world')

        return ModelPoseResponse(model_state.pose.position, model_state.success)
    

if __name__ == '__main__':
    try:
        GetModelPose()
    except rospy.ROSException as e:
        rospy.logerr(e)
