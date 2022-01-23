import re
from matplotlib.transforms import Transform
from gazebo_msgs.msg import ContactState, ModelState  # , GetModelList
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from std_msgs.msg import String
# Used for publishing mara joint angles.
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSProfile
import argparse
import subprocess
from gazebo_msgs.srv import SpawnEntity
from gym_gazebo2.utils import ut_generic, ut_launch, ut_mara, ut_math, ut_gazebo, tree_urdf, general_utils
from scipy.stats import skew
from tf2_msgs import msg
from geometry_msgs.msg import Twist, Vector3
from PyKDL import ChainJntToJacSolver  # For KDL Jacobians
from ros2pkg.api import get_prefix_path
from std_srvs.srv import Empty
from control_msgs.msg import JointTrajectoryControllerState
# Used for publishing mara joint angles.
from trajectory_msgs.msg import JointTrajectory
from rclpy.qos import qos_profile_sensor_data
import rclpy
from gym.utils import seeding
from gym_gazebo2.utils import ut_launch, ut_generic, ut_mara, ut_math, tree_urdf, general_utils
from gym import utils, spaces
import transforms3d as tf3d
import math
import sys
import signal
import psutil
import os
import copy
import numpy as np
import time
import gym
gym.logger.set_level(40)  # hide warnings


class MyRobot(gym.Env):
    """
    TODO. Define the environment.
    """

    def __init__(self):
        """
        Initialize the Robot environemnt
        """
        # Manage command line args
        args = ut_generic.getArgsParserMARA().parse_args()
        self.gzclient = args.gzclient
        self.realSpeed = args.realSpeed
        self.velocity = args.velocity
        self.multiInstance = args.multiInstance
        self.port = args.port

        # Set the path of the corresponding URDF file
        # xacro my_robot.urdf.xacro > my_robot.urdf
        urdfPath = get_prefix_path(
            "my_robot_description") + "/share/my_robot_description/urdf/my_robot.urdf"

        # Launch robot in a new Process
        self.launch_subp = ut_launch.startLaunchServiceProcess(
            ut_launch.generateLaunchDescriptionROBOT(
                self.gzclient, self.realSpeed, self.multiInstance, self.port, urdfPath))

        # Create the node after the new ROS_DOMAIN_ID is set in generate_launch_description()
        rclpy.init(args=None)
        self.node = rclpy.create_node(self.__class__.__name__)

        # class variables
        self._observation_msg = None
        self.max_episode_steps = 1024  # default value, can be updated from baselines
        self.iterator = 0
        self.reset_jnts = True

        #############################
        #   Environment hyperparams
        #############################
        # Target, where should the agent reach

        self.targetPosition = Vector3(x=10., y=0., z=0.)
        self.target_orientation = np.asarray(
            [0., 0., 0., 0.])  # orientation of free wheel

        # Subscribe to the appropriate topics, taking into account the particular robot
        self._pub = self.node.create_publisher(
            Twist, '/cmd_vel')
        self._sub = self.node.create_subscription(
            msg.TFMessage, '/tf', self.observation_callback)

        # self.action_space = spaces.Box(-float('inf'), float('inf'))
        # self.observation_space = spaces.Box(-float('inf'), float('inf'))
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(
            np.array([-1, -1]), np.array([1, 1]))

        # spawn_cli = self.node.create_client(SpawnEntity, '/spawn_entity')
        # Seed the environment
        self.seed()

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg = message.transforms[0].transform

    def set_episode_size(self, episode_size):
        self.max_episode_steps = episode_size

    def take_observation(self):
        """
        Take observation from the environment and return it.
        :return: state.
        """
        # Take an observation
        rclpy.spin_once(self.node)
        # Robot State
        # print("\n matrix", general_utils.quaternion_to_matrix(quaternion))
        rotation = general_utils.euler_from_quaternion(self._observation_msg.rotation.x,
                                                       self._observation_msg.rotation.y,
                                                       self._observation_msg.rotation.z,
                                                       self._observation_msg.rotation.w)
        state = [self._observation_msg.translation.x,
                 self._observation_msg.translation.y]

        return state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - action
            - observation
            - reward
            - done (status)
        """
        self.iterator += 1

        # Execute "action"
        # Control only x and yaw
        print('\n', float(action[0]))
        self._pub.publish(Twist(linear=Vector3(
            x=float(action[0]), y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=float(action[1]))))
        # Take an observation
        obs = self.take_observation()

        # Compute reward
        distance = ut_math.computeDistance(
            self._observation_msg, self.targetPosition)
        print("\ndistance:", distance)

        reward = ut_math.computeRewardDistance(distance)
        print("\nreward:", reward)

        # Calculate if the env has been solved
        done = bool(self.iterator == self.max_episode_steps)
        # Return the corresponding observations, rewards, etc.
        return obs, reward, done, {}

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0

        # Take an observation
        obs = self.take_observation()

        # Return the corresponding observation
        return obs

    def close(self):
        print("Closing " + self.__class__.__name__ + " environment.")
        self.node.destroy_node()
        parent = psutil.Process(self.launch_subp.pid)
        for child in parent.children(recursive=True):
            child.kill()
        rclpy.shutdown()
        parent.kill()
