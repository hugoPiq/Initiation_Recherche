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
        """ ENV GYM"""
        self.action_space = spaces.Box(
            np.array([-np.pi, 0]).astype(np.float32),
            np.array([np.pi, 10]).astype(np.float32))
        self.observation_space = spaces.Box(
            np.array([0, -np.float('inf'), -np.float('inf')]
                     ).astype(np.float32),
            np.array([2*np.pi, np.float('inf'), np.float('inf')]).astype(np.float32))
        """ TARGET"""
        spawn_cli = self.node.create_client(SpawnEntity, '/spawn_entity')
        self.targetPosition = np.asarray(
            [3., 3., 0.])
        self.target_orientation = np.asarray(
            [0., 0., 0., 0.])  # orientation of free wheel
        modelXml = ut_gazebo.getTargetSdfRobot()
        pose = Pose()
        pose.position.x = self.targetPosition[0]
        pose.position.y = self.targetPosition[1]
        pose.position.z = self.targetPosition[2]
        pose.orientation.x = self.target_orientation[1]
        pose.orientation.y = self.target_orientation[2]
        pose.orientation.z = self.target_orientation[3]
        pose.orientation.w = self.target_orientation[0]

        # override previous spawn_request element.
        self.spawn_request = SpawnEntity.Request()
        self.spawn_request.name = "target"
        self.spawn_request.xml = modelXml
        self.spawn_request.robot_namespace = ""
        self.spawn_request.initial_pose = pose
        self.spawn_request.reference_frame = "world"

        # #ROS2 Spawn Entity
        target_future = spawn_cli.call_async(self.spawn_request)
        rclpy.spin_until_future_complete(self.node, target_future)

        """ TOPICS """
        # Subscribe to the appropriate topics, taking into account the particular robot
        self._pub = self.node.create_publisher(
            Twist, '/cmd_vel')
        self._sub = self.node.create_subscription(
            msg.TFMessage, '/tf', self.observation_callback)

        # For reset purpose
        self.reset_sim = self.node.create_client(Empty, '/reset_simulation')

        self.seed()
        self.buffer_dist_rewards = []
        self.buffer_tot_rewards = []

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
        current_position = np.array([self._observation_msg.translation.x,
                                     self._observation_msg.translation.y,
                                     self._observation_msg.translation.z])
        diff_position = current_position - self.targetPosition
        state = np.r_[rotation[2], np.reshape(diff_position[0:2], -1)]
        # state = np.r_[np.reshape(diff_position[0:2], -1)]
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
        # print("action:", action)
        # Execute "action"
        # Control only x and yaw
        self._pub.publish(Twist(linear=Vector3(
            x=float(action[1])), angular=Vector3(z=float(action[0]))))
        # Take an observation
        obs = self.take_observation()
        # Compute reward
        rewardDist = ut_math.rmseFunc(obs[1:3])
        # print(rewardDist, " ")
        reward = ut_math.computeRewardDistance(rewardDist)
        # reward = rewardDist
        # Calculate if the env has been solved
        done = bool(self.iterator == self.max_episode_steps)
        # self.buffer_dist_rewards.append(rewardDist)
        # self.buffer_tot_rewards.append(reward)
        info = {}
        # if self.iterator % self.max_episode_steps == 0:

        #     max_dist_tgt = max(self.buffer_dist_rewards)
        #     mean_dist_tgt = np.mean(self.buffer_dist_rewards)
        #     std_dist_tgt = np.std(self.buffer_dist_rewards)
        #     min_dist_tgt = min(self.buffer_dist_rewards)
        #     skew_dist_tgt = skew(self.buffer_dist_rewards)

        #     max_tot_rew = max(self.buffer_tot_rewards)
        #     mean_tot_rew = np.mean(self.buffer_tot_rewards)
        #     std_tot_rew = np.std(self.buffer_tot_rewards)
        #     min_tot_rew = min(self.buffer_tot_rewards)
        #     skew_tot_rew = skew(self.buffer_tot_rewards)

        #     info = {"infos": {"ep_dist_max": max_dist_tgt, "ep_dist_mean": mean_dist_tgt, "ep_dist_min": min_dist_tgt,
        #                       "ep_rew_max": max_tot_rew, "ep_rew_mean": mean_tot_rew, "ep_rew_min": min_tot_rew,
        #                       "ep_dist_skew": skew_dist_tgt, "ep_dist_std": std_dist_tgt, "ep_rew_std": std_tot_rew, "ep_rew_skew": skew_tot_rew}}
        #     self.buffer_dist_rewards = []
        #     self.buffer_tot_rewards = []
        # Return the corresponding observations, rewards, etc.
        return obs, reward, done, info

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0

        if self.reset_jnts is True:
            # reset simulation
            while not self.reset_sim.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('/reset_simulation service not available, waiting again...')

            reset_future = self.reset_sim.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_future)

        self.ros_clock = rclpy.clock.Clock().now().nanoseconds

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
