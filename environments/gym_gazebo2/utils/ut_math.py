import math
import numpy as np


def rmseFunc(eePoints):
    """
    Computes the Residual Mean Square Error of the difference between current and desired
     end-effector position
    """
    rmse = np.sqrt(np.mean(np.square(eePoints), dtype=np.float32))
    return rmse


def computeReward(rewardDist, rewardOrientation=0, collision=False):
    alpha = 5
    beta = 1.5
    gamma = 1
    delta = 3
    eta = 0.03
    done = 0.02

    distanceReward = (math.exp(-alpha * rewardDist) - math.exp(-alpha)) \
        / (1 - math.exp(-alpha)) + 10 * (math.exp(-alpha/done * rewardDist) - math.exp(-alpha/done)) \
        / (1 - math.exp(-alpha/done))
    orientationReward = (1 - (rewardOrientation / math.pi)
                         ** beta + gamma) / (1 + gamma)

    if collision:
        rewardDist = min(rewardDist, 0.5)
        collisionReward = delta * (2 * rewardDist)**eta
    else:
        collisionReward = 0

    return distanceReward * orientationReward - 1 - collisionReward


def computeDistance(vect1, vect2):
    return np.sqrt(1/2*((vect1.translation.x - vect2.x)**2+(vect1.translation.y - vect2.y)**2))


def computeRewardDistance(rewardDist):
    alpha = 20
    done = 0.002

    distanceReward = (math.exp(-alpha * rewardDist) - math.exp(-alpha)) \
        / (1 - math.exp(-alpha)) + 10 * (math.exp(-alpha/done * rewardDist) - math.exp(-alpha/done)) \
        / (1 - math.exp(-alpha/done))
    return distanceReward - 1
