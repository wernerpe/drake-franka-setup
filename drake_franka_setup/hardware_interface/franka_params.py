import numpy as np

FRANKA1_IP = "128.30.16.50"
FRANKA2_IP = "128.30.16.55"

# taken from https://frankarobotics.github.io/docs/control_parameters.html#limits-for-franka-research-3

FRANKA_MAX_Q = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
FRANKA_MIN_Q = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.876, 0.4398, -3.0508])

FRANKA_MAX_DQ = np.array([2, 1 ,1.5, 1.25, 3, 1.5, 3]) # rad/s
FRANKA_MAX_DDQ = np.array([10, 10, 10, 10, 10, 10, 10])

FRANKA_MAX_CARTESIAN_V = 3. #m/s
FRANKA_MAX_CARTESIAN_A = 9. #m/s^2
FRANKA_MAX_CARTESIAN_J = 4500. #m/s^3

FRANKA_MAX_CARTESIAN_ROT = 2.5 #rad/s

# LCM URLs for dual arm setup
# Each arm runs on its own LCM bus but uses standard channel names
# (PANDA_COMMAND, PANDA_STATUS, PANDA_HAND_COMMAND, PANDA_HAND_STATUS)
FRANKA1_LCM_URL = "udpm://239.255.76.67:7667?ttl=0"  # Arm 1 LCM bus
FRANKA2_LCM_URL = "udpm://239.255.76.68:7667?ttl=0"  # Arm 2 LCM bus