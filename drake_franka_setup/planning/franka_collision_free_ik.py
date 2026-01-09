import franka_analytical_ik as frik
import pydrake.all as pd
import numpy as np

_q_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_q_high = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

def SolveCollisionFreeIKDrake(X_W_EE: pd.RigidTransform,
                         checker : pd.SceneGraphCollisionChecker,
                         red_disc = 20,
                         q_curr = np.zeros(7),
                         domain_padding = 0.05):
    
    domain = pd.HPolyhedron.MakeBox(_q_low+domain_padding, _q_high-domain_padding)
    """
    Solve collision-free inverse kinematics for a desired end-effector pose.

    Discretizes the redundant parameter (elbow angle) and uses analytical IK
    (see franka_analytical_ik/) to generate candidate configurations. The analytical
    IK returns all valid solutions (up to 4 per redundant parameter value), which are
    then filtered for collision-free configurations.

    Args:
        X_W_EE: Desired end-effector pose in world frame.
        checker: Collision checker for validating configurations.
        red_disc: Number of discretization points for redundant parameter in [-pi, pi].
        q_curr: Reference configuration for IK solver.

    Returns:
        Array of collision-free joint configurations (N x 7).
    """
    red = np.linspace(-np.pi, np.pi, red_disc)
    configs = []
    for r in red:
        solutions = frik.SolveIK(X_W_EE.GetAsMatrix4(), r, q_curr)
        for s in solutions:
            if not np.any(np.isnan(s)) and domain.PointInSet(s):
                configs.append(s)

    configs = np.array(configs)
    res = checker.CheckConfigsCollisionFree(configs, parallelize=True)
    idx_safe = np.where(res)[0]
    return configs[idx_safe]

def FindClosestCollisionFreeConfigDrake(X_W_EE: pd.RigidTransform,
                                   q_0: np.ndarray,
                                   checker : pd.SceneGraphCollisionChecker,
                                   red_disc = 20,
                                   domain_padding = 0.05
                                  ):
    domain = pd.HPolyhedron.MakeBox(_q_low+domain_padding, _q_high-domain_padding)
    """
    Find closest collision-free config via analytic inverse kinematics to q0 for a desired end-effector pose.

    Discretizes the redundant parameter (elbow angle) and uses analytical IK
    (see franka_analytical_ik/) to generate candidate configurations. The analytical
    IK returns all valid solutions (up to 4 per redundant parameter value), which are
    then filtered for collision-free configurations.

    Args:
        X_W_EE: Desired end-effector pose in world frame.
        q_0: Ref configuration .
        checker: Collision checker for validating configurations.
        red_disc: Number of discretization points for redundant parameter in [-pi, pi].

    Returns:
        Either None (if cannot solve collision-free IK problem) or the closest config.
    """
    red = np.linspace(-np.pi, np.pi, red_disc)
    configs = []
    for r in red:
        solutions = frik.SolveIK(X_W_EE.GetAsMatrix4(), r, q_0)
        for s in solutions:
            if not np.any(np.isnan(s)) and domain.PointInSet(s):
                configs.append(s)
    if not len(configs):
        return None
    configs = np.array(configs)
    res = checker.CheckConfigsCollisionFree(configs, parallelize=True)
    idx_safe = np.where(res)[0]
    cf_s = configs[idx_safe]
    closest = np.argmin(np.linalg.norm(cf_s - q_0.reshape(1,-1), axis=1))
    return cf_s[closest]
