from plot import *
from data import *
from config import *


#### ProMP
from movement_primitives.promp import ProMP

## ProMP
def train_ProMP(Y, T): 
    promp = ProMP(n_dims=Y.shape[-1], n_weights_per_dim=20)
    promp.imitate([T] * Y.shape[0], Y)
    y_conditional_cov = np.array([1]*Y.shape[-1])
    return promp
    
def generate_ProMP(promp, y_start, y_end, t_start, t_end, mp_length):
    cpromp = promp.condition_position(y_start, t=t_start, t_max=t_end).condition_position(y_end, t=t_end, t_max=t_end)
    T = np.arange(mp_length) * 1.0/30
    Y_pred = cpromp.mean_trajectory(T)
    return Y_pred

def train_ProMPs(subject):
    DMPs = {}
    for dmp_name in MP_NAMES:
        # print(f"Training DMP for {dmp_name}")
        segments = get_demonstrations_for_mp(data_loader, subject, dmp_name)
        Y_left = np.array(segments)[:, :, POSE_COLUMNS]
        Y_right = np.array(segments)[:, :, [11 + x for x in POSE_COLUMNS]]
        T = np.arange(Y_left.shape[1]) * 1.0/30
        promp_left = train_ProMP(Y_left, T)
        promp_right = train_ProMP(Y_right, T)
        DMPs[dmp_name] = (promp_left, promp_right)
    return DMPs


# CartesianMPs
from dmp.dmp_cartesian import DMPs_cartesian
from dmp.rotation_matrix import roto_dilatation

def train_CartesianMP(Y, T): 
    dt = 1.0/30.0
    cmp = DMPs_cartesian(n_dmps = Y.shape[-1], n_bfs = 100, K = 1000, alpha_s = 4, rescale = 'rotodilatation', dt=dt, T = Y.shape[1] * dt)
    cmp.paths_regression(Y, [T] * Y.shape[0])
    return cmp
    
def generate_CartesianMP(cmp, y_start, y_end):
    cmp.x_0 = y_start
    cmp.x_goal = y_end
    Y_pred, _, _, _ = cmp.rollout()
    return Y_pred

def train_CartesianMPs(subject):
    DMPs = {}
    for dmp_name in MP_NAMES:
        print(f"Training DMP for {dmp_name}")
        segments = get_demonstrations_for_mp(data_loader, subject, dmp_name)
        Y_left = np.array(segments)[:, :, POSE_COLUMNS]
        Y_right = np.array(segments)[:, :, [11 + x for x in POSE_COLUMNS]]
        T = np.arange(Y_left.shape[1]) * 1.0/30
        cartesian_left = train_CartesianMP(Y_left, T)
        cartesian_right = train_CartesianMP(Y_right, T)
        DMPs[dmp_name] = (cartesian_left, cartesian_right)
    return DMPs


# CNMP
# TLFSD


def generate_MP(mp, y_start, y_end, t_start, t_end, mp_length, type):
    if type == "promp":
        return generate_ProMP(mp, y_start, y_end, t_start, t_end, mp_length)
    elif type == "cartesian":
        return generate_CartesianMP(mp, y_start, y_end)
    elif type == "DMP": generate_DMP(mp, y_start, y_end, t_start, t_end, mp_length, type)




if __name__ == "__main__":
    base_path = "./Datasets/dV/Peg_Transfer"
    data_loader = TaskDataLoader(base_path)

    # Compute median lengths and resample all motion primitive demonstrations
    data_loader.resample_all_motion_primitives()
