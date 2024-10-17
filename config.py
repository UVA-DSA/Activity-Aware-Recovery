MP_NAMES = ['Touch(R, Ball/Block/Sleeve)',
 'Grasp(R, Ball/Block/Sleeve)',
 'Untouch(Ball/Block/Sleeve, Pole)',
 'Touch(L, Ball/Block/Sleeve)',
 'Grasp(L, Ball/Block/Sleeve)',
 'Release(R, Ball/Block/Sleeve)',
 'Untouch(R, Ball/Block/Sleeve)',
 'Touch(Ball/Block/Sleeve, Pole)',
 'Release(L, Ball/Block/Sleeve)']


# Hyper Parameters
hidden_size = 128
batch_size = 32
num_input_kinematics = 11
label_embedding_dim = 4
num_output_pose = 8 # px, py, pz, qx, qy, qz, qw, grp
seconds = 1
fps = 30
pred_window = seconds * fps
num_labels = len(MP_NAMES) + 1 # to account for Idle label