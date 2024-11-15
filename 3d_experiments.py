from data import *
from pyquaternion import Quaternion 

POSITION_LEFT = [ "PSML_position_x", "PSML_position_y", "PSML_position_z"]
ORIENTATION_LEFT = [ "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w"]

POSITION_RIGHT = [ "PSMR_position_x", "PSMR_position_y", "PSMR_position_z"]
ORIENTATION_RIGHT = [ "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w"]



def map_gripper_angle(value):
    """Maps gripper angle values from [-0.433, 1.042] to [0, 1] interval, with 1.042 mapping to 0 and -0.433 mapping to 1"""
    min_val = -0.4332701399999999
    max_val = 1.0417540066666666
    return (value - min_val) / (max_val - min_val)



def assign_label_column(data: pd.DataFrame, labels: pd.DataFrame):
    data['label'] = [""]*len(data)
    for i, row in labels.iterrows():
        start = row['start_frame']
        end = row['end_frame']
        data.loc[start:end, 'label'] = row['MP_name']
        return data

def plot_3d_trajectory(df, segments_df=None, animate=False, output_path='trajectory.gif'):
    """
    Create an interactive 3D plot of trajectory data with time slider and tool orientation for both left and right tools
    
    Args:
        df (pd.DataFrame): Dataframe containing position and orientation data for both tools
        segments_df (pd.DataFrame): DataFrame with start_frame, end_frame and label columns
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
    from matplotlib.widgets import Slider
    from transforms3d.quaternions import quat2mat, mat2quat
    from transforms3d.euler import quat2euler
    import numpy as np
    
    # Get orientation columns for both tools
    prefix_right = 'PSMR'
    prefix_left = 'PSML'
    
    # Right tool columns
    x_col_r = f"{prefix_right}_position_x"
    y_col_r = f"{prefix_right}_position_y"
    z_col_r = f"{prefix_right}_position_z"
    qw_col_r = f"{prefix_right}_orientation_w"
    qx_col_r = f"{prefix_right}_orientation_x"
    qy_col_r = f"{prefix_right}_orientation_y"
    qz_col_r = f"{prefix_right}_orientation_z"
    jaw_col_r = f"{prefix_right}_gripper_angle"
    
    # Left tool columns 
    x_col_l = f"{prefix_left}_position_x"
    y_col_l = f"{prefix_left}_position_y"
    z_col_l = f"{prefix_left}_position_z"
    qw_col_l = f"{prefix_left}_orientation_w"
    qx_col_l = f"{prefix_left}_orientation_x"
    qy_col_l = f"{prefix_left}_orientation_y"
    qz_col_l = f"{prefix_left}_orientation_z"
    jaw_col_l = f"{prefix_left}_gripper_angle"

    # Transform matrix to rotate orientation
    R = np.array([[1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]])
        
    # Apply transform to quaternions for both tools
    for prefix, qw, qx, qy, qz in [(prefix_right, qw_col_r, qx_col_r, qy_col_r, qz_col_r),
                                  (prefix_left, qw_col_l, qx_col_l, qy_col_l, qz_col_l)]:
        transformed_quats = []
        for _, row in df.iterrows():
            quat = np.array([row[qw], row[qx], row[qy], row[qz]])
            rot_mat = quat2mat(quat)
            transformed_rot = R @ rot_mat
            w, x, y, z = mat2quat(transformed_rot)
            transformed_quats.append([w, x, y, z])
        transformed_quats = np.array(transformed_quats)
        
        df[qw] = transformed_quats[:,0]
        df[qx] = transformed_quats[:,1]
        df[qy] = transformed_quats[:,2]
        df[qz] = transformed_quats[:,3]
        
        # Convert quaternions to euler angles
        euler_angles = np.array([quat2euler([row[qw], row[qx], row[qy], row[qz]]) 
                               for _, row in df.iterrows()])
        df[f'{prefix}_roll'] = euler_angles[:, 0]
        df[f'{prefix}_pitch'] = euler_angles[:, 1]
        df[f'{prefix}_yaw'] = euler_angles[:, 2]

    # Create color map for segments if provided
    if segments_df is not None:
        unique_labels = segments_df['MP_name'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 0.75, 0.75])
    
    # Create subplots
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Right tool 2D plots
    ax_x_r = fig.add_subplot(gs[0, 1])
    ax_y_r = fig.add_subplot(gs[1, 1])
    ax_z_r = fig.add_subplot(gs[2, 1])
    
    # Left tool 2D plots
    ax_x_l = fig.add_subplot(gs[0, 2])
    ax_y_l = fig.add_subplot(gs[1, 2])
    ax_z_l = fig.add_subplot(gs[2, 2])
    
    # Add axis labels
    ax_3d.set_xlabel('X Position')
    ax_3d.set_ylabel('Y Position')
    ax_3d.set_zlabel('Z Position')
    
    # Right tool labels
    ax_x_r.set_ylabel('Right X Position')
    ax_y_r.set_ylabel('Right Y Position')
    ax_z_r.set_xlabel('Time Frame')
    ax_z_r.set_ylabel('Right Z Position')
    
    # Left tool labels
    ax_x_l.set_ylabel('Left X Position')
    ax_y_l.set_ylabel('Left Y Position')
    ax_z_l.set_xlabel('Time Frame')
    ax_z_l.set_ylabel('Left Z Position')
    
    # Set axis limits considering both tools
    x_min = min(df[x_col_r].min(), df[x_col_l].min())
    x_max = max(df[x_col_r].max(), df[x_col_l].max())
    y_min = min(df[y_col_r].min(), df[y_col_l].min())
    y_max = max(df[y_col_r].max(), df[y_col_l].max())
    z_min = min(df[z_col_r].min(), df[z_col_l].min())
    z_max = max(df[z_col_r].max(), df[z_col_l].max())
    
    # Add some padding to the limits
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax_3d.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
    ax_3d.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
    ax_3d.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
    
    # Right tool limits
    ax_x_r.set_xlim([0, len(df)])
    ax_x_r.set_ylim([df[x_col_r].min(), df[x_col_r].max()])
    ax_y_r.set_xlim([0, len(df)])
    ax_y_r.set_ylim([df[y_col_r].min(), df[y_col_r].max()])
    ax_z_r.set_xlim([0, len(df)])
    ax_z_r.set_ylim([df[z_col_r].min(), df[z_col_r].max()])
    
    # Left tool limits
    ax_x_l.set_xlim([0, len(df)])
    ax_x_l.set_ylim([df[x_col_l].min(), df[x_col_l].max()])
    ax_y_l.set_xlim([0, len(df)])
    ax_y_l.set_ylim([df[y_col_l].min(), df[y_col_l].max()])
    ax_z_l.set_xlim([0, len(df)])
    ax_z_l.set_ylim([df[z_col_l].min(), df[z_col_l].max()])
    
    # Initialize empty lines for each segment and tool
    lines_3d_r = []
    lines_3d_l = []
    lines_x_r = []
    lines_y_r = []
    lines_z_r = []
    lines_x_l = []
    lines_y_l = []
    lines_z_l = []
    
    # Create tool geometry
    def create_tool(ax, pos, quat, jaw_angle, color='r'):
        # Tool shaft
        shaft_length = 0.02
        shaft = np.array([[0,0,0], [0,0,shaft_length]])
        
        # Jaws (V shape when open)
        jaw_length = 0.01
        jaw_angle_rad = (jaw_angle) * np.pi/3  # Map [0,1] to [60,0] degrees
        jaw1 = np.array([[0,0,shaft_length], 
                        [jaw_length*np.sin(jaw_angle_rad/2), 0, 
                         shaft_length+jaw_length*np.cos(jaw_angle_rad/2)]])
        jaw2 = np.array([[0,0,shaft_length],
                        [-jaw_length*np.sin(jaw_angle_rad/2), 0,
                         shaft_length+jaw_length*np.cos(jaw_angle_rad/2)]])
        
        # Apply rotation
        R = quat2mat([quat[3], quat[0], quat[1], quat[2]])
        shaft = (R @ shaft.T).T + pos
        jaw1 = (R @ jaw1.T).T + pos
        jaw2 = (R @ jaw2.T).T + pos
        
        # Plot tool parts
        ax.plot3D(shaft[:,0], shaft[:,1], shaft[:,2], 'k-', linewidth=2)
        ax.plot3D(jaw1[:,0], jaw1[:,1], jaw1[:,2], color+'-', linewidth=2)
        ax.plot3D(jaw2[:,0], jaw2[:,1], jaw2[:,2], color+'-', linewidth=2)
    
    if segments_df is not None:
        unique_labels = segments_df['MP_name'].unique()
        for label in unique_labels:
            # Right tool
            line_3d_r, = ax_3d.plot3D([], [], [], '-', color=color_map[label], label=label)
            line_x_r, = ax_x_r.plot([], [], '-', color=color_map[label])
            line_y_r, = ax_y_r.plot([], [], '-', color=color_map[label])
            line_z_r, = ax_z_r.plot([], [], '-', color=color_map[label])
            
            # Left tool
            line_3d_l, = ax_3d.plot3D([], [], [], '--', color=color_map[label])
            line_x_l, = ax_x_l.plot([], [], '--', color=color_map[label])
            line_y_l, = ax_y_l.plot([], [], '--', color=color_map[label])
            line_z_l, = ax_z_l.plot([], [], '--', color=color_map[label])
            
            lines_3d_r.append(line_3d_r)
            lines_3d_l.append(line_3d_l)
            lines_x_r.append(line_x_r)
            lines_y_r.append(line_y_r)
            lines_z_r.append(line_z_r)
            lines_x_l.append(line_x_l)
            lines_y_l.append(line_y_l)
            lines_z_l.append(line_z_l)
            
        # Add single legend to figure
        fig.legend(lines_3d_r, unique_labels, loc='center right')
            
    else:
        # Right tool
        line_3d_r, = ax_3d.plot3D([], [], [], 'r-', label='Right Tool')
        line_x_r, = ax_x_r.plot([], [], 'r-')
        line_y_r, = ax_y_r.plot([], [], 'r-')
        line_z_r, = ax_z_r.plot([], [], 'r-')
        
        # Left tool
        line_3d_l, = ax_3d.plot3D([], [], [], 'b--', label='Left Tool')
        line_x_l, = ax_x_l.plot([], [], 'b--')
        line_y_l, = ax_y_l.plot([], [], 'b--')
        line_z_l = ax_z_l.plot([], [], 'b--')
        
        lines_3d_r = [line_3d_r]
        lines_3d_l = [line_3d_l]
        lines_x_r = [line_x_r]
        lines_y_r = [line_y_r]
        lines_z_r = [line_z_r]
        lines_x_l = [line_x_l]
        lines_y_l = [line_y_l]
        lines_z_l = [line_z_l]
        
        # Add single legend to figure
        fig.legend(['Right Tool', 'Left Tool'], loc='center right')
    
    # Add slider
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    time_slider = Slider(
        ax=slider_ax,
        label='Time Frame',
        valmin=0,
        valmax=len(df)-1,
        valinit=0,
        valstep=1
    )
    
    def update(frame):
        ax_3d.cla()
        ax_3d.set_xlabel('X Position')
        ax_3d.set_ylabel('Y Position')
        ax_3d.set_zlabel('Z Position')
        
        # Reset 3D axis limits with padding
        ax_3d.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax_3d.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax_3d.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
        
        if segments_df is not None:
            # Get all segments up to current frame
            relevant_segments = segments_df[segments_df['end_frame'] <= frame]
            
            # Plot completed segments in order of appearance
            for i, segment in relevant_segments.iterrows():
                label = segment['MP_name']
                start = segment['start_frame']
                end = segment['end_frame']
                label_idx = np.where(unique_labels == label)[0][0]
                
                # Plot 3D trajectories
                ax_3d.plot3D(df[x_col_r][start:end], df[y_col_r][start:end], 
                           df[z_col_r][start:end], '-', color=color_map[label])
                ax_3d.plot3D(df[x_col_l][start:end], df[y_col_l][start:end], 
                           df[z_col_l][start:end], '--', color=color_map[label])
                
                # Update 2D plots for right tool
                lines_x_r[label_idx].set_data(range(start, end), df[x_col_r][start:end])
                lines_y_r[label_idx].set_data(range(start, end), df[y_col_r][start:end])
                lines_z_r[label_idx].set_data(range(start, end), df[z_col_r][start:end])
                
                # Update 2D plots for left tool
                lines_x_l[label_idx].set_data(range(start, end), df[x_col_l][start:end])
                lines_y_l[label_idx].set_data(range(start, end), df[y_col_l][start:end])
                lines_z_l[label_idx].set_data(range(start, end), df[z_col_l][start:end])
            
            # Plot current segment
            current_segment = segments_df[
                (segments_df['start_frame'] <= frame) & 
                (segments_df['end_frame'] > frame)
            ]
            
            if not current_segment.empty:
                label = current_segment.iloc[0]['MP_name']
                start = current_segment.iloc[0]['start_frame']
                label_idx = np.where(unique_labels == label)[0][0]
                
                # Plot 3D trajectories
                ax_3d.plot3D(df[x_col_r][start:frame], df[y_col_r][start:frame], 
                           df[z_col_r][start:frame], '-', color=color_map[label])
                ax_3d.plot3D(df[x_col_l][start:frame], df[y_col_l][start:frame], 
                           df[z_col_l][start:frame], '--', color=color_map[label])
                
                # Update 2D plots for right tool
                lines_x_r[label_idx].set_data(range(start, frame), df[x_col_r][start:frame])
                lines_y_r[label_idx].set_data(range(start, frame), df[y_col_r][start:frame])
                lines_z_r[label_idx].set_data(range(start, frame), df[z_col_r][start:frame])
                
                # Update 2D plots for left tool
                lines_x_l[label_idx].set_data(range(start, frame), df[x_col_l][start:frame])
                lines_y_l[label_idx].set_data(range(start, frame), df[y_col_l][start:frame])
                lines_z_l[label_idx].set_data(range(start, frame), df[z_col_l][start:frame])
                
        else:
            frames = range(frame)
            
            # Plot trajectories
            ax_3d.plot3D(df[x_col_r][:frame], df[y_col_r][:frame], df[z_col_r][:frame], 'r-')
            ax_3d.plot3D(df[x_col_l][:frame], df[y_col_l][:frame], df[z_col_l][:frame], 'b--')
            
            # Update 2D plots for right tool
            lines_x_r[0].set_data(frames, df[x_col_r][:frame])
            lines_y_r[0].set_data(frames, df[y_col_r][:frame])
            lines_z_r[0].set_data(frames, df[z_col_r][:frame])
            
            # Update 2D plots for left tool
            lines_x_l[0].set_data(frames, df[x_col_l][:frame])
            lines_y_l[0].set_data(frames, df[y_col_l][:frame])
            lines_z_l[0].set_data(frames, df[z_col_l][:frame])
        
        # Draw tools at current position
        if frame > 0:
            # Right tool
            pos_r = np.array([df[x_col_r][frame], df[y_col_r][frame], df[z_col_r][frame]])
            quat_r = np.array([df[qx_col_r][frame], df[qy_col_r][frame], 
                             df[qz_col_r][frame], df[qw_col_r][frame]])
            jaw_angle_r = map_gripper_angle(df[jaw_col_r][frame])
            create_tool(ax_3d, pos_r, quat_r, jaw_angle_r, 'r')
            
            # Left tool
            pos_l = np.array([df[x_col_l][frame], df[y_col_l][frame], df[z_col_l][frame]])
            quat_l = np.array([df[qx_col_l][frame], df[qy_col_l][frame], 
                             df[qz_col_l][frame], df[qw_col_l][frame]])
            jaw_angle_l = map_gripper_angle(df[jaw_col_l][frame])
            create_tool(ax_3d, pos_l, quat_l, jaw_angle_l, 'b')
            
        fig.canvas.draw_idle()
    
    time_slider.on_changed(update)
    update(0)  # Initialize plot
    
    plt.tight_layout()
    plt.show()

def extract_commands(data):
    """Extract delta commands from absolute positions/orientations"""
    # Initialize commands dataframe
    commands = pd.DataFrame()
    
    # For both PSMs
    for arm in ['PSML', 'PSMR']:
        # Position deltas
        for axis in ['x', 'y', 'z']:
            col = f"{arm}_position_{axis}"
            commands[f"{arm}_delta_{axis}"] = data[col].diff()
        
        # Orientation deltas
        for axis in ['x', 'y', 'z', 'w']:
            col = f"{arm}_orientation_{axis}"
            commands[f"{arm}_delta_ori_{axis}"] = data[col].diff()
            
        # Gripper angle deltas
        col = f"{arm}_gripper_angle"
        commands[f"{arm}_delta_gripper"] = data[col].diff()
            
    return commands.fillna(0)

def apply_command(curr_pose, command):
    """Apply a single command to current pose"""
    new_pose = curr_pose.copy()
    
    # For both PSMs
    for arm in ['PSML', 'PSMR']:
        # Apply position deltas
        for axis in ['x', 'y', 'z']:
            pos_col = f"{arm}_position_{axis}"
            delta_col = f"{arm}_delta_{axis}"
            new_pose[pos_col] = curr_pose[pos_col] + command[delta_col]
        
        # Apply orientation deltas
        for axis in ['x', 'y', 'z', 'w']:
            ori_col = f"{arm}_orientation_{axis}"
            delta_col = f"{arm}_delta_ori_{axis}"
            new_pose[ori_col] = curr_pose[ori_col] + command[delta_col]
            
        # Apply gripper angle deltas
        grip_col = f"{arm}_gripper_angle"
        delta_col = f"{arm}_delta_gripper"
        new_pose[grip_col] = curr_pose[grip_col] + command[delta_col]
    
    return new_pose

def simulate_losses(data, commands, p_loss=0.1, min_loss_len=5, max_loss_len=20):
    """Simulate communication losses when applying commands"""
    # Initialize output dataframes
    new_data = pd.DataFrame(columns=data.columns)
    new_data.loc[0] = data.iloc[0]  # Start with initial pose
    
    loss_intervals = pd.DataFrame(columns=['start_idx', 'end_idx'])
    
    curr_frame = 0
    while curr_frame < len(data)-1:
        # Check for loss
        if np.random.random() < p_loss:
            print(f"Loss at frame {curr_frame}")
            # Generate random loss length
            loss_len = np.random.randint(min_loss_len, max_loss_len+1)
            end_frame = min(curr_frame + loss_len, len(data)-1)
            
            # Record loss interval
            loss_intervals = pd.concat([loss_intervals, 
                                      pd.DataFrame({'start_idx': [curr_frame],
                                                  'end_idx': [end_frame]})],
                                     ignore_index=True)
            
            # Repeat last valid pose during loss
            for i in range(curr_frame+1, end_frame+1):
                new_data.loc[i] = new_data.loc[curr_frame]
            
            curr_frame = end_frame
        else:
            # Apply command to last pose
            new_data.loc[curr_frame+1] = apply_command(new_data.loc[curr_frame], 
                                                     commands.iloc[curr_frame+1])
            curr_frame += 1
            
    return new_data, loss_intervals


if __name__ == "__main__":
    import argparse
    import json

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process surgical data with simulated losses')
    parser.add_argument('--sid', type=str, help='Subject ID (e.g. S01)', default="S01")
    parser.add_argument('--tid', type=str, help='Trial ID (e.g. T04)', default="T04")
    args = parser.parse_args()

    # Load config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    base_path = "./Datasets/dV/Peg_Transfer"
    data_loader = TaskDataLoader(base_path, "G")
    
    data = data_loader.data[args.sid][args.tid][0].iloc[:1531]
    labels = data_loader.data[args.sid][args.tid][1].iloc[:1531]

    commands = extract_commands(data)
    data_with_losses, loss_intervals = simulate_losses(data, commands, 
                                                     p_loss=config['communication_loss_probability'],
                                                     min_loss_len=config['min_loss_duration'],
                                                     max_loss_len=config['max_loss_duration'])

    # rename the columns to match the expected format
    data.rename(columns={"PSML_position_x": "PSML_position_x", "PSML_position_y": "PSML_position_z", "PSML_position_z": "PSML_position_y"}, inplace=True)
    data.rename(columns={"PSMR_position_x": "PSMR_position_x", "PSMR_position_y": "PSMR_position_z", "PSMR_position_z": "PSMR_position_y"}, inplace=True)

    data.loc[:, "PSMR_position_y"] *= -1.0
    data.loc[:, "PSML_position_y"] *= -1.0
    data.loc[:, "PSML_position_x"] -= 0.095
    
    # map the gripper angle to the [0, 1] interval
    data.loc[:, "PSML_gripper_angle"] = data["PSML_gripper_angle"].apply(map_gripper_angle)
    data.loc[:, "PSMR_gripper_angle"] = data["PSMR_gripper_angle"].apply(map_gripper_angle)
    ###########################################################
    data_with_losses.rename(columns={"PSML_position_x": "PSML_position_x", "PSML_position_y": "PSML_position_z", "PSML_position_z": "PSML_position_y"}, inplace=True)
    data_with_losses.rename(columns={"PSMR_position_x": "PSMR_position_x", "PSMR_position_y": "PSMR_position_z", "PSMR_position_z": "PSMR_position_y"}, inplace=True)

    data_with_losses.loc[:, "PSMR_position_y"] *= -1.0
    data_with_losses.loc[:, "PSML_position_y"] *= -1.0
    data_with_losses.loc[:, "PSML_position_x"] -= 0.095
    # map the gripper angle to the [0, 1] interval
    data_with_losses.loc[:, "PSML_gripper_angle"] = data["PSML_gripper_angle"].apply(map_gripper_angle)
    data_with_losses.loc[:, "PSMR_gripper_angle"] = data["PSMR_gripper_angle"].apply(map_gripper_angle)

    

    plot_3d_trajectory(data_with_losses, labels, animate=True)

    # Simulate losses using config parameters
    