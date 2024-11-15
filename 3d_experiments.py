from data import *

POSITION_LEFT = [ "PSML_position_x", "PSML_position_y", "PSML_position_z"]
ORIENTATION_LEFT = [ "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w"]

POSITION_RIGHT = [ "PSMR_position_x", "PSMR_position_y", "PSMR_position_z"]
ORIENTATION_RIGHT = [ "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w"]

base_path = "./Datasets/dV/Peg_Transfer"
data_loader = TaskDataLoader(base_path, "G")

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
dd = assign_label_column(*data_loader.data["S01"]["T02"])

def plot_3d_trajectory(df, segments_df=None, animate=False, output_path='trajectory.gif'):
    """
    Create an interactive 3D plot of trajectory data with time slider and tool orientation for both left and right tools
    
    Args:
        df (pd.DataFrame): Dataframe containing position and orientation data for both tools
        x_col (str): Name of x position column for right tool
        y_col (str): Name of y position column for right tool 
        z_col (str): Name of z position column for right tool
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
    
    # Create subplots
    ax_3d = fig.add_subplot(231, projection='3d')
    ax_x_r = fig.add_subplot(232)
    ax_x_l = fig.add_subplot(233)
    ax_y_r = fig.add_subplot(234)
    ax_y_l = fig.add_subplot(235)
    ax_z_r = fig.add_subplot(236)
    
    # Add axis labels
    ax_3d.set_xlabel('X Position')
    ax_3d.set_ylabel('Y Position')
    ax_3d.set_zlabel('Z Position')
    
    ax_x_r.set_xlabel('Time Frame')
    ax_x_r.set_ylabel('Right X Position')
    ax_x_l.set_xlabel('Time Frame')
    ax_x_l.set_ylabel('Left X Position')
    
    ax_y_r.set_xlabel('Time Frame')
    ax_y_r.set_ylabel('Right Y Position')
    ax_y_l.set_xlabel('Time Frame')
    ax_y_l.set_ylabel('Left Y Position')
    
    ax_z_r.set_xlabel('Time Frame')
    ax_z_r.set_ylabel('Right Z Position')
    
    # Set axis limits for both tools
    x_min = min(df[x_col_r].min(), df[x_col_l].min())
    x_max = max(df[x_col_r].max(), df[x_col_l].max())
    y_min = min(df[y_col_r].min(), df[y_col_l].min())
    y_max = max(df[y_col_r].max(), df[y_col_l].max())
    z_min = min(df[z_col_r].min(), df[z_col_l].min())
    z_max = max(df[z_col_r].max(), df[z_col_l].max())
    
    ax_3d.set_xlim([x_min, x_max])
    ax_3d.set_ylim([y_min, y_max])
    ax_3d.set_zlim([z_min, z_max])
    
    ax_x_r.set_xlim([0, len(df)])
    ax_x_r.set_ylim([df[x_col_r].min(), df[x_col_r].max()])
    ax_x_l.set_xlim([0, len(df)])
    ax_x_l.set_ylim([df[x_col_l].min(), df[x_col_l].max()])
    
    ax_y_r.set_xlim([0, len(df)])
    ax_y_r.set_ylim([df[y_col_r].min(), df[y_col_r].max()])
    ax_y_l.set_xlim([0, len(df)])
    ax_y_l.set_ylim([df[y_col_l].min(), df[y_col_l].max()])
    
    ax_z_r.set_xlim([0, len(df)])
    ax_z_r.set_ylim([df[z_col_r].min(), df[z_col_r].max()])
    
    # Initialize empty lines for each segment and tool
    lines_3d_r = []
    lines_3d_l = []
    lines_x_r = []
    lines_x_l = []
    lines_y_r = []
    lines_y_l = []
    lines_z_r = []
    
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
        for label in unique_labels:
            # Right tool
            line_3d_r, = ax_3d.plot3D([], [], [], '-', color=color_map[label], label=f'Right {label}')
            line_x_r, = ax_x_r.plot([], [], '-', color=color_map[label])
            line_y_r, = ax_y_r.plot([], [], '-', color=color_map[label])
            line_z_r, = ax_z_r.plot([], [], '-', color=color_map[label])
            
            # Left tool
            line_3d_l, = ax_3d.plot3D([], [], [], '--', color=color_map[label], label=f'Left {label}')
            line_x_l, = ax_x_l.plot([], [], '-', color=color_map[label])
            line_y_l, = ax_y_l.plot([], [], '-', color=color_map[label])
            
            lines_3d_r.append(line_3d_r)
            lines_3d_l.append(line_3d_l)
            lines_x_r.append(line_x_r)
            lines_x_l.append(line_x_l)
            lines_y_r.append(line_y_r)
            lines_y_l.append(line_y_l)
            lines_z_r.append(line_z_r)
            
        ax_3d.legend()
    else:
        # Right tool
        line_3d_r, = ax_3d.plot3D([], [], [], 'r-', label='Right Tool')
        line_x_r, = ax_x_r.plot([], [], 'r-')
        line_y_r, = ax_y_r.plot([], [], 'r-')
        line_z_r, = ax_z_r.plot([], [], 'r-')
        
        # Left tool
        line_3d_l, = ax_3d.plot3D([], [], [], 'b-', label='Left Tool')
        line_x_l, = ax_x_l.plot([], [], 'b-')
        line_y_l, = ax_y_l.plot([], [], 'b-')
        
        lines_3d_r = [line_3d_r]
        lines_3d_l = [line_3d_l]
        lines_x_r = [line_x_r]
        lines_x_l = [line_x_l]
        lines_y_r = [line_y_r]
        lines_y_l = [line_y_l]
        lines_z_r = [line_z_r]
        
        ax_3d.legend()
    
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
        ax_3d.set_xlim([x_min, x_max])
        ax_3d.set_ylim([y_min, y_max])
        ax_3d.set_zlim([z_min, z_max])
        
        if segments_df is not None:
            for i, label in enumerate(sorted(unique_labels)):
                relevant_segments = segments_df[
                    (segments_df['MP_name'] == label) & 
                    (segments_df['end_frame'] <= frame)
                ]
                
                # Initialize points arrays for both tools
                x_points_r, y_points_r, z_points_r = [], [], []
                x_points_l, y_points_l, z_points_l = [], [], []
                frames = []
                
                if not relevant_segments.empty:
                    for _, segment in relevant_segments.iterrows():
                        start = segment['start_frame']
                        end = segment['end_frame']
                        
                        # Right tool
                        x_points_r.extend(df[x_col_r][start:end])
                        y_points_r.extend(df[y_col_r][start:end])
                        z_points_r.extend(df[z_col_r][start:end])
                        
                        # Left tool
                        x_points_l.extend(df[x_col_l][start:end])
                        y_points_l.extend(df[y_col_l][start:end])
                        z_points_l.extend(df[z_col_l][start:end])
                        
                        frames.extend(range(start, end))
                        
                        # Plot completed segments
                        ax_3d.plot3D(df[x_col_r][start:end], df[y_col_r][start:end], 
                                   df[z_col_r][start:end], '-', 
                                   color=color_map[label], label=f'Right {label}' if start==0 else "")
                        ax_3d.plot3D(df[x_col_l][start:end], df[y_col_l][start:end], 
                                   df[z_col_l][start:end], '--', 
                                   color=color_map[label], label=f'Left {label}' if start==0 else "")
                
                current_segment = segments_df[
                    (segments_df['MP_name'] == label) & 
                    (segments_df['start_frame'] <= frame) & 
                    (segments_df['end_frame'] > frame)
                ]
                
                if not current_segment.empty:
                    start = current_segment.iloc[0]['start_frame']
                    
                    # Right tool
                    x_points_r.extend(df[x_col_r][start:frame])
                    y_points_r.extend(df[y_col_r][start:frame])
                    z_points_r.extend(df[z_col_r][start:frame])
                    
                    # Left tool
                    x_points_l.extend(df[x_col_l][start:frame])
                    y_points_l.extend(df[y_col_l][start:frame])
                    z_points_l.extend(df[z_col_l][start:frame])
                    
                    frames.extend(range(start, frame))
                    
                    # Plot current segments
                    ax_3d.plot3D(df[x_col_r][start:frame], df[y_col_r][start:frame], 
                               df[z_col_r][start:frame], '-', color=color_map[label])
                    ax_3d.plot3D(df[x_col_l][start:frame], df[y_col_l][start:frame], 
                               df[z_col_l][start:frame], '--', color=color_map[label])
                
                # Update 2D plots
                lines_x_r[i].set_data(frames, x_points_r)
                lines_x_l[i].set_data(frames, x_points_l)
                lines_y_r[i].set_data(frames, y_points_r)
                lines_y_l[i].set_data(frames, y_points_l)
                lines_z_r[i].set_data(frames, z_points_r)
                
            ax_3d.legend()
                
        else:
            frames = range(frame)
            
            # Plot trajectories
            ax_3d.plot3D(df[x_col_r][:frame], df[y_col_r][:frame], df[z_col_r][:frame], 'r-', label='Right Tool')
            ax_3d.plot3D(df[x_col_l][:frame], df[y_col_l][:frame], df[z_col_l][:frame], 'b-', label='Left Tool')
            
            # Update 2D plots
            lines_x_r[0].set_data(frames, df[x_col_r][:frame])
            lines_x_l[0].set_data(frames, df[x_col_l][:frame])
            lines_y_r[0].set_data(frames, df[y_col_r][:frame])
            lines_y_l[0].set_data(frames, df[y_col_l][:frame])
            lines_z_r[0].set_data(frames, df[z_col_r][:frame])
            
            ax_3d.legend()
        
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


if __name__ == "__main__":
    columns = [POSITION_RIGHT[0], POSITION_RIGHT[2], POSITION_RIGHT[1]]
    data = data_loader.data["S01"]["T04"][0][:1381]

    # rename the columns to match the expected format
    data.rename(columns={"PSML_position_x": "PSML_position_x", "PSML_position_y": "PSML_position_z", "PSML_position_z": "PSML_position_y"}, inplace=True)
    data.rename(columns={"PSMR_position_x": "PSMR_position_x", "PSMR_position_y": "PSMR_position_z", "PSMR_position_z": "PSMR_position_y"}, inplace=True)

    data.loc[:, "PSMR_position_y"] *= -1.0
    data.loc[:, "PSML_position_y"] *= -1.0

    # map the gripper angle to the [0, 1] interval
    data.loc[:, "PSML_gripper_angle"] = data["PSML_gripper_angle"].apply(map_gripper_angle)
    data.loc[:, "PSMR_gripper_angle"] = data["PSMR_gripper_angle"].apply(map_gripper_angle)

    labels = data_loader.data["S01"]["T04"][1][:1381]
    plot_3d_trajectory(data, labels, animate=True)