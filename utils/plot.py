# A good version of the plot function

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

def plot_3d_trajectories(trajectories, ground_truth=None, labels=None, fig_label=None, to_mm = 1000.0, filename=None):
    """
    Plots a set of 3D trajectories with enhanced visuals.
    
    :param trajectories: List of 3D trajectories, where each trajectory is an array-like structure of shape (N, 3),
                         representing N points with 3 coordinates (x, y, z).
    :param ground_truth: Ground truth trajectory, optional.
    :param labels: List of labels for each trajectory, optional. Default is None.
    :param fig_label: Title for the plot, optional.
    """

    z_floor = 0.063*to_mm

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18  # Global font size

    fig = plt.figure(figsize=(10, 8), dpi=300)  # Larger figure with higher resolution
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up a color map to differentiate trajectories
    cmap = plt.get_cmap("viridis")  

    loss_color = cmap(0.5)
    kalman_color = cmap(0.9)
    model_color = 'mediumvioletred'
    colors = [loss_color, kalman_color, model_color]

    # Plot the ground truth if provided
    if ground_truth is not None:
        ground_truth = ground_truth*to_mm
        
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                linewidth=2.5, color='black', label="Original Trajectory", alpha=0.8)

        # Project the ground truth trajectory onto z=0.076
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], np.full_like(ground_truth[:, 2], z_floor), 
                linestyle='-', color='black', linewidth=1, alpha=0.6)

    # Plot each trajectory
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        trajectory = trajectory*to_mm
        color = colors[i]
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                label=labels[i] if labels else f'Trajectory {i+1}',
                linewidth=2, color=color, linestyle='--')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color=color, s=30, marker='s')

        # Project trajectory onto the plane z=0.076
        ax.plot(trajectory[:, 0], trajectory[:, 1], np.full_like(trajectory[:, 2], z_floor), 
                linestyle=':', linewidth=2, color=color, alpha=0.6)
        
    ax.scatter(ground_truth[0, 0], ground_truth[0, 1], ground_truth[0, 2],
                linewidth=2.5, color='red', marker='x', s=40)
    ax.scatter(ground_truth[0, 0], ground_truth[0, 1], z_floor,
                linewidth=2.5, color='red', marker='x', s=40, alpha=0.6)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    print(zlim)

    eps = 0.0
    ax.set_zlim(z_floor, 0.092*to_mm)
    ax.set_xlim((xlim[0] - eps), (xlim[1] + eps))
    ax.set_ylim((ylim[0] - eps), (ylim[1] + eps))

    # Create a colored XY plane at z=0.076
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = np.full_like(X, z_floor)  # Set Z to 0.076 for the XY plane
    ax.plot_surface(X, Y, Z, color='grey', alpha=0.2, rstride=100, cstride=100)


    # Set axis labels with larger font sizes
    ax.set_xlabel('X (mm)', fontsize=18, labelpad=10)
    ax.set_ylabel('Y (mm)', fontsize=18, labelpad=15)
    ax.set_zlabel('Z (mm)', fontsize=18, labelpad=1)  # Adjusted labelpad for Z axis

    # Customize ticks: reduce tick density and format them to no more than 3 decimal places
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
    ax.set_zticks(np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 5))
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#     ax.tick_params(axis='x', pad=10)  # Offset X-axis ticks by 10 points
    ax.tick_params(axis='y', pad=12)  # Offset Y-axis ticks by 15 points
#     ax.tick_params(axis='z', pad=20)  # Offset Z-axis ticks by 20 points
    
    
    # Customize the grid and ticks
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Optional: Add lighting effect for depth
    ax.view_init(elev=30, azim=135)  

    # Display legend with adjustments, moved further inside using bbox_to_anchor
    if labels:
        ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.56, 0.7), frameon=False)  # Fine-tuned position

    # Adjust layout to fit all elements well
    plt.tight_layout()
    
    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Show the plot
    # plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

def plot_kinematic_variables(ground_truth, predictions, pred_labels, time=None, filename=None):
    
    # Check that the number of prediction labels matches the number of predictions
    if len(predictions) != len(pred_labels):
        raise ValueError("Number of prediction trajectories and labels must be the same.")
    
    # Get the number of timesteps for the ground truth (assume all ground truth trajectories have the same length)
    num_timesteps_gt = ground_truth[0].shape[0]

    # Set default time for ground truth if not provided
    if time is None:
        time = np.arange(num_timesteps_gt) * 1./30  # Uniform time for ground truth

    # Use the 'viridis' colormap for predictions and set up colors
    cmap = plt.get_cmap('viridis')
    loss_color = cmap(0.5)
    kalman_color = cmap(0.9)
    model_color = 'mediumvioletred'
    colors = [model_color,kalman_color,loss_color]

    # Extend predictions to match the length of the ground truth trajectories
    extended_predictions = []
    for pred_traj in predictions:
        num_timesteps_pred = pred_traj.shape[0]
        if num_timesteps_pred < num_timesteps_gt:
            # Repeat the last value for each kinematic variable to extend the prediction
            repeated_last_value = np.tile(pred_traj[-1, :], (num_timesteps_gt - num_timesteps_pred, 1))
            extended_pred = np.vstack([pred_traj, repeated_last_value])
        else:
            extended_pred = pred_traj[:num_timesteps_gt]  # Truncate if longer
        extended_predictions.append(extended_pred)

    # Titles for the kinematic variables (adjust as per your variables)
    variable_titles = "X Y Z Qx Qy Qz Qw Gripper".split()

    import copy
    gtt = []
    prds = []
    
    for gt in ground_truth:
        x = copy.deepcopy(gt)
        x[:, :3] = x[:, :3]*1000
        gtt.append(x)
    for prd in extended_predictions:
        x = copy.deepcopy(prd)
        x[:, :3] = x[:, :3]*1000
        prds.append(x)
    
    

    # Iterate through each kinematic variable and save individual figures
    for i in range(8):
        plt.figure(figsize=(10, 6), dpi=300)  # Higher resolution for saved figures

        to_plot_ind = int(gtt[0].shape[0]*0.8)
        time = time[:to_plot_ind]

        # Plot ground truth trajectories (all in black)
        for gt_idx, gt_traj in enumerate(gtt):
            plt.plot(time, gt_traj[:to_plot_ind, i], color='black', linewidth=5, label='Ground Truth' if gt_idx == 0 else "")

        # Plot extended prediction trajectories (each in a different color from viridis colormap)
        for j, pred_traj in enumerate(prds):
            plt.plot(time, pred_traj[:to_plot_ind, i], color=colors[j], linewidth=3, label=pred_labels[j])

        # Set title and labels
        # plt.title(f'{variable_titles[i]}', fontsize=30)
        plt.ylabel(variable_titles[i] + " (mm)" if i in [0, 1, 2] else variable_titles[i], fontsize=36)
        plt.xlabel('Time (Seconds)', fontsize=36)
        plt.tick_params(axis='both', which='minor', labelsize=14)

        # Remove the grid for a cleaner look
        plt.grid(False)

        # Add a legend only to the first plot for the ground truth and prediction labels
        plt.legend(loc='lower left', fontsize=20)

        # Save the figure with high resolution
        plt.tight_layout()
        plt.savefig(filename+"_"+variable_titles[i]+".png", dpi=300)
        plt.close()  # Close the figure to avoid overlapping of plots