from config import *

KINEMATIC_COLUMNS = [ "PSML_position_x", "PSML_position_y", "PSML_position_z", \
                # "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z", \
                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w", \
                "PSML_gripper_angle", \
                "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
                # "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z", \
                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w", \
                "PSMR_gripper_angle"]
POSE_COLUMNS = [0, 1, 2, 6, 7, 8, 9, 10]

RIGHT_ARM_KINEMARICS = [
    "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
                "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w", \
                "PSMR_gripper_angle"
]

LEFT_ARM_KINEMARICS = [
    "PSML_position_x", "PSML_position_y", "PSML_position_z", \
                "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w", \
                "PSML_gripper_angle"
]

EXPERT = 2
INTERMEDIATE = 1
NOVICE = 0
SUBJECT_EXPERTISE = {
    "S01": EXPERT,
    "S02": NOVICE,
    "S03": INTERMEDIATE,
    "S04": NOVICE,
    "S05": EXPERT,
    "S06": INTERMEDIATE,
    "S07": INTERMEDIATE,
}
EXPERTS = [k for k, v in SUBJECT_EXPERTISE.items() if v == EXPERT]
INTERMEDIATES = [k for k, v in SUBJECT_EXPERTISE.items() if v == INTERMEDIATE]
NOVICES = [k for k, v in SUBJECT_EXPERTISE.items() if v == NOVICE]

N_S = 8
N_T = 6

def get_side(X, side):
    if side == 'left': return X[:8]
    else: return X[8:]

####################################################################################################################################
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class TaskDataLoader:
    MP_NAMES = []
    def __init__(self, base_path):
        self.base_path = base_path
        self.task = os.path.basename(self.base_path)
        self.data = {}
        self.median_lengths = {}

        # Load all data
        self._load_data()

    def _load_data(self):
        mp_names_set = set()
        for subject in range(1, N_S + 1):
            subject_id = f"S{subject:02d}"
            self.data[subject_id] = {}

            for trial in range(1, N_T + 1):
                trial_id = f"T{trial:02d}"
                kinematics_path = os.path.join(self.base_path, "kinematics", f"{self.task}_{subject_id}_{trial_id}.csv")
                motion_primitives_path = os.path.join(self.base_path, "motion_primitives_baseline", f"{self.task}_{subject_id}_{trial_id}.txt")

                if not os.path.exists(kinematics_path):
                    print(kinematics_path)
                    continue

                # Load kinematics data
                kinematics_df = pd.read_csv(kinematics_path, index_col=False)
                # kinematics_df = kinematics_df[KINEMATIC_COLUMNS]

                # Load motion primitives labels
                motion_primitives_data = []
                with open(motion_primitives_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i == 0: continue
                        parts = line.strip().split(' ', 2)
                        start_frame = int(parts[0])
                        end_frame = int(parts[1])
                        mp_name = parts[2]
                        if len(mp_name.split(' ')) > 2:
                            parts = mp_name.split(')', 1)
                            mp_name = parts[1][1:]
                        mp_names_set.add(mp_name) # mp_name.replace("R,", "Arm,").replace("L,", "Arm,")
                        motion_primitives_data.append([start_frame, end_frame, mp_name])
                motion_primitives_df = pd.DataFrame(motion_primitives_data, columns=["start_frame", "end_frame", "MP_name"])
                motion_primitives_df["start_frame"] = motion_primitives_df["start_frame"].astype(int)
                motion_primitives_df["end_frame"] = motion_primitives_df["end_frame"].astype(int)

                # Store the merged data in the data structure
                self.data[subject_id][trial_id] = (kinematics_df, motion_primitives_df)

        TaskDataLoader.MP_NAMES = list(mp_names_set) 

    def compute_median_lengths(self):
        """
        Computes the median length of each motion primitive across all subjects and trials.
        """
        mp_lengths = {}

        # Loop through all subjects and trials
        for subject_id, trials in self.data.items():
            for trial_id, (kinematics_df, motion_primitives_df) in trials.items():
                # Go through each motion primitive
                for _, row in motion_primitives_df.iterrows():
                    mp_name = row["MP_name"]
                    start_frame = row["start_frame"]
                    end_frame = row["end_frame"]
                    length = end_frame - start_frame + 1
                    
                    if mp_name in mp_lengths:
                        mp_lengths[mp_name].append(length)
                    else:
                        mp_lengths[mp_name] = [length]

        # Compute the median length for each motion primitive
        self.median_lengths = {mp_name: int(np.median(lengths)) for mp_name, lengths in mp_lengths.items()}

    def resample_trajectory(self, traj, target_length):
        """
        Resamples a single trajectory to the target length using linear interpolation.
        
        Args:
            traj (np.ndarray): The input trajectory of shape (n, 7).
            target_length (int): The target length to which the trajectory should be resampled.
        
        Returns:
            np.ndarray: The resampled trajectory of shape (target_length, 7).
        """
        n, d = traj.shape
        current_indices = np.linspace(0, 1, n)
        target_indices = np.linspace(0, 1, target_length)
        
        resampled_traj = np.zeros((target_length, d))
        
        for i in range(d):
            interpolator = interp1d(current_indices, traj[:, i], kind='linear')
            resampled_traj[:, i] = interpolator(target_indices)
        
        return resampled_traj

    def resample_all_motion_primitives(self):
        """
        Resamples each instance of every motion primitive across all subjects and trials to their median length,
        and saves the resampled version in the same order. Also updates the motion_primitives_df with the new 
        start_frame and end_frame after resampling.
        """
        if not self.median_lengths:
            self.compute_median_lengths()  # Make sure median lengths are computed

        # Loop through all subjects and trials
        for subject_id, trials in self.data.items():
            for trial_id, (kinematics_df, motion_primitives_df) in trials.items():
                resampled_kinematics = []  # This will store the resampled segments
                new_motion_primitives = []  # To store updated motion primitive data (start_frame, end_frame)

                current_start_frame = 0  # Keep track of the starting frame for the next resampled segment

                # Go through each motion primitive
                for _, row in motion_primitives_df.iterrows():
                    mp_name = row["MP_name"]
                    start_frame = row["start_frame"]
                    end_frame = row["end_frame"]

                    # Extract the trajectory segment corresponding to this motion primitive
                    trajectory_segment = kinematics_df.iloc[start_frame:end_frame+1].to_numpy()

                    # Resample to the median length for this motion primitive
                    target_length = self.median_lengths[mp_name]
                    resampled_segment = self.resample_trajectory(trajectory_segment, target_length)

                    # Append the resampled segment to the list
                    resampled_kinematics.append(resampled_segment)

                    # Calculate the new start and end frames
                    new_end_frame = current_start_frame + target_length - 1

                    # Update the motion primitives data with the new start_frame and end_frame
                    new_motion_primitives.append({
                        "MP_name": mp_name,
                        "start_frame": current_start_frame,
                        "end_frame": new_end_frame
                    })

                    # Update the starting frame for the next segment
                    current_start_frame = new_end_frame + 1

                # Combine all resampled segments into a single DataFrame
                resampled_kinematics = np.vstack(resampled_kinematics)
                resampled_kinematics_df = pd.DataFrame(resampled_kinematics, columns=kinematics_df.columns)

                # Update motion_primitives_df with new start and end frames
                updated_motion_primitives_df = pd.DataFrame(new_motion_primitives)

                # Store the resampled kinematics and updated motion_primitives_df back into the data structure
                self.data[subject_id][trial_id] = (resampled_kinematics_df, updated_motion_primitives_df)

    def get_motion_primitive_segments(self, subject_id, trial_id, mp_name):
        """
        Extracts all the kinematics segments for a given motion primitive (mp_name)
        from a specified subject and trial.
        
        Args:
            subject_id (str): The ID of the subject (e.g., "S01").
            trial_id (str): The ID of the trial (e.g., "T01").
            mp_name (str): The name of the motion primitive to extract (e.g., "MP_name_example").
            
        Returns:
            List[np.ndarray]: A list of kinematics segments (as np.ndarray) corresponding to the given motion primitive.
        """
        if subject_id not in self.data or trial_id not in self.data[subject_id]:
            raise ValueError(f"Data for {subject_id} and {trial_id} not found.")

        # Retrieve resampled kinematics and motion_primitives_df for the subject and trial
        kinematics_df, motion_primitives_df = self.data[subject_id][trial_id]

        # Filter the motion_primitives_df for the specified motion primitive name
        mp_segments = motion_primitives_df[motion_primitives_df["MP_name"] == mp_name]

        # Extract the corresponding kinematics segments
        segments = []
        for _, row in mp_segments.iterrows():
            start_frame = row["start_frame"]
            end_frame = row["end_frame"]
            segment = kinematics_df.iloc[start_frame:end_frame+1].to_numpy()  # Extract the kinematics data
            segments.append(segment)

        return segments

def get_label_array_from_df(n_samples, mp_df):
    mp_gestures = np.empty(n_samples, dtype=object)
    mp_gestures.fill("Idle")
    for _, row in mp_df.iterrows():
        mp_name = row['MP_name']
        start = row['start_frame']
        end = row['end_frame']
        mp_gestures[start:end+1] = mp_name
    return mp_gestures

def get_demonstrations_for_mp(data_loader, sbj: str, mp_name: str):
    all_mp_data = []
    for t in range(1, N_T + 1):
        mp_data = data_loader.get_motion_primitive_segments(sbj, f"T0{t}", mp_name)
        all_mp_data += mp_data
    return all_mp_data

def get_mp_segments_in_order(data_loader, sbj, trial):
    trial_data = data_loader.data[sbj][trial]
    for _, row in trial_data[1].iterrows():
        mp_name = row['MP_name']
        start = row['start_frame']
        end = row['end_frame']
        kinematics = trial_data[0].iloc[start:end+1].to_numpy()
        yield mp_name, kinematics

def generate_LOUO_split(data_loader, sbj):
    mp, mp_test = [], []
    kin, kin_test = [], []
    for s in data_loader.data:
        if s != sbj:
            for t in data_loader.data[s]:
                kinematics_data = data_loader.data[s][t][0]
                mp_data = data_loader.data[s][t][1]
                mp_gestures = get_label_array_from_df(kinematics_data.shape[0], mp_data)
                kin.append(kinematics_data)
                mp.append(mp_gestures)
    kin = pd.concat(kin, axis=0)
    mp = np.concatenate(mp)

    for t in data_loader.data[sbj]:
        kinematics_data = data_loader.data[sbj][t][0]
        mp_data = data_loader.data[sbj][t][1]
        mp_gestures = get_label_array_from_df(kinematics_data.shape[0], mp_data)
        kin_test.append(kinematics_data)
        mp_test.append(mp_gestures)
    kin_test = pd.concat(kin_test, axis=0)
    mp_test = np.concatenate(mp_test)

    return kin, mp, kin_test, mp_test

def generate_LOSO_split(data_loader, trl):
    mp, mp_test = [], []
    kin, kin_test = [], []
    for s in data_loader.data:
        for t in data_loader.data[s]:
            if t != trl:
                kinematics_data = data_loader.data[s][t][0]
                mp_data = data_loader.data[s][t][1]
                mp_gestures = get_label_array_from_df(kinematics_data.shape[0], mp_data)
                kin.append(kinematics_data)
                mp.append(mp_gestures)
    kin = pd.concat(kin, axis=0)
    mp = np.concatenate(mp)

    for s in data_loader.data:
        kinematics_data = data_loader.data[s][trl][0]
        mp_data = data_loader.data[s][trl][1]
        mp_gestures = get_label_array_from_df(kinematics_data.shape[0], mp_data)
        kin_test.append(kinematics_data)
        mp_test.append(mp_gestures)
    kin_test = pd.concat(kin_test, axis=0)
    mp_test = np.concatenate(mp_test)

    return kin, mp, kin_test, mp_test
