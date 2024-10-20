import sys
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import mplcursors
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit, QHBoxLayout, QSplitter
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# Custom stream class to redirect stdout
class OutputStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        self.text_edit.append(text)

    def flush(self):
        pass  # No need to implement flush for QTextEdit

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1))  # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 900.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt
            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt * dt) / 2.0
        T_3 = (dt * dt * dt) / 3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            md = float(row[11])
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((mr, ma, me, mt, md, x, y, z))
    return measurements

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def form_clusters_via_association(tracks, reports, kalman_filter):
    association_list = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])  # 3x3 covariance matrix for position only
    chi2_threshold = kalman_filter.gate_threshold  

    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)
            print("check dissssssss",distance)
            print("check chiiiiiiiiii",chi2_threshold)

            if distance < chi2_threshold:
                association_list.append((i, j))

    clusters = []
    while association_list:
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]
        
        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]
        
        clusters.append((list(cluster_tracks), [reports[r] for r in cluster_reports]))

    return clusters

def mahalanobis_distance(track, report, cov_inv):
    residual = np.array(report) - np.array(track)
    distance = np.dot(np.dot(residual.T, cov_inv), residual)
    return distance

def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    best_report = None
    best_track_idx = None
    max_weight = -np.inf

    for i, track in enumerate(cluster_tracks):
        for j, report in enumerate(cluster_reports):
            residual = np.array(report) - np.array(track)
            weight = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
            if weight > max_weight:
                max_weight = weight
                best_report = report
                best_track_idx = i

    return best_track_idx, best_report

def select_initiation_mode(mode):
    if mode == '3-state':
        return 3
    elif mode == '5-state':
        return 5
    elif mode == '7-state':
        return 7
    else:
        raise ValueError("Invalid mode selected.")
    
def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

def correlation_check(track, measurement, doppler_threshold, range_threshold):
    last_measurement = track['measurements'][-1][0]
    last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
    measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
    distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
    
    doppler_correlated = doppler_correlation(measurement[4], last_measurement[4], doppler_threshold)
    range_satisfied = distance < range_threshold
    
    return doppler_correlated and range_satisfied

def initialize_filter_state(kalman_filter, x, y, z, vx, vy, vz, time):
    kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)

def perform_jpda(tracks, reports, kalman_filter):
    clusters = form_clusters_via_association(tracks, reports, kalman_filter)
    best_reports = []
    hypotheses = []
    probabilities = []

    print("Clusters formed:")
    for cluster_tracks, cluster_reports in clusters:
        print(f"Cluster Tracks: {cluster_tracks}, Cluster Reports: {cluster_reports}")

    for cluster_tracks, cluster_reports in clusters:
        # Generate hypotheses for each cluster
        cluster_hypotheses = []
        cluster_probabilities = []
        for track in cluster_tracks:
            for report in cluster_reports:
                # Calculate the probability of the hypothesis
                cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])
                residual = np.array(report) - np.array(track)
                probability = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
                cluster_hypotheses.append((track, report))
                cluster_probabilities.append(probability)

        # Normalize probabilities
        total_probability = sum(cluster_probabilities)
        cluster_probabilities = [p / total_probability for p in cluster_probabilities]

        print("Hypotheses and probabilities for current cluster:")
        for hypo, prob in zip(cluster_hypotheses, cluster_probabilities):
            print(f"Hypothesis: {hypo}, Probability: {prob}")

        # Select the best hypothesis based on the highest probability
        best_hypothesis_index = np.argmax(cluster_probabilities)
        best_track, best_report = cluster_hypotheses[best_hypothesis_index]

        best_reports.append((best_track, best_report))
        hypotheses.append(cluster_hypotheses)
        probabilities.append(cluster_probabilities)

    return clusters,best_reports, hypotheses, probabilities

def perform_munkres(tracks, reports, kalman_filter):
    cost_matrix = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    for track in tracks:
        track_costs = []
        for report in reports:
            distance = mahalanobis_distance(track, report, cov_inv)
            track_costs.append(distance)
        cost_matrix.append(track_costs)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_reports = [(row, reports[col]) for row, col in zip(row_ind, col_ind)]
    return best_reports

def check_track_timeout(tracks, current_time, poss_timeout=2.0, firm_tent_timeout=5.0):
    tracks_to_remove = []
    for track_id, track in enumerate(tracks):
        last_measurement_time = track['measurements'][-1][0][3]  # Assuming the time is at index 3
        time_since_last_measurement = current_time - last_measurement_time
        
        if track['current_state'] == 'Poss1' and time_since_last_measurement > poss_timeout:
            tracks_to_remove.append(track_id)
        elif track['current_state'] in ['Tentative1', 'Firm'] and time_since_last_measurement > firm_tent_timeout:
            tracks_to_remove.append(track_id)
    
    return tracks_to_remove

def plot_measurements(tracks, ax, plot_type):
    ax.clear()
    for track in tracks:
        times = [m[0][3] for m in track['measurements']]
        measurements_x = [sph2cart(*m[0][:3])[0] for m in track['measurements']]
        measurements_y = [sph2cart(*m[0][:3])[1] for m in track['measurements']]
        measurements_z = [sph2cart(*m[0][:3])[2] for m in track['measurements']]

        # Check if Sf is a list of state vectors or a single state vector
        if isinstance(track['Sf'], list) and len(track['Sf']) == len(times):
            Sf_x = [state[0] for state in track['Sf']]
            Sf_y = [state[1] for state in track['Sf']]
            Sf_z = [state[2] for state in track['Sf']]
        else:
            # If Sf is a single state vector, repeat it to match the length of times
            Sf_x = [track['Sf'][0, 0]] * len(times)
            Sf_y = [track['Sf'][1, 0]] * len(times)
            Sf_z = [track['Sf'][2, 0]] * len(times)

        if plot_type == "Range vs Time":
            ax.plot(times, measurements_x, label=f'Track {track["track_id"]} Measurement X', marker='o')
            ax.plot(times, Sf_x, label=f'Track {track["track_id"]} Sf X', linestyle='--')
            ax.set_ylabel('X Coordinate')
        elif plot_type == "Azimuth vs Time":
            ax.plot(times, measurements_y, label=f'Track {track["track_id"]} Measurement Y', marker='o')
            ax.plot(times, Sf_y, label=f'Track {track["track_id"]} Sf Y', linestyle='--')
            ax.set_ylabel('Y Coordinate')
        elif plot_type == "Elevation vs Time":
            ax.plot(times, measurements_z, label=f'Track {track["track_id"]} Measurement Z', marker='o')
            ax.plot(times, Sf_z, label=f'Track {track["track_id"]} Sf Z', linestyle='--')
            ax.set_ylabel('Z Coordinate')

    ax.set_xlabel('Time')
    ax.set_title(f'Tracks {plot_type}')
    ax.legend()

    # Add interactive data tips
    cursor = mplcursors.cursor(hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = sel.target.index
        track_id = tracks[index // len(tracks[0]['measurements'])]['track_id']
        measurement = tracks[index // len(tracks[0]['measurements'])]['measurements'][index % len(tracks[0]['measurements'])]
        time = measurement[0][3]
        sp = tracks[index // len(tracks[0]['measurements'])]['Sp']
        sf = tracks[index // len(tracks[0]['measurements'])]['Sf']
        plant_noise = tracks[index // len(tracks[0]['measurements'])]['Pf'][0, 0]  # Example of accessing plant noise

        sel.annotation.set(text=f"Track ID: {track_id}\nMeasurement: {measurement}\nTime: {time}\nSp: {sp}\nSf: {sf}\nPlant Noise: {plant_noise}")

def log_to_csv(log_file_path, data):
    with open(log_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writerow(data)

def main(input_file, track_mode, filter_option, association_type):
    log_file_path = 'detailed_log.csv'
    
    # Initialize CSV log file
    with open(log_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Time', 'Measurement X', 'Measurement Y', 'Measurement Z', 'Current State',
                      'Correlation Output', 'Associated Track ID', 'Associated Position X',
                      'Associated Position Y', 'Associated Position Z', 'Association Type',
                      'Clusters Formed', 'Hypotheses Generated', 'Probability of Hypothesis',
                      'Best Report Selected']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    measurements = read_measurements_from_csv(input_file)

    kalman_filter = CVFilter()
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    tracks = []
    track_id_list = []
    filter_states = []

    doppler_threshold = 100
    range_threshold = 100
    firm_threshold = select_initiation_mode(track_mode)
    association_method = association_type  # 'JPDA' or 'Munkres'

    # Initialize variables outside the loop
    miss_counts = {}
    hit_counts = {}
    firm_ids = set()
    state_map = {}
    state_transition_times = {}
    progression_states = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }[firm_threshold]

    last_check_time = 0
    check_interval = 0.0005  # 0.5 ms

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")
        
        current_time = group[0][3]  # Assuming the time is at index 3 of each measurement
        
        # Periodic checking
        if current_time - last_check_time >= check_interval:
            tracks_to_remove = check_track_timeout(tracks, current_time)
            for track_id in reversed(tracks_to_remove):
                print(f"Removing track {track_id} due to timeout")
                del tracks[track_id]
                track_id_list[track_id]['state'] = 'free'
                if track_id in firm_ids:
                    firm_ids.remove(track_id)
                if track_id in state_map:
                    del state_map[track_id]
                if track_id in hit_counts:
                    del hit_counts[track_id]
                if track_id in miss_counts:
                    del miss_counts[track_id]
            last_check_time = current_time

        if len(group) == 1:  # Single measurement
            measurement = group[0]
            assigned = False
            for track_id, track in enumerate(tracks):
                if correlation_check(track, measurement, doppler_threshold, range_threshold):
                    current_state = state_map.get(track_id, None)
                    if current_state == 'Poss1':
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])
                    elif current_state == 'Tentative1':
                        last_measurement = track['measurements'][-1][0]
                        dt = measurement[3] - last_measurement[3]
                        vx = (sph2cart(*measurement[:3])[0] - sph2cart(*last_measurement[:3])[0]) / dt
                        vy = (sph2cart(*measurement[:3])[1] - sph2cart(*last_measurement[:3])[1]) / dt
                        vz = (sph2cart(*measurement[:3])[2] - sph2cart(*last_measurement[:3])[2]) / dt
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), vx, vy, vz, measurement[3])
                    elif current_state == 'Firm':
                        kalman_filter.predict_step(measurement[3])
                        kalman_filter.update_step(np.array(sph2cart(*measurement[:3])).reshape(3, 1))
                    
                    track['measurements'].append((measurement, current_state))
                    track['Sf'] = kalman_filter.Sf.copy()
                    track['Sp'] = kalman_filter.Sp.copy()
                    track['Pp'] = kalman_filter.Pp.copy()
                    track['Pf'] = kalman_filter.Pf.copy()
                    hit_counts[track_id] = hit_counts.get(track_id, 0) + 1
                    assigned = True

                    # Log data to CSV
                    log_data = {
                        'Time': measurement[3],
                        'Measurement X': measurement[5],
                        'Measurement Y': measurement[6],
                        'Measurement Z': measurement[7],
                        'Current State': current_state,
                        'Correlation Output': 'Yes',
                        'Associated Track ID': track_id,
                        'Associated Position X': track['Sf'][0, 0],
                        'Associated Position Y': track['Sf'][1, 0],
                        'Associated Position Z': track['Sf'][2, 0],
                        'Association Type': 'Single',
                        'Clusters Formed': '',
                        'Hypotheses Generated': '',
                        'Probability of Hypothesis': '',
                        'Best Report Selected': ''
                    }
                    log_to_csv(log_file_path, log_data)
                    break

            if not assigned:
                new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                if new_track_id is None:
                    new_track_id = len(track_id_list)
                    track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                else:
                    track_id_list[new_track_id]['state'] = 'occupied'
                
                tracks.append({
                    'track_id': new_track_id,
                    'measurements': [(measurement, 'Poss1')],
                    'current_state': 'Poss1',
                    'Sf': kalman_filter.Sf.copy(),
                    'Sp': kalman_filter.Sp.copy(),
                    'Pp': kalman_filter.Pp.copy(),
                    'Pf': kalman_filter.Pf.copy()
                })
                state_map[new_track_id] = 'Poss1'
                state_transition_times[new_track_id] = {'Poss1': current_time}
                hit_counts[new_track_id] = 1
                initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])

                # Log data to CSV
                log_data = {
                    'Time': measurement[3],
                    'Measurement X': measurement[5],
                    'Measurement Y': measurement[6],
                    'Measurement Z': measurement[7],
                    'Current State': 'Poss1',
                    'Correlation Output': 'No',
                    'Associated Track ID': new_track_id,
                    'Associated Position X': '',
                    'Associated Position Y': '',
                    'Associated Position Z': '',
                    'Association Type': 'New',
                    'Clusters Formed': '',
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': ''
                }
                log_to_csv(log_file_path, log_data)

        else:  # Multiple measurements
            reports = [sph2cart(*m[:3]) for m in group]
            if association_method == 'JPDA':
                clusters,best_reports, hypotheses, probabilities = perform_jpda(
                    [track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter
                )            
            elif association_method == 'Munkres':
                best_reports = perform_munkres([track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter)

            for track_id, best_report in best_reports:
                current_state = state_map.get(track_id, None)
                if current_state == 'Poss1':
                    initialize_filter_state(kalman_filter, *best_report, 0, 0, 0, group[0][3])
                elif current_state == 'Tentative1':
                    last_measurement = tracks[track_id]['measurements'][-1][0]
                    dt = group[0][3] - last_measurement[3]
                    vx = (best_report[0] - sph2cart(*last_measurement[:3])[0]) / dt
                    vy = (best_report[1] - sph2cart(*last_measurement[:3])[1]) / dt
                    vz = (best_report[2] - sph2cart(*last_measurement[:3])[2]) / dt
                    initialize_filter_state(kalman_filter, *best_report, vx, vy, vz, group[0][3])
                elif current_state == 'Firm':
                    kalman_filter.predict_step(group[0][3])
                    kalman_filter.update_step(np.array(best_report).reshape(3, 1))
                
                tracks[track_id]['measurements'].append((cart2sph(*best_report) + (group[0][3], group[0][4]), current_state))
                tracks[track_id]['Sf'] = kalman_filter.Sf.copy()
                tracks[track_id]['Sp'] = kalman_filter.Sp.copy()
                tracks[track_id]['Pp'] = kalman_filter.Pp.copy()
                tracks[track_id]['Pf'] = kalman_filter.Pf.copy()
                hit_counts[track_id] = hit_counts.get(track_id, 0) + 1

                # Log data to CSV
                log_data = {
                    'Time': group[0][3],
                    'Measurement X': best_report[0],
                    'Measurement Y': best_report[1],
                    'Measurement Z': best_report[2],
                    'Current State': current_state,
                    'Correlation Output': 'Yes',
                    'Associated Track ID': track_id,
                    'Associated Position X': tracks[track_id]['Sf'][0, 0],
                    'Associated Position Y': tracks[track_id]['Sf'][1, 0],
                    'Associated Position Z': tracks[track_id]['Sf'][2, 0],
                    'Association Type': association_method,
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': best_report
                }
                log_to_csv(log_file_path, log_data)

            # Handle unassigned measurements
            assigned_reports = set(best_report for _, best_report in best_reports)
            for report in reports:
                if tuple(report) not in assigned_reports:
                    new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                    if new_track_id is None:
                        new_track_id = len(track_id_list)
                        track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                    else:
                        track_id_list[new_track_id]['state'] = 'occupied'
                    
                    tracks.append({
                        'track_id': new_track_id,
                        'measurements': [(cart2sph(*report) + (group[0][3], group[0][4]), 'Poss1')],
                        'current_state': 'Poss1',
                        'Sf': kalman_filter.Sf.copy(),
                        'Sp': kalman_filter.Sp.copy(),
                        'Pp': kalman_filter.Pp.copy(),
                        'Pf': kalman_filter.Pf.copy()
                    })
                    state_map[new_track_id] = 'Poss1'
                    state_transition_times[new_track_id] = {'Poss1': current_time}
                    hit_counts[new_track_id] = 1
                    initialize_filter_state(kalman_filter, *report, 0, 0, 0, group[0][3])

                    # Log data to CSV
                    log_data = {
                        'Time': group[0][3],
                        'Measurement X': report[0],
                        'Measurement Y': report[1],
                        'Measurement Z': report[2],
                        'Current State': 'Poss1',
                        'Correlation Output': 'No',
                        'Associated Track ID': new_track_id,
                        'Associated Position X': '',
                        'Associated Position Y': '',
                        'Associated Position Z': '',
                        'Association Type': 'New',
                        'Hypotheses Generated': '',
                        'Probability of Hypothesis': '',
                        'Best Report Selected': ''
                    }
                    log_to_csv(log_file_path, log_data)

        # Update states based on hit counts
        for track_id, track in enumerate(tracks):
            if hit_counts[track_id] >= firm_threshold and state_map[track_id] != 'Firm':
                state_map[track_id] = 'Firm'
                firm_ids.add(track_id)
                state_transition_times.setdefault(track_id, {})['Firm'] = current_time
            elif hit_counts[track_id] == 2 and state_map[track_id] != 'Tentative1':
                state_map[track_id] = 'Tentative1'
                state_transition_times.setdefault(track_id, {})['Tentative1'] = current_time
            elif hit_counts[track_id] == 1 and track_id not in state_map:
                state_map[track_id] = 'Poss1'
                state_transition_times.setdefault(track_id, {})['Poss1'] = current_time
            track['current_state'] = state_map[track_id]

    # Prepare data for CSV
    csv_data = []
    for track_id, track in enumerate(tracks):
        print(f"Track {track_id}:")
        print(f"  Current State: {track['current_state']}")
        print(f"  State Transition Times:")
        for state, time in state_transition_times.get(track_id, {}).items():
            print(f"    {state}: {time}")
        print("  Measurement History:")
        for state in ['Poss1', 'Tentative1', 'Firm']:
            measurements = [m for m, s in track['measurements'] if s == state][:3]
            print(f"    {state}: {measurements}")
        print(f"  Track Status: {track_id_list[track_id]['state']}")
        print(f"  SF: {track['Sf']}")
        print(f"  SP: {track['Sp']}")
        print(f"  PF: {track['Pf']}")
        print(f"  PP: {track['Pp']}")
        print()

        # Prepare data for CSV
        csv_data.append({
            'Track ID': track_id,
            'Current State': track['current_state'],
            'Poss1 Time': state_transition_times.get(track_id, {}).get('Poss1', ''),
            'Tentative1 Time': state_transition_times.get(track_id, {}).get('Tentative1', ''),
            'Firm Time': state_transition_times.get(track_id, {}).get('Firm', ''),
            'Poss1 Measurements': str([m for m, s in track['measurements'] if s == 'Poss1'][:3]),
            'Tentative1 Measurements': str([m for m, s in track['measurements'] if s == 'Tentative1'][:3]),
            'Firm Measurements': str([m for m, s in track['measurements'] if s == 'Firm'][:3]),
            'Track Status': track_id_list[track_id]['state'],
            'SF': track['Sf'].tolist(),
            'SP': track['Sp'].tolist(),
            'PF': track['Pf'].tolist(),
            'PP': track['Pp'].tolist()
        })
        
    # Write to CSV
    csv_file_path = 'track_summary.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Track ID', 'Current State', 'Poss1 Time', 'Tentative1 Time', 'Firm Time',
                      'Poss1 Measurements', 'Tentative1 Measurements', 'Firm Measurements',
                      'Track Status', 'SF', 'SP', 'PF', 'PP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"Track summary has been written to {csv_file_path}")

    # Add this line at the end of the function
    return tracks

class KalmanFilterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.tracks = []  # Initialize tracks
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Kalman Filter GUI')
        self.setGeometry(100, 100, 800, 600)  # Set window size

        # Main layout
        main_layout = QVBoxLayout()

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(QFont('Arial', 10))
        self.file_button = QPushButton("Select Input File")
        self.file_button.setFont(QFont('Arial', 10))
        self.file_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        main_layout.addLayout(file_layout)

        # Dropdown menus in one row
        dropdown_layout = QHBoxLayout()

        # Track initiation mode selection
        self.track_mode_label = QLabel("Track Initiation Mode")
        self.track_mode_label.setFont(QFont('Arial', 10))
        self.track_mode_combo = QComboBox()
        self.track_mode_combo.addItems(["3-state", "5-state", "7-state"])
        self.track_mode_combo.setFont(QFont('Arial', 10))
        dropdown_layout.addWidget(self.track_mode_label)
        dropdown_layout.addWidget(self.track_mode_combo)

        # Filter option selection
        self.filter_option_label = QLabel("Filter Option")
        self.filter_option_label.setFont(QFont('Arial', 10))
        self.filter_option_combo = QComboBox()
        self.filter_option_combo.addItems(["CV", "CA", "CT"])
        self.filter_option_combo.setFont(QFont('Arial', 10))
        dropdown_layout.addWidget(self.filter_option_label)
        dropdown_layout.addWidget(self.filter_option_combo)

        main_layout.addLayout(dropdown_layout)

        # Association type selection
        self.association_type_label = QLabel("Association Type")
        self.association_type_label.setFont(QFont('Arial', 10))
        self.association_type_combo = QComboBox()
        self.association_type_combo.addItems(["JPDA", "Munkres"])
        self.association_type_combo.setFont(QFont('Arial', 10))
        main_layout.addWidget(self.association_type_label)
        main_layout.addWidget(self.association_type_combo)

        # Plot type selection
        self.plot_type_label = QLabel("Plot Type")
        self.plot_type_label.setFont(QFont('Arial', 10))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Range vs Time", "Azimuth vs Time", "Elevation vs Time"])
        self.plot_type_combo.setFont(QFont('Arial', 10))
        main_layout.addWidget(self.plot_type_label)
        main_layout.addWidget(self.plot_type_combo)

        # Button layout
        button_layout = QHBoxLayout()

        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.setFont(QFont('Arial', 10))
        self.process_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.process_button.setFixedSize(100, 30)
        self.process_button.clicked.connect(self.process_data)
        button_layout.addWidget(self.process_button)

        # Plot button
        self.plot_button = QPushButton("Plot")
        self.plot_button.setFont(QFont('Arial', 10))
        self.plot_button.setStyleSheet("background-color: #FF5722; color: white;")
        self.plot_button.setFixedSize(100, 30)
        self.plot_button.clicked.connect(self.show_plot)
        button_layout.addWidget(self.plot_button)

        # Clear plot button
        self.clear_plot_button = QPushButton("Clear Plot")
        self.clear_plot_button.setFont(QFont('Arial', 10))
        self.clear_plot_button.setStyleSheet("background-color: #FF5722; color: white;")
        self.clear_plot_button.setFixedSize(100, 30)
        self.clear_plot_button.clicked.connect(self.clear_plot)
        button_layout.addWidget(self.clear_plot_button)

        # Clear output button
        self.clear_output_button = QPushButton("Clear Output")
        self.clear_output_button.setFont(QFont('Arial', 10))
        self.clear_output_button.setStyleSheet("background-color: #FF5722; color: white;")
        self.clear_output_button.setFixedSize(100, 30)
        self.clear_output_button.clicked.connect(self.clear_output)
        button_layout.addWidget(self.clear_output_button)

        # PPI plot button
        self.ppi_button = QPushButton("PPI Plot")
        self.ppi_button.setFont(QFont('Arial', 10))
        self.ppi_button.setStyleSheet("background-color: #FF5722; color: white;")
        self.ppi_button.setFixedSize(100, 30)
        self.ppi_button.clicked.connect(self.show_ppi_plot)
        button_layout.addWidget(self.ppi_button)

        # RHI plot button
        self.rhi_button = QPushButton("RHI Plot")
        self.rhi_button.setFont(QFont('Arial', 10))
        self.rhi_button.setStyleSheet("background-color: #FF5722; color: white;")
        self.rhi_button.setFixedSize(100, 30)
        self.rhi_button.clicked.connect(self.show_rhi_plot)
        button_layout.addWidget(self.rhi_button)

        main_layout.addLayout(button_layout)

        # Splitter to separate output display and plot canvas
        splitter = QSplitter(Qt.Horizontal)

        # Output display
        self.output_display = QTextEdit()
        self.output_display.setFont(QFont('Courier', 10))
        self.output_display.setStyleSheet("background-color: black; color: white;")
        self.output_display.setReadOnly(True)
        splitter.addWidget(self.output_display)

        # Plot canvas
        self.canvas = FigureCanvas(plt.Figure(facecolor='black'))  # Set the face color to black
        splitter.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        main_layout.addWidget(splitter)

        # Redirect stdout to the output display
        sys.stdout = OutputStream(self.output_display)

        # Set main layout
        self.setLayout(main_layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.file_label.setText(file_name)
            self.input_file = file_name

    def process_data(self):
        input_file = getattr(self, 'input_file', None)
        track_mode = self.track_mode_combo.currentText()
        filter_option = self.filter_option_combo.currentText()
        association_type = self.association_type_combo.currentText()

        if not input_file:
            print("Please select an input file.")
            return

        print(f"Processing with:\nInput File: {input_file}\nTrack Mode: {track_mode}\nFilter Option: {filter_option}\nAssociation Type: {association_type}")

        # Call the main processing function with the selected options and assign the return value
        self.tracks = main(input_file, track_mode, filter_option, association_type)
        
        if self.tracks is None:
            print("No tracks were generated.")
        else:
            print(f"Number of tracks: {len(self.tracks)}")

    def show_plot(self):
        if not self.tracks:
            print("No tracks to plot.")
            return

        if len(self.tracks) == 0:
            print("Track list is empty.")
            return

        plot_type = self.plot_type_combo.currentText()
        ax = self.canvas.figure.subplots()
        plot_measurements(self.tracks, ax, plot_type)
        self.canvas.draw()

    def show_ppi_plot(self):
        if not self.tracks:
            print("No tracks to plot.")
            return

        ax = self.canvas.figure.subplots()
        self.plot_ppi(self.tracks, ax)
        self.canvas.draw()

    def show_rhi_plot(self):
        if not self.tracks:
            print("No tracks to plot.")
            return

        ax = self.canvas.figure.subplots()
        self.plot_rhi(self.tracks, ax)
        self.canvas.draw()

    def plot_ppi(self, tracks, ax):
        ax.clear()
        for track in tracks:
            measurements = track['measurements']
            x_coords = [sph2cart(*m[0][:3])[0] for m in measurements]
            y_coords = [sph2cart(*m[0][:3])[1] for m in measurements]

            # PPI plot (x vs y)
            ax.plot(x_coords, y_coords, label=f'Track {track["track_id"]} PPI', marker='o')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('PPI Plot (360°)')
        ax.legend()

    def plot_rhi(self, tracks, ax):
        ax.clear()
        for track in tracks:
            measurements = track['measurements']
            x_coords = [sph2cart(*m[0][:3])[0] for m in measurements]
            z_coords = [sph2cart(*m[0][:3])[2] for m in measurements]

            # RHI plot (x vs z)
            ax.plot(x_coords, z_coords, label=f'Track {track["track_id"]} RHI', linestyle='--')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Z Coordinate')
        ax.set_title('RHI Plot')
        ax.legend()

    def clear_plot(self):
        self.canvas.figure.clear()
        self.canvas.draw()

    def clear_output(self):
        self.output_display.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = KalmanFilterGUI()
    ex.show()
    sys.exit(app.exec_())