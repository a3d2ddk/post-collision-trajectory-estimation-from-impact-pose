import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils_pinn import *
from scipy.spatial.transform import Rotation as R
import sys
import os

# --- CONFIGURATION ---
if len(sys.argv) > 1:
    RUN_ID = sys.argv[1]
else:
    RUN_ID = "run_0002"

# Cube size for visualization (Slightly smaller than 1m to avoid visual clutter in trajectory plot)
VIS_CUBE_SIZE = 0.6 

# ==========================================
# MATH HELPERS
# ==========================================

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to rotation matrix"""
    if np.linalg.norm(quat) < 1e-6:
        return np.eye(3)
    # Ensure correct component order [x, y, z, w] for scipy if input is [w, x, y, z]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) 
    return rot.as_matrix()

def compute_orientation_error(q_true, q_pred):
    """Compute geodesic distance (angle in degrees) between quaternion sequences"""
    # Normalize predictions
    norms = np.linalg.norm(q_pred, axis=0, keepdims=True)
    q_pred_norm = q_pred / (norms + 1e-8)
    
    # Dot product
    dot_products = np.sum(q_true * q_pred_norm, axis=0)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Angle: theta = 2 * arccos(|<q1, q2>|)
    angles_rad = 2 * np.arccos(np.abs(dot_products))
    return np.degrees(angles_rad)

def compute_collision_geometry(collision_state):
    """Compute impact angle, faces, and relative metrics"""
    pos_alpha = collision_state['pos_alpha'].numpy()
    pos_beta = collision_state['pos_beta'].numpy()
    quat_alpha = collision_state['quat_alpha'].numpy()
    quat_beta = collision_state['quat_beta'].numpy()
    vel_alpha = collision_state['vel_alpha'].numpy()
    vel_beta = collision_state['vel_beta'].numpy()
    
    r_ab = pos_beta - pos_alpha
    distance = np.linalg.norm(r_ab)
    r_ab_norm = r_ab / (distance + 1e-8)
    
    v_rel = vel_alpha - vel_beta
    v_rel_mag = np.linalg.norm(v_rel)
    v_rel_norm = v_rel / (v_rel_mag + 1e-8)
    
    # Impact angle: 0 = Head-on
    impact_angle = np.arccos(np.clip(np.dot(v_rel_norm, r_ab_norm), -1, 1))
    impact_angle_deg = np.degrees(impact_angle)
    
    R_alpha = quaternion_to_rotation_matrix(quat_alpha)
    R_beta = quaternion_to_rotation_matrix(quat_beta)
    
    # Determine contact faces
    cube_faces = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    faces_alpha_world = (R_alpha @ cube_faces.T).T
    faces_beta_world = (R_beta @ cube_faces.T).T
    
    contact_face_alpha = np.argmax(faces_alpha_world @ r_ab_norm)
    contact_face_beta = np.argmax(faces_beta_world @ (-r_ab_norm))
    
    geometry = {
        'distance': distance,
        'impact_angle_deg': impact_angle_deg,
        'relative_velocity_mag': v_rel_mag,
        'contact_face_alpha': contact_face_alpha,
        'contact_face_beta': contact_face_beta,
        'R_alpha': R_alpha, 'R_beta': R_beta
    }
    return geometry

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def draw_wireframe_cube(ax, center, quat, color, alpha=0.8, lw=1, size=VIS_CUBE_SIZE):
    """Helper to draw a single oriented cube"""
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s], # Bottom
        [-s, -s, s],  [s, -s, s],  [s, s, s],  [-s, s, s]   # Top
    ])
    
    if np.linalg.norm(quat) > 1e-6:
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        vertices = r.apply(vertices)
    
    vertices = vertices + center
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4], # Top
        [0, 4], [1, 5], [2, 6], [3, 7]  # Pillars
    ]
    
    for edge in edges:
        ax.plot3D(*vertices[edge].T, color=color, alpha=alpha, lw=lw)

def plot_trajectories_with_poses(true_pos_a, true_quat_a, pred_pos_a, pred_quat_a,
                               true_pos_b, true_quat_b, pred_pos_b, pred_quat_b,
                               skip_frames=8):
    """Plot trajectories showing wireframe orientation snapshots"""
    fig = plt.figure(figsize=(16, 8))
    
    # --- Alpha Subplot ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(true_pos_a[0], true_pos_a[1], true_pos_a[2], 'b--', lw=1, alpha=0.4, label='GT Path')
    ax1.plot(pred_pos_a[0], pred_pos_a[1], pred_pos_a[2], 'r-', lw=2, alpha=0.8, label='Pred Path')
    
    num_frames = true_pos_a.shape[1]
    for i in range(0, num_frames, skip_frames):
        draw_wireframe_cube(ax1, true_pos_a[:, i], true_quat_a[:, i], color='blue', alpha=0.2, lw=1)
        draw_wireframe_cube(ax1, pred_pos_a[:, i], pred_quat_a[:, i], color='red', alpha=0.9, lw=1.5)
        
    ax1.set_title('Alpha Trajectory & Orientation')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.grid(True, alpha=0.3)

    # --- Beta Subplot ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(true_pos_b[0], true_pos_b[1], true_pos_b[2], color='purple', ls='--', lw=1, alpha=0.4, label='GT Path')
    ax2.plot(pred_pos_b[0], pred_pos_b[1], pred_pos_b[2], color='green', ls='-', lw=2, alpha=0.8, label='Pred Path')
    
    for i in range(0, num_frames, skip_frames):
        draw_wireframe_cube(ax2, true_pos_b[:, i], true_quat_b[:, i], color='purple', alpha=0.2, lw=1)
        draw_wireframe_cube(ax2, pred_pos_b[:, i], pred_quat_b[:, i], color='green', alpha=0.9, lw=1.5)
        
    ax2.set_title('Beta Trajectory & Orientation')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.grid(True, alpha=0.3)
    
    # Scale axes
    for ax in [ax1, ax2]:
        all_pts = np.concatenate([true_pos_a, true_pos_b], axis=1)
        mid = np.mean(all_pts, axis=1)
        max_range = np.max(np.ptp(all_pts, axis=1)) / 2.0 + 1.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    return fig

def plot_position_errors(errors_alpha, errors_beta, time_points):
    """Plot Euclidean position errors"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(time_points, errors_alpha * 1000, 'r-', linewidth=2)
    ax1.set_ylabel('Error (mm)')
    ax1.set_title(f'Alpha Position Error (Mean: {errors_alpha.mean()*1000:.2f}mm)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_points, errors_beta * 1000, 'b-', linewidth=2)
    ax2.set_ylabel('Error (mm)'); ax2.set_xlabel('Time (s)')
    ax2.set_title(f'Beta Position Error (Mean: {errors_beta.mean()*1000:.2f}mm)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_orientation_errors(errors_alpha_deg, errors_beta_deg, time_points):
    """Plot Angular errors"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(time_points, errors_alpha_deg, 'r-', linewidth=2)
    ax1.set_ylabel('Error (degrees)')
    ax1.set_title(f'Alpha Orientation Error (Mean: {errors_alpha_deg.mean():.2f}°)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_points, errors_beta_deg, 'b-', linewidth=2)
    ax2.set_ylabel('Error (degrees)'); ax2.set_xlabel('Time (s)')
    ax2.set_title(f'Beta Orientation Error (Mean: {errors_beta_deg.mean():.2f}°)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_momentum_conservation(pred_vel_alpha, pred_vel_beta, time_points):
    """Plot momentum conservation"""
    m_alpha, m_beta = 10.0, 10.0
    momentum = m_alpha * pred_vel_alpha + m_beta * pred_vel_beta
    momentum_mag = np.linalg.norm(momentum, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(time_points, momentum[0], 'r-'); axes[0,0].set_ylabel('Px')
    axes[0, 1].plot(time_points, momentum[1], 'g-'); axes[0,1].set_ylabel('Py')
    axes[1, 0].plot(time_points, momentum[2], 'b-'); axes[1,0].set_ylabel('Pz')
    axes[1, 1].plot(time_points, momentum_mag, 'k-'); axes[1,1].set_ylabel('|P|')
    
    for ax in axes.flatten(): ax.grid(True, alpha=0.3)
    plt.suptitle('Momentum Conservation (Predicted)')
    plt.tight_layout()
    return fig

def plot_collision_geometry_3d(collision_state, geometry):
    """Visualize collision instant"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pos_alpha = collision_state['pos_alpha'].numpy()
    pos_beta = collision_state['pos_beta'].numpy()
    
    # --- FIX: Extract Quaternions from collision_state instead of Matrices from geometry ---
    quat_alpha = collision_state['quat_alpha'].numpy()
    quat_beta = collision_state['quat_beta'].numpy()
    
    # Now passing the correct 4D quaternion arrays
    draw_wireframe_cube(ax, pos_alpha, quat_alpha, 'cyan', size=1.0)
    draw_wireframe_cube(ax, pos_beta, quat_beta, 'orange', size=1.0)
    
    ax.plot3D(*np.array([pos_alpha, pos_beta]).T, 'k--', label='Rel Pos')
    ax.set_title(f"Collision Geometry\nImpact Angle: {geometry['impact_angle_deg']:.1f}°")
    
    mid = (pos_alpha + pos_beta)/2
    ax.set_xlim(mid[0]-1.5, mid[0]+1.5)
    ax.set_ylim(mid[1]-1.5, mid[1]+1.5)
    ax.set_zlim(mid[2]-1.5, mid[2]+1.5)
    
    return fig

def plot_orientation_influence(collision_state, predictions):
    """Analyze energy and velocity changes"""
    geometry = compute_collision_geometry(collision_state)
    vel_alpha_pre = collision_state['vel_alpha'].numpy()
    vel_beta_pre = collision_state['vel_beta'].numpy()
    vel_alpha_post = predictions['vel_alpha'][:, 0].cpu().numpy()
    vel_beta_post = predictions['vel_beta'][:, 0].cpu().numpy()
    
    delta_v_alpha = vel_alpha_post - vel_alpha_pre
    delta_v_beta = vel_beta_post - vel_beta_pre
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Velocity Change
    ax = axes[0]
    x = np.arange(3)
    ax.bar(x - 0.2, delta_v_alpha, 0.4, label='Alpha', color='cyan')
    ax.bar(x + 0.2, delta_v_beta, 0.4, label='Beta', color='orange')
    ax.set_xticks(x); ax.set_xticklabels(['X','Y','Z'])
    ax.set_title('Velocity Change (Impulse)')
    ax.legend()
    
    # 2. Energy
    ax = axes[1]
    m = 10.0
    ke_pre = 0.5 * m * (np.sum(vel_alpha_pre**2) + np.sum(vel_beta_pre**2))
    ke_post = 0.5 * m * (np.sum(vel_alpha_post**2) + np.sum(vel_beta_post**2))
    ax.bar(['Pre-Collision', 'Post-Collision'], [ke_pre, ke_post], color=['green', 'blue'])
    ax.set_title(f'Kinetic Energy (Loss: {(ke_pre-ke_post)/ke_pre*100:.1f}%)')
    
    plt.tight_layout()
    return fig, geometry

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("="*60)
    print(f"LOADING DATA FOR: {RUN_ID}")
    print("="*60)
    
    # Load Predictions
    pred_path = 'saved_models/predictions_001.pt'
    try:
        predictions = torch.load(pred_path, weights_only=False, map_location='cpu')
    except:
        predictions = torch.load(pred_path, map_location='cpu')
    
    # Load Ground Truth
    dataset_path = f"Datasets/{RUN_ID}"
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}"); return
        
    data = load_collision_data(dataset_path)
    collision_frame = detect_collision_frame(data['pos_alpha'], data['pos_beta'])
    
    # Align Data
    initial_frame = collision_frame - 1
    collision_state = get_collision_state(data, initial_frame)
    trajectory = get_post_collision_trajectory(data, collision_frame, start_offset=0)
    pinn_data = prepare_pinn_data(collision_state, trajectory)
    
    # --- DATA EXTRACTION ---
    # Positions (m)
    true_pos_a = pinn_data['ground_truth_pos'][:3, :80].cpu().numpy()
    true_pos_b = pinn_data['ground_truth_pos'][3:6, :80].cpu().numpy()
    time_points = pinn_data['time'][:80].cpu().numpy()
    
    pred_pos_a = predictions['pos_alpha'].cpu().numpy()
    pred_pos_b = predictions['pos_beta'].cpu().numpy()
    pred_vel_a = predictions['vel_alpha'].cpu().numpy()
    pred_vel_b = predictions['vel_beta'].cpu().numpy()
    
    # Quaternions (Robust Slicing logic)
    if isinstance(collision_frame, torch.Tensor): t_start = int(collision_frame.item())
    else: t_start = int(collision_frame)
    t_end = t_start + true_pos_a.shape[1]
    
    def robust_slice(arr, start, end):
        if isinstance(arr, torch.Tensor): arr = arr.cpu().numpy()
        # Handle (T, 4) vs (4, T)
        if arr.shape[0] > arr.shape[1]: return arr[start:end].T
        return arr[:, start:end]

    true_quat_a = robust_slice(data['quat_alpha'], t_start, t_end)
    true_quat_b = robust_slice(data['quat_beta'], t_start, t_end)
    
    # Check Predictions for orientation
    has_orientation = False
    if 'quat_alpha' in predictions:
        pred_quat_a = predictions['quat_alpha'].cpu().numpy()
        pred_quat_b = predictions['quat_beta'].cpu().numpy()
        if pred_quat_a.shape[0] != 4: 
            pred_quat_a = pred_quat_a.T
            pred_quat_b = pred_quat_b.T
        has_orientation = True
    else:
        # Dummy quats if missing
        pred_quat_a = np.zeros_like(true_quat_a); pred_quat_a[0,:] = 1.0
        pred_quat_b = np.zeros_like(true_quat_b); pred_quat_b[0,:] = 1.0
        print("⚠ Prediction lacks orientation data.")

    # --- ERROR CALCULATION ---
    pos_err_a = np.linalg.norm(pred_pos_a - true_pos_a, axis=0)
    pos_err_b = np.linalg.norm(pred_pos_b - true_pos_b, axis=0)
    
    rot_err_a = np.zeros_like(pos_err_a)
    rot_err_b = np.zeros_like(pos_err_b)
    if has_orientation:
        rot_err_a = compute_orientation_error(true_quat_a, pred_quat_a)
        rot_err_b = compute_orientation_error(true_quat_b, pred_quat_b)

    # --- PLOTTING ---
    os.makedirs('plots', exist_ok=True)
    print("Generating plots...")
    
    # 1. Trajectories with Poses
    print("- 3D Trajectories with Poses...")
    fig = plot_trajectories_with_poses(
        true_pos_a, true_quat_a, pred_pos_a, pred_quat_a,
        true_pos_b, true_quat_b, pred_pos_b, pred_quat_b
    )
    plt.savefig('plots/1_trajectories_poses.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Position Errors
    print("- Position Errors...")
    fig = plot_position_errors(pos_err_a, pos_err_b, time_points)
    plt.savefig('plots/2_position_errors.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Orientation Errors
    if has_orientation:
        print("- Orientation Errors...")
        fig = plot_orientation_errors(rot_err_a, rot_err_b, time_points)
        plt.savefig('plots/3_orientation_errors.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    # 4. Momentum
    print("- Momentum...")
    fig = plot_momentum_conservation(pred_vel_a, pred_vel_b, time_points)
    plt.savefig('plots/4_momentum.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Geometry
    print("- Collision Geometry...")
    geometry = compute_collision_geometry(collision_state)
    fig = plot_collision_geometry_3d(collision_state, geometry)
    plt.savefig('plots/5_collision_geometry.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Influence
    print("- Orientation Influence...")
    fig, _ = plot_orientation_influence(collision_state, predictions)
    plt.savefig('plots/6_orientation_influence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nDone! All plots saved to /plots directory.")
    print(f"Mean Pos Error: {(pos_err_a.mean()+pos_err_b.mean())/2*1000:.2f}mm")
    if has_orientation:
        print(f"Mean Rot Error: {(rot_err_a.mean()+rot_err_b.mean())/2:.2f} deg")

if __name__ == "__main__":
    main()