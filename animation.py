import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils_pinn import *
from scipy.spatial.transform import Rotation as R
import sys
import os

# --- CONFIGURATION ---
SAVE_GIF = True
FPS = 15
CUBE_SIZE = 1.0  # 1 meter
TRAIL_LENGTH = 80
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "run_0002"

def get_cube_edges(center, quat, size):
    """Calculate the 3D coordinates of a cube's edges."""
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s],  [s, -s, s],  [s, s, s],  [-s, s, s]
    ])
    
    # Rotate [w, x, y, z] -> [x, y, z, w] for Scipy
    if np.linalg.norm(quat) > 1e-6:
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        vertices = r.apply(vertices)
    
    vertices = vertices + center
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4], # Top
        [0, 4], [1, 5], [2, 6], [3, 7]  # Sides
    ]
    return vertices, edges

def update_graph(num, data_dict, lines, ax):
    """Update function for the animation loop."""
    
    if num % 10 == 0:
        print(f"Rendering frame {num}/{data_dict['total_frames']}...", end='\r')

    # Unpack data
    t_pos_alpha = data_dict['true_pos_alpha']
    t_pos_beta  = data_dict['true_pos_beta']
    p_pos_alpha = data_dict['pred_pos_alpha']
    p_pos_beta  = data_dict['pred_pos_beta']
    
    # Get Quaternions (Expected shape: 4, T)
    t_quat_alpha = data_dict.get('true_quat_alpha')
    p_quat_alpha = data_dict.get('pred_quat_alpha')
    t_quat_beta  = data_dict.get('true_quat_beta')
    p_quat_beta  = data_dict.get('pred_quat_beta')

    ax.set_title(f"Collision Prediction: {RUN_ID}\nFrame: {num}")

    # --- HELPER: UPDATE CUBE WIREFRAME ---
    def update_cube(pos_array, quat_array, line_obj):
        # Position Index
        idx = min(num, pos_array.shape[1]-1)
        curr_pos = pos_array[:, idx]
        
        # Quaternion Index
        curr_quat = [1, 0, 0, 0] # Default
        if quat_array is not None and quat_array.shape[1] > 0:
            q_idx = min(num, quat_array.shape[1]-1)
            curr_quat = quat_array[:, q_idx]

        verts, edges = get_cube_edges(curr_pos, curr_quat, CUBE_SIZE)
        
        x_lines, y_lines, z_lines = [], [], []
        for edge in edges:
            x_lines.extend([verts[edge[0], 0], verts[edge[1], 0], np.nan])
            y_lines.extend([verts[edge[0], 1], verts[edge[1], 1], np.nan])
            z_lines.extend([verts[edge[0], 2], verts[edge[1], 2], np.nan])
            
        line_obj.set_data(np.array(x_lines), np.array(y_lines))
        line_obj.set_3d_properties(np.array(z_lines))

    # --- HELPER: UPDATE TRAIL ---
    def update_trail(pos_array, line_obj):
        start = max(0, num - TRAIL_LENGTH)
        end = min(num + 1, pos_array.shape[1])
        
        if end - start > 1:
            segment = pos_array[:, start:end]
            line_obj.set_data(np.array(segment[0]), np.array(segment[1]))
            line_obj.set_3d_properties(np.array(segment[2]))
        else:
            line_obj.set_data([], [])
            line_obj.set_3d_properties([])

    # Update Cubes
    update_cube(t_pos_alpha, t_quat_alpha, lines['gt_cube_a'])
    update_cube(t_pos_beta, t_quat_beta, lines['gt_cube_b'])
    update_cube(p_pos_alpha, p_quat_alpha, lines['pred_cube_a'])
    update_cube(p_pos_beta, p_quat_beta, lines['pred_cube_b'])

    # Update Trails
    update_trail(t_pos_alpha, lines['gt_trail_a'])
    update_trail(t_pos_beta, lines['gt_trail_b'])
    update_trail(p_pos_alpha, lines['trail_a'])
    update_trail(p_pos_beta, lines['trail_b'])

    return list(lines.values())

def main():
    print(f"Initializing Animation for {RUN_ID}...")
    
    # --- DATA LOADING ---
    pred_path = 'saved_models/predictions_001.pt'
    try:
        predictions = torch.load(pred_path, map_location='cpu', weights_only=False)
    except:
        predictions = torch.load(pred_path, map_location='cpu')
    
    dataset_path = f"Datasets/{RUN_ID}"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    data = load_collision_data(dataset_path)
    collision_frame = detect_collision_frame(data['pos_alpha'], data['pos_beta'])
    
    # Setup PINN Data
    initial_frame = collision_frame - 1
    collision_state = get_collision_state(data, initial_frame)
    trajectory = get_post_collision_trajectory(data, collision_frame, start_offset=0)
    pinn_data = prepare_pinn_data(collision_state, trajectory)
    
    # 1. EXTRACT POSITIONS
    gt_tensor = pinn_data['ground_truth_pos'].cpu().numpy()
    true_pos_alpha = gt_tensor[:3, :80]
    true_pos_beta  = gt_tensor[3:6, :80]
    pred_pos_alpha = predictions['pos_alpha'].cpu().numpy()
    pred_pos_beta  = predictions['pos_beta'].cpu().numpy()
    
    # 2. EXTRACT QUATERNIONS (ROBUST METHOD)
    # Define the robust slicing helper
    if isinstance(collision_frame, torch.Tensor): t_start = int(collision_frame.item())
    else: t_start = int(collision_frame)
    t_end = t_start + true_pos_alpha.shape[1]

    def robust_slice(arr, start, end):
        if isinstance(arr, torch.Tensor): arr = arr.cpu().numpy()
        # Case: (4, Time) -> Return slice of 2nd dim
        if arr.shape[0] == 4 and arr.shape[1] > 4: 
            return arr[:, start:end]
        # Case: (Time, 4) -> Slice 1st dim and transpose to (4, Time)
        elif arr.shape[1] == 4 and arr.shape[0] > 4:
            return arr[start:end].T
        return arr # Fallback

    # Apply robust slicing to Ground Truth
    true_quat_alpha = robust_slice(data['quat_alpha'], t_start, t_end)
    true_quat_beta  = robust_slice(data['quat_beta'], t_start, t_end)
    
    # Handle Predictions
    pred_quat_alpha = None
    pred_quat_beta = None
    
    if 'quat_alpha' in predictions:
        pq_a = predictions['quat_alpha'].cpu().numpy()
        pq_b = predictions['quat_beta'].cpu().numpy()
        # Ensure (4, T) format
        if pq_a.shape[0] != 4: pq_a = pq_a.T
        if pq_b.shape[0] != 4: pq_b = pq_b.T
        pred_quat_alpha = pq_a
        pred_quat_beta = pq_b
    else:
        # Fill with Identity if missing to prevent crashes
        print("Warning: Prediction has no quaternion data.")
        pred_quat_alpha = np.zeros_like(true_quat_alpha); pred_quat_alpha[0,:] = 1.0
        pred_quat_beta = np.zeros_like(true_quat_beta); pred_quat_beta[0,:] = 1.0

    num_frames = min(true_pos_alpha.shape[1], pred_pos_alpha.shape[1])
    
    # Pack data
    anim_data = {
        'true_pos_alpha': true_pos_alpha,
        'true_pos_beta': true_pos_beta,
        'pred_pos_alpha': pred_pos_alpha,
        'pred_pos_beta': pred_pos_beta,
        'true_quat_alpha': true_quat_alpha,
        'true_quat_beta': true_quat_beta,
        'pred_quat_alpha': pred_quat_alpha,
        'pred_quat_beta': pred_quat_beta,
        'total_frames': num_frames
    }
    
    # --- PLOTTING SETUP ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Bounds
    all_x = np.concatenate([true_pos_alpha[0], true_pos_beta[0]])
    all_y = np.concatenate([true_pos_alpha[1], true_pos_beta[1]])
    all_z = np.concatenate([true_pos_alpha[2], true_pos_beta[2]])
    
    margin = 1.5
    ax.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax.set_ylim(all_y.min()-margin, all_y.max()+margin)
    ax.set_zlim(all_z.min()-margin, all_z.max()+margin)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=2, label='Pred Alpha'),
        Line2D([0], [0], color='orange', lw=2, label='Pred Beta'),
        Line2D([0], [0], color='blue', lw=1, linestyle='--', label='GT Alpha'),
        Line2D([0], [0], color='red', lw=1, linestyle='--', label='GT Beta'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Initialize Line Objects
    lines = {
        'gt_cube_a': ax.plot([], [], [], color='blue', linestyle='--', lw=1, alpha=0.3)[0],
        'gt_cube_b': ax.plot([], [], [], color='red', linestyle='--', lw=1, alpha=0.3)[0],
        'pred_cube_a': ax.plot([], [], [], color='cyan', linestyle='-', lw=2, alpha=1.0)[0],
        'pred_cube_b': ax.plot([], [], [], color='orange', linestyle='-', lw=2, alpha=1.0)[0],
        'gt_trail_a': ax.plot([], [], [], color='blue', linestyle=':', lw=1.5, alpha=0.4)[0],
        'gt_trail_b': ax.plot([], [], [], color='red', linestyle=':', lw=1.5, alpha=0.4)[0],
        'trail_a': ax.plot([], [], [], color='cyan', linestyle=':', lw=2, alpha=0.8)[0],
        'trail_b': ax.plot([], [], [], color='orange', linestyle=':', lw=2, alpha=0.8)[0],
    }
    
    print(f"Generating animation with {num_frames} frames...")
    
    anim = animation.FuncAnimation(
        fig, update_graph, frames=num_frames, 
        fargs=(anim_data, lines, ax), interval=1000/FPS, blit=False
    )
    
    os.makedirs('animations', exist_ok=True)
    save_path = f"animations/{RUN_ID}_trajectory.gif"
    
    print(f"Saving to {save_path}...")
    try:
        anim.save(save_path, writer='pillow', fps=FPS)
        print("\nDone! Animation saved.")
    except Exception as e:
        print(f"\nError saving animation: {e}")

if __name__ == "__main__":
    main()