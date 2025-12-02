# utils_pinn.py

import torch
import numpy as np
import os

def load_collision_data(dataset_path):
    
    data = {
        'alpha_pose': torch.load(os.path.join(dataset_path, 'alpha_pose_data.pt'), weights_only=True),
        'beta_pose': torch.load(os.path.join(dataset_path, 'beta_pose_data.pt'), weights_only=True),
        'alpha_vel': torch.load(os.path.join(dataset_path, 'alpha_velocities.pt'), weights_only=True),
        'beta_vel': torch.load(os.path.join(dataset_path, 'beta_velocities.pt'), weights_only=True)
    }
    
    # Extract positions, quaternions and linear velocities
    data['pos_alpha'] = data['alpha_pose'][:3, 1:].cpu()  # (3, T)
    data['pos_beta'] = data['beta_pose'][:3, 1:].cpu()
    
    data['quat_alpha'] = data['alpha_pose'][3:7, 1:].cpu()  # (4, T) - [w, x, y, z]
    data['quat_beta'] = data['beta_pose'][3:7, 1:].cpu()    # (4, T)
    
    data['vel_alpha'] = data['alpha_vel'][:3, 1:].cpu() 
    data['vel_beta'] = data['beta_vel'][:3, 1:].cpu()
    
    return data

def detect_collision_frame(pos_alpha, pos_beta):
   
    distances = torch.norm(pos_alpha - pos_beta, dim=0)
    collision_frame = torch.argmin(distances).item()
    return collision_frame

def get_collision_state(data, collision_frame):
    
    state = {
        'pos_alpha': data['pos_alpha'][:, collision_frame],
        'pos_beta': data['pos_beta'][:, collision_frame],
        'vel_alpha': data['vel_alpha'][:, collision_frame],
        'vel_beta': data['vel_beta'][:, collision_frame],
        'quat_alpha': data['quat_alpha'][:, collision_frame],  # (4,)
        'quat_beta': data['quat_beta'][:, collision_frame],    # (4,)
        
        'frame': collision_frame
    }

    return state

def get_post_collision_trajectory(data, collision_frame, start_offset=1, end_frame=None):
    
    start_frame = collision_frame + start_offset
    if end_frame is None:
        end_frame = data['pos_alpha'].shape[1]
    
    num_frames = end_frame - start_frame
    dt = 1.0 / 30.0  # 30 fps
    
    trajectory = {
        'pos_alpha': data['pos_alpha'][:, start_frame:end_frame],  # (3, T)
        'pos_beta': data['pos_beta'][:, start_frame:end_frame],
        'vel_alpha': data['vel_alpha'][:, start_frame:end_frame],
        'vel_beta': data['vel_beta'][:, start_frame:end_frame],
        'quat_alpha': data['quat_alpha'][:, start_frame:end_frame],
        'quat_beta': data['quat_beta'][:, start_frame:end_frame],
        'time': torch.arange(0, num_frames * dt, dt),
        'num_frames': num_frames
    }
    return trajectory

def compute_total_momentum(vel_alpha, vel_beta, mass_alpha=10.0, mass_beta=10.0):
   
    p_alpha = mass_alpha * vel_alpha
    p_beta = mass_beta * vel_beta
    return p_alpha + p_beta

def verify_momentum_conservation(vel_alpha, vel_beta, mass_alpha=10.0, mass_beta=10.0):
    
    momentum = compute_total_momentum(vel_alpha, vel_beta, mass_alpha, mass_beta)
    
    momentum_initial = momentum[:, 0]
    momentum_variation = torch.std(momentum, dim=1)
    
    print("Momentum Conservation Check:")
    print(f"Initial momentum: {momentum_initial.numpy()}")
    print(f"Momentum std deviation: {momentum_variation.numpy()}")
    print(f"Momentum conserved: {torch.allclose(momentum_variation, torch.zeros(3), atol=1e-4)}")
    
    return momentum

def verify_constant_velocity(vel_data, start_frame, name="object"):
   
    post_collision_vel = vel_data[:, start_frame:]
    vel_std = torch.std(post_collision_vel, dim=1)
    vel_mean = torch.mean(post_collision_vel, dim=1)
    
    print(f"\n{name} post-collision velocity:")
    print(f"  Mean: {vel_mean.numpy()}")
    print(f"  Std: {vel_std.numpy()}")
    print(f"  Constant: {torch.allclose(vel_std, torch.zeros(3), atol=1e-3)}")

def prepare_pinn_data(collision_state, trajectory):
    
    initial_state = torch.cat([
        collision_state['pos_alpha'],      # (3,)
        collision_state['quat_alpha'],     # (4,)
        collision_state['vel_alpha'],      # (3,)
        collision_state['pos_beta'],       # (3,)
        collision_state['quat_beta'],      # (4,)
        collision_state['vel_beta']        # (3,)
    ])  # (20,) vector
    
    # Ground truth trajectories
    ground_truth_pos = torch.cat([
        trajectory['pos_alpha'],
        trajectory['pos_beta']
    ], dim=0)
    
    ground_truth_vel = torch.cat([
        trajectory['vel_alpha'],
        trajectory['vel_beta']
    ], dim=0)

    ground_truth_quat = torch.cat([
        trajectory['quat_alpha'],
        trajectory['quat_beta']
    ])
    
    pinn_data = {
        'initial_state': initial_state,      # (20,)
        'ground_truth_pos': ground_truth_pos,
        'ground_truth_vel': ground_truth_vel,
        'ground_truth_quat': ground_truth_quat,
        'time': trajectory['time'],
        'num_timesteps': trajectory['num_frames']
    }
    
    return pinn_data