# pinn_model.py

import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

def pinn_model(input_dim=21, hidden_dims=[128, 256, 256, 128], output_dim=21):

    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(nn.Tanh())
    
    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        layers.append(nn.Tanh())
    
    # Output layer
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    
    model = nn.Sequential(*layers)
    
    return model

def data_loss_position(pred_pos, true_pos):
    return torch.mean((pred_pos - true_pos) ** 2)


def data_loss_velocity(pred_vel, true_vel):
    return torch.mean((pred_vel - true_vel) ** 2)


def momentum_conservation_loss(pred_vel, mass_alpha=10.0, mass_beta=10.0):

    vel_alpha = pred_vel[0:3, :]  # (3, T)
    vel_beta = pred_vel[3:6, :]   # (3, T)
    
    # Total momentum at each timestep
    momentum_alpha = mass_alpha * vel_alpha  # (3, T)
    momentum_beta = mass_beta * vel_beta     # (3, T)
    momentum_total = momentum_alpha + momentum_beta  # (3, T)
    
    # Initial momentum (should be [0, 0, 0])
    momentum_initial = momentum_total[:, 0:1]  # (3, 1)
    
    # Loss: deviation from initial momentum
    loss = torch.mean((momentum_total - momentum_initial) ** 2)
    
    return loss


def kinematic_consistency_loss(pred_pos, pred_vel, dt=1.0/30.0):
   
    # velocity from position derivative (finite difference)
    # v(t) ≈ [p(t+1) - p(t)] / dt
    pos_derivative = (pred_pos[:, 1:] - pred_pos[:, :-1]) / dt  # (6, T-1)
    
    # Comparing with predicted velocities (exclude last timestep)
    vel_predicted = pred_vel[:, :-1]  # (6, T-1)
    
    loss = torch.mean((vel_predicted - pos_derivative) ** 2)
    
    return loss


def constant_velocity_loss(pred_vel, stabilization_timestep=12):
    
    if pred_vel.shape[1] <= stabilization_timestep:
        return torch.tensor(0.0, device=pred_vel.device)
    
    # Velocities after stabilization
    vel_stable = pred_vel[:, stabilization_timestep:]  # (6, T-stabilization_timestep)
    
    # Reference velocity (first stable velocity)
    vel_reference = vel_stable[:, 0:1]  # (6, 1)
    
    # All stable velocities should match reference
    loss = torch.mean((vel_stable - vel_reference) ** 2)
    
    return loss


def data_loss_orientation(pred_quat, true_quat):
    """
    Computes loss between predicted and ground truth quaternions.
    
    Why not simple MSE?
    Quaternions q and -q represent the exact same rotation (double cover).
    MSE would penalize the network if it predicts -q when ground truth is q.
    Instead, we use: Loss = 1 - |<q1, q2>|^2 or 1 - |<q1, q2>|
    
    Args:
        pred_quat: (4, T) tensor
        true_quat: (4, T) tensor
    """
    # 1. Normalize predictions to ensure valid quaternions
    pred_norm = F.normalize(pred_quat, p=2, dim=0)
    true_norm = F.normalize(true_quat, p=2, dim=0)
    
    # 2. Compute dot product along the channel dimension (dim=0)
    # shape: (T,)
    dot_prod = torch.sum(pred_norm * true_norm, dim=0)
    
    # 3. Loss = 1 - absolute dot product (Cosine Similarity)
    # If aligned, dot = 1 or -1 -> abs = 1 -> loss = 0
    loss = 1.0 - torch.mean(torch.abs(dot_prod))
    
    return loss

def angular_momentum_loss(pred_pos_alpha, pred_vel_alpha, pred_quat_alpha,
                          pred_pos_beta, pred_vel_beta, pred_quat_beta,
                          mass=10.0, side_length=0.5, dt=1.0/30.0):
    """
    Enforces Conservation of Total Angular Momentum (Orbital + Spin).
    Crucial for grazing/eccentric collisions.
    
    L_total = L_orbital + L_spin
    L_orbital = r x (mv)
    L_spin = I * omega
    """
    
    # --- 1. Calculate Orbital Angular Momentum (L = r x p) ---
    # Momentum vectors
    p_alpha = mass * pred_vel_alpha
    p_beta = mass * pred_vel_beta
    
    # Cross product: r x p
    # Note: We assume origin (0,0,0) is the reference, or CoM of system.
    # Since conservation applies to the system, origin choice cancels out if no external torque.
    L_orb_alpha = torch.linalg.cross(pred_pos_alpha, p_alpha, dim=0) # (3, T)
    L_orb_beta = torch.linalg.cross(pred_pos_beta, p_beta, dim=0)    # (3, T)
    
    L_orbital_total = L_orb_alpha + L_orb_beta
    
    # --- 2. Calculate Spin Angular Momentum (L = I * omega) ---
    
    # Moment of Inertia for a Cube: I = 1/6 * m * s^2
    # Since a cube's inertia tensor is isotropic (spherical), I is a scalar.
    # We don't need to rotate the tensor! I_world = I_body.
    inertia = (1.0/6.0) * mass * (side_length ** 2)
    
    # Helper to calculate angular velocity from quaternions
    def get_omega(q):
        # q shape: (4, T) -> [w, x, y, z]
        # Finite difference derivative: q_dot approx (q[t] - q[t-1]) / dt
        # We lose the first frame
        q_curr = q[:, 1:]
        q_prev = q[:, :-1]
        q_dot = (q_curr - q_prev) / dt
        
        # Formula: omega_world = 2 * q_dot * q_conjugate
        # We implement simplified quaternion mult for vector part
        # q = [w, x, y, z]
        # q_conj = [w, -x, -y, -z]
        
        qc = q_curr # Approximation: use current frame for q
        w, x, y, z = qc[0], qc[1], qc[2], qc[3]
        wd, xd, yd, zd = q_dot[0], q_dot[1], q_dot[2], q_dot[3]
        
        # Quaternion multiplication (q_dot * q_conj) vector part:
        # We only need the x, y, z components of the result
        ox =  x*wd + w*xd + z*yd - y*zd # Actually this is q * q_dot?
        # Let's use the explicit relationship: 2 * q_dot * q_inv
        # q_inv = [w, -x, -y, -z] (since unit quat)
        
        # Result of (wd, xd, yd, zd) * (w, -x, -y, -z)
        # Vector parts (i, j, k):
        omega_x = 2.0 * (wd * (-x) + xd * w + yd * (-z) - zd * (-y))
        omega_y = 2.0 * (wd * (-y) - xd * (-z) + yd * w + zd * (-x))
        omega_z = 2.0 * (wd * (-z) + xd * (-y) - yd * (-x) + zd * w)
        
        return torch.stack([omega_x, omega_y, omega_z], dim=0)

    omega_alpha = get_omega(pred_quat_alpha) # (3, T-1)
    omega_beta = get_omega(pred_quat_beta)   # (3, T-1)
    
    # Spin momentum
    L_spin_alpha = inertia * omega_alpha
    L_spin_beta = inertia * omega_beta
    
    L_spin_total = L_spin_alpha + L_spin_beta
    
    # --- 3. Total Angular Momentum ---
    # Truncate orbital to match spin length (T-1)
    L_orbital_total = L_orbital_total[:, 1:]
    
    L_total = L_orbital_total + L_spin_total # (3, T-1)
    
    # --- 4. Conservation Loss ---
    # We want L_total to be constant over time.
    # We calculate the deviation from the mean vector.
    L_mean = torch.mean(L_total, dim=1, keepdim=True)
    L_deviation = L_total - L_mean
    
    # MSE of the deviation
    loss = torch.mean(L_deviation ** 2)
    
    return loss

def total_loss(pred_pos, pred_vel, pred_quat, true_pos, true_vel, true_quat, 
               lambda_data_pos=1.0, lambda_data_vel=0.5, 
               lambda_data_quat=0.5, lambda_ang_mom=0.1,
               lambda_momentum=0.1, lambda_kinematics=0.1, 
               lambda_velocity=0.1, stabilization_timestep=12):
    """
    Computes total weighted loss for the collision PINN, including linear and rotational physics.
    
    Args:
        pred_pos: (6, T) Predicted positions [alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z]
        pred_vel: (6, T) Predicted velocities
        pred_quat: (8, T) Predicted quaternions [a_w, a_x, a_y, a_z, b_w, b_x, b_y, b_z]
        true_pos: (6, T) Ground truth positions
        true_vel: (6, T) Ground truth velocities
        true_quat: (8, T) Ground truth quaternions
    """
    
    # --- 1. Slice Inputs for Specific Calculations ---
    # Positions (3, T)
    pos_alpha = pred_pos[:3, :]
    pos_beta = pred_pos[3:, :]
    
    # Velocities (3, T)
    vel_alpha = pred_vel[:3, :]
    vel_beta = pred_vel[3:, :]
    
    # Quaternions (4, T)
    # Assumes shape [8, T] where 0-3 is Alpha, 4-7 is Beta
    quat_alpha = pred_quat[:4, :]
    quat_beta = pred_quat[4:, :]
    
    true_quat_alpha = true_quat[:4, :]
    true_quat_beta = true_quat[4:, :]

    # --- 2. Calculate Individual Losses ---
    
    # Standard Linear Losses
    loss_data_pos = data_loss_position(pred_pos, true_pos)
    loss_data_vel = data_loss_velocity(pred_vel, true_vel)
    loss_momentum = momentum_conservation_loss(pred_vel)
    loss_kinematics = kinematic_consistency_loss(pred_pos, pred_vel)
    loss_velocity = constant_velocity_loss(pred_vel, stabilization_timestep)
    
    # Rotational / Orientation Losses
    loss_quat_alpha = data_loss_orientation(quat_alpha, true_quat_alpha)
    loss_quat_beta = data_loss_orientation(quat_beta, true_quat_beta)
    loss_quat_total = loss_quat_alpha + loss_quat_beta

    # Angular Momentum Loss (The "Torque" Physics)
    loss_ang_mom = angular_momentum_loss(
        pos_alpha, vel_alpha, quat_alpha,
        pos_beta, vel_beta, quat_beta
    )

    # --- 3. Weighted Sum ---
    total_loss_val = (lambda_data_pos * loss_data_pos + 
                      lambda_data_vel * loss_data_vel +
                      lambda_momentum * loss_momentum +
                      lambda_kinematics * loss_kinematics +
                      lambda_velocity * loss_velocity +
                      lambda_data_quat * loss_quat_total + 
                      lambda_ang_mom * loss_ang_mom)
    
    # --- 4. Return for Logging ---
    losses = {
        'total': total_loss_val.item(),
        'data_pos': loss_data_pos.item(),
        'data_vel': loss_data_vel.item(),
        'momentum': loss_momentum.item(),
        'kinematics': loss_kinematics.item(),
        'velocity': loss_velocity.item(),
        'data_quat': loss_quat_total.item(),
        'ang_mom': loss_ang_mom.item()
    }
    
    return total_loss_val, losses

def train_step(model, optimizer, initial_state, time_points, true_pos, true_vel, true_quat,
               lambda_data_pos=1.0, lambda_data_vel=0.5, 
               lambda_data_quat=0.5, lambda_ang_mom=0.1,
               lambda_momentum=0.1, lambda_kinematics=0.1, 
               lambda_velocity=0.1, stabilization_timestep=12, device='cuda'):
    """
    Performs one training step for the PINN, now including orientation/rotation.
    """
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    initial_state = initial_state.to(device)
    time_points = time_points.to(device)
    true_pos = true_pos.to(device)
    true_vel = true_vel.to(device)
    true_quat = true_quat.to(device)
    
    # Forward pass
    num_timesteps = len(time_points)
    initial_state_repeated = initial_state.unsqueeze(0).repeat(num_timesteps, 1)
    time_input = time_points.unsqueeze(1)
    
    # Input: [Initial State (20), Time (1)]
    network_input = torch.cat([initial_state_repeated, time_input], dim=1)
    
    output = model(network_input)  # Expected shape: (T, 21)
    
    # --- Extract Predictions (Assuming 20-dim state vector) ---
    # Structure: [Pos_A(3), Quat_A(4), Vel_A(3), Pos_B(3), Quat_B(4), Vel_B(3)]
    
    # Positions: Indices 0-3 (Alpha) and 10-13 (Beta)
    pred_pos = torch.cat([output[:, 0:3].T, output[:, 10:13].T], dim=0)  # (6, T)
    
    # Quaternions: Indices 3-7 (Alpha) and 13-17 (Beta)
    pred_quat = torch.cat([output[:, 3:7].T, output[:, 13:17].T], dim=0) # (8, T)
    
    # Velocities: Indices 7-10 (Alpha) and 17-20 (Beta)
    pred_vel = torch.cat([output[:, 7:10].T, output[:, 17:20].T], dim=0) # (6, T)
    
    # Compute loss (using the imported total_loss from pinn_losses.py)
    total_loss_, losses = total_loss(
        pred_pos, pred_vel, pred_quat, 
        true_pos, true_vel, true_quat,
        lambda_data_pos, lambda_data_vel, 
        lambda_data_quat, lambda_ang_mom,
        lambda_momentum, lambda_kinematics, 
        lambda_velocity, stabilization_timestep
    )
    
    # Backward pass
    total_loss_.backward()
    optimizer.step()
    
    return losses


def validate(model, initial_state, time_points, true_pos, true_vel, true_quat,
             lambda_data_pos=1.0, lambda_data_vel=0.5, 
             lambda_data_quat=0.5, lambda_ang_mom=0.1,
             lambda_momentum=0.1, lambda_kinematics=0.1, 
             lambda_velocity=0.1, stabilization_timestep=12, device='cuda'):
    """
    Validation step. Assumes predict_trajectory now returns quaternions as well.
    """
    model.eval()
    
    with torch.no_grad():
        # Move data to device
        initial_state = initial_state.to(device)
        time_points = time_points.to(device)
        true_pos = true_pos.to(device)
        true_vel = true_vel.to(device)
        true_quat = true_quat.to(device)
        
        # Forward pass
        num_timesteps = len(time_points)
        initial_state_repeated = initial_state.unsqueeze(0).repeat(num_timesteps, 1)
        time_input = time_points.unsqueeze(1)
        network_input = torch.cat([initial_state_repeated, time_input], dim=1)
        
        output = model(network_input)
        
        # Extract Predictions
        pred_pos = torch.cat([output[:, 0:3].T, output[:, 10:13].T], dim=0)
        pred_quat = torch.cat([output[:, 3:7].T, output[:, 13:17].T], dim=0)
        pred_vel = torch.cat([output[:, 7:10].T, output[:, 17:20].T], dim=0)
        
        # Compute loss
        total_loss_, losses = total_loss(
            pred_pos, pred_vel, pred_quat, 
            true_pos, true_vel, true_quat,
            lambda_data_pos, lambda_data_vel, 
            lambda_data_quat, lambda_ang_mom,
            lambda_momentum, lambda_kinematics, 
            lambda_velocity, stabilization_timestep
        )
        
        # Package predictions for plotting
        predictions = {
            'pos_alpha': pred_pos[:3, :],
            'pos_beta': pred_pos[3:, :],
            'quat_alpha': pred_quat[:4, :],
            'quat_beta': pred_quat[4:, :],
            'vel_alpha': pred_vel[:3, :],
            'vel_beta': pred_vel[3:, :]
        }
    
    return losses, predictions