# train_pinn.py

import torch
import torch.optim as optim
from pinn import *
from utils_pinn import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dataset_path = "Datasets/run_0002"
    
    # Training parameters
    num_epochs = 1000
    learning_rate = 1e-4
    prediction_horizon = 80  # frames (2 seconds)
    
    # Loss weights
    lambda_data_pos = 1.0
    lambda_data_vel = 1.0
    lambda_data_quat = 0.5
    lambda_ang_mom = 0.1
    lambda_momentum = 1.0
    lambda_kinematics = 0.5
    lambda_velocity = 1.0
    stabilization_timestep = 12
    
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load and preprocess data
    data = load_collision_data(dataset_path)
    collision_frame = detect_collision_frame(data['pos_alpha'], data['pos_beta'])
    initial_frame = collision_frame - 1
    
    collision_state = get_collision_state(data, initial_frame)
    trajectory = get_post_collision_trajectory(data, collision_frame, start_offset=0)
    
    pinn_data = prepare_pinn_data(collision_state, trajectory)
    
    # Limit to prediction horizon
    initial_state = pinn_data['initial_state'].float()
    time_points = pinn_data['time'][:prediction_horizon].float()
    true_pos = pinn_data['ground_truth_pos'][:, :prediction_horizon].float()
    true_vel = pinn_data['ground_truth_vel'][:, :prediction_horizon].float()
    true_quat = pinn_data['ground_truth_quat'][:, :prediction_horizon].float()
    
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Training horizon: {prediction_horizon} frames ({time_points[-1]:.2f}s)")
    print(f"Ground truth positions: {true_pos.shape}")
    print(f"Ground truth velocities: {true_vel.shape}")
    
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    # Create model
    model = pinn_model(
        input_dim=21, 
        hidden_dims=[128, 256, 256, 128],
        output_dim=21
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model architecture:\n{model}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    # Training history
    history = {
        'total': [],
        'data_pos': [],
        'data_vel': [],
        'momentum': [],
        'kinematics': [],
        'velocity': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        losses = train_step(
            model, optimizer, initial_state, time_points, true_pos, true_vel, true_quat,
            lambda_data_pos, lambda_data_vel, lambda_data_quat, lambda_ang_mom,lambda_momentum, 
            lambda_kinematics, lambda_velocity, stabilization_timestep, device
        )
        
        # Log losses
        for key in history.keys():
            history[key].append(losses[key])
        
        # Print progress
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Total Loss: {losses['total']:.6f}")
            print(f"  Data (pos): {losses['data_pos']:.6f}, Data (vel): {losses['data_vel']:.6f}")
            print(f"  Momentum: {losses['momentum']:.6f}, Kinematics: {losses['kinematics']:.6f}")
            print(f"  Velocity: {losses['velocity']:.6f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Save model
    torch.save(model.state_dict(), 'saved_models/pinn_model.pth')
    print("Model saved to: pinn_model.pth")
    
    # Plot training history
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (key, values) in enumerate(history.items()):
        axes[idx].plot(values)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_title(f'{key.capitalize()} Loss')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png', dpi=150)
    print("Training history saved to: plots/training_history_001.png")
    
    # Final validation
    print("\n" + "="*60)
    print("FINAL VALIDATION")
    print("="*60)
    
    val_losses, predictions = validate(
        model, initial_state, time_points, true_pos, true_vel, true_quat,
        lambda_data_pos, lambda_data_vel, lambda_data_quat, lambda_ang_mom, lambda_momentum, 
        lambda_kinematics, lambda_velocity, stabilization_timestep, device
    )
    
    print("Validation Losses:")
    for key, value in val_losses.items():
        print(f"  {key}: {value:.6f}")
    
    # Save predictions
    torch.save(predictions, 'saved_models/predictions_001.pt')
    print("\nPredictions saved to: saved_models/predictions_001.pt")

if __name__ == "__main__":
    main()