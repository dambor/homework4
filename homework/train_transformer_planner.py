"""
Training script for the TransformerPlanner model.

Usage:
    python3 -m homework.train_transformer_planner.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import TransformerPlanner, save_model

def train_transformer_planner():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = TransformerPlanner(
        n_track=10,
        n_waypoints=3,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    model.to(device)
    
    # Create dataloaders
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="state_only",  # Only track_left and track_right
        batch_size=64,
        num_workers=2,
        shuffle=True
    )
    
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="state_only",
        batch_size=64,
        num_workers=2,
        shuffle=False
    )
    
    # Define loss function
    criterion = nn.L1Loss(reduction='none')  # Use L1 loss for waypoint regression
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize metric
    metric = PlannerMetric()

    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints/transformer_planner')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(epoch):
        model.train()
        total_loss = 0.0
        metric.reset()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            output = model(batch["track_left"], batch["track_right"])
            
            # Calculate loss
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]
            
            # Apply mask to compute loss only on valid waypoints
            loss_per_element = criterion(output, waypoints)  # (B, n_waypoints, 2)
            loss_masked = loss_per_element * waypoints_mask.unsqueeze(-1)  # Apply mask
            loss = loss_masked.sum() / (waypoints_mask.sum() * 2 + 1e-6)  # Normalize by number of valid waypoints * 2 (x,y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update metrics
            metric.add(output.detach(), waypoints, waypoints_mask)
            total_loss += loss.item()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(batch['waypoints'])}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        # Compute and log epoch metrics
        metrics = metric.compute()
        
        # Log to console
        print(f"Train Epoch: {epoch}, Loss: {total_loss / len(train_loader):.6f}, "
              f"L1 Error: {metrics['l1_error']:.6f}, "
              f"Longitudinal Error: {metrics['longitudinal_error']:.6f}, "
              f"Lateral Error: {metrics['lateral_error']:.6f}")
        
        return metrics

    def validate(epoch):
        model.eval()
        val_loss = 0.0
        metric.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                output = model(batch["track_left"], batch["track_right"])
                
                # Calculate loss
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]
                
                loss_per_element = criterion(output, waypoints)
                loss_masked = loss_per_element * waypoints_mask.unsqueeze(-1)
                loss = loss_masked.sum() / (waypoints_mask.sum() * 2 + 1e-6)
                
                # Update metrics
                metric.add(output, waypoints, waypoints_mask)
                val_loss += loss.item()
        
        # Compute and log metrics
        metrics = metric.compute()
        avg_loss = val_loss / len(val_loader)
        
        # Log to console
        print(f"Validation Epoch: {epoch}, Loss: {avg_loss:.6f}, "
              f"L1 Error: {metrics['l1_error']:.6f}, "
              f"Longitudinal Error: {metrics['longitudinal_error']:.6f}, "
              f"Lateral Error: {metrics['lateral_error']:.6f}")
        
        return avg_loss, metrics

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Train for 40 epochs (more than MLP because transformer might take longer to converge)
    for epoch in range(1, 41):
        # Train
        train_metrics = train_epoch(epoch)
        
        # Validate
        val_loss, val_metrics = validate(epoch)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save model
            save_path = save_model(model)
            print(f"Saved best model at epoch {epoch} to {save_path}")

            # Also save a checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Early stopping (stop if no improvement for 10 epochs)
        if epoch - best_epoch > 10:
            print(f"Early stopping at epoch {epoch} since no improvement for 10 epochs")
            break
    
    print(f"Best model saved at epoch {best_epoch} with validation loss {best_val_loss:.6f}")
    
    # Final metrics on best model
    print(f"Best validation metrics:")
    print(f"L1 Error: {val_metrics['l1_error']:.6f}")
    print(f"Longitudinal Error: {val_metrics['longitudinal_error']:.6f}")
    print(f"Lateral Error: {val_metrics['lateral_error']:.6f}")


if __name__ == "__main__":
    train_transformer_planner()