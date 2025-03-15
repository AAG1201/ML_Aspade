import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import soundfile as sf
from time import time
from scipy.signal import resample
from spade_segmentation import spade_segmentation_eval
import pandas as pd
from clip_sdr_modified import clip_sdr_modified
from typing import Tuple, List, Optional, Dict
from sdr import sdr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from ML_model import ASPADEDataset,ASPADEModel
import sys

# def preprocess_data(training_data):
#     inputs = []
#     targets = []
    
#     for data_point in training_data:
#         initial_estimate = data_point[0]  # nx2000 complex initial estimate
#         best_estimate = data_point[1]    # nx2000 complex best estimate
#         best_sparsity = data_point[2]    # nx1 best sparsity
        
#         # Convert complex arrays to real representation
#         # For each complex number, we'll have its real and imaginary parts
#         inputs.append(np.hstack([initial_estimate.real, initial_estimate.imag]))
#         targets.append((np.hstack([best_estimate.real, best_estimate.imag]), best_sparsity))
    
#     inputs = np.array(inputs)
#     targets_estimates = np.array([t[0] for t in targets])
#     targets_sparsity = np.array([t[1] for t in targets])
    
#     return inputs, targets_estimates, targets_sparsity


# def train_and_save_model(training_data, model_path, num_epochs=50, lr=0.001, batch_size=32):
#     # Get the shape of your data to correctly set up the model
#     sample_initial = training_data[0][0]  # First sample's initial estimate
#     input_dim = 2 * sample_initial.shape[0]  # Double for real and imaginary parts
#     output_dim = input_dim  # Same dimension for output
    
#     print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
#     # Initialize model with correct dimensions
#     model = ASPADEModel(input_dim, output_dim)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     # Process the data
#     inputs, targets_estimates, targets_sparsity = preprocess_data(training_data)
#     dataset = ASPADEDataset(inputs, targets_estimates, targets_sparsity)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # Store loss values for plotting
#     epoch_losses = []
#     epoch_estimate_losses = []
#     epoch_sparsity_losses = []
    
#     # Training loop
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         running_estimate_loss = 0.0
#         running_sparsity_loss = 0.0
#         batch_count = 0
        
#         for batch_x, batch_y, batch_k in dataloader:
#             optimizer.zero_grad()
#             pred_output, pred_k = model(batch_x)
            
#             loss_estimate = nn.MSELoss()(pred_output, batch_y)
#             loss_sparsity = nn.MSELoss()(pred_k, batch_k)
#             loss = loss_estimate + 0.1 * loss_sparsity
            
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             running_estimate_loss += loss_estimate.item()
#             running_sparsity_loss += loss_sparsity.item()
#             batch_count += 1
            
#             # Overwrite the previous line instead of printing a new line every update
#             sys.stdout.write(
#                 f"\rEpoch {epoch+1}/{num_epochs} | "
#                 f"Loss: {running_loss/batch_count:.4f} | "
#                 f"Est: {running_estimate_loss/batch_count:.4f} | "
#                 f"Spar: {running_sparsity_loss/batch_count:.4f}"
#             )
#             sys.stdout.flush()
        
#         epoch_losses.append(running_loss / batch_count)
#         epoch_estimate_losses.append(running_estimate_loss / batch_count)
#         epoch_sparsity_losses.append(running_sparsity_loss / batch_count)
    
#     print("\nTraining complete.")
#     torch.save(model.state_dict(), model_path)
    
#     # Plot loss curves separately
#     fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
#     axes[0].plot(range(1, num_epochs+1), epoch_losses, color='b')
#     axes[0].set_title('Total Loss')
#     axes[0].set_ylabel('Loss')
#     axes[0].grid(True)

#     axes[1].plot(range(1, num_epochs+1), epoch_estimate_losses, color='g')
#     axes[1].set_title('Estimate Loss')
#     axes[1].set_ylabel('Loss')
#     axes[1].grid(True)

#     axes[2].plot(range(1, num_epochs+1), epoch_sparsity_losses, color='r')
#     axes[2].set_title('Sparsity Loss')
#     axes[2].set_xlabel('Epoch')
#     axes[2].set_ylabel('Loss')
#     axes[2].grid(True)

#     plt.tight_layout()
#     plt.show()
    
#     return model






def preprocess_data(training_data):
    inputs = []
    targets = []
    
    for data_point in training_data:
        initial_estimate = data_point[0]  # nx2000 complex initial estimate
        best_estimate = data_point[1]    # nx2000 complex best estimate
        best_sparsity = data_point[2]    # nx1 best sparsity
        
        # Convert complex arrays to real representation
        # For each complex number, we'll have its real and imaginary parts
        inputs.append(np.hstack([initial_estimate.real, initial_estimate.imag]))
        targets.append((np.hstack([best_estimate.real, best_estimate.imag]), best_sparsity))
    
    inputs = np.array(inputs)
    targets_estimates = np.array([t[0] for t in targets])
    targets_sparsity = np.array([t[1] for t in targets], dtype=np.int64)  # Ensure this is integer
    
    return inputs, targets_estimates, targets_sparsity

from tqdm import tqdm

def train_and_save_model(training_data, model_path, num_epochs=50, lr=0.001, batch_size=32, max_sparsity=1000):
    sample_initial = training_data[0][0]  
    input_dim = 2 * sample_initial.shape[0]  
    output_dim = input_dim  
    
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
    model = ASPADEModel(input_dim, output_dim, max_sparsity)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    estimate_criterion = nn.MSELoss()
    sparsity_criterion = nn.CrossEntropyLoss()  
    
    inputs, targets_estimates, targets_sparsity = preprocess_data(training_data)
    dataset = ASPADEDataset(inputs, targets_estimates, targets_sparsity)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    epoch_losses = []
    epoch_estimate_losses = []
    epoch_sparsity_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_estimate_loss = 0.0
        running_sparsity_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        for batch_x, batch_y, batch_k in dataloader:
            optimizer.zero_grad()
            pred_output, pred_k_logits = model(batch_x)
            
            loss_estimate = estimate_criterion(pred_output, batch_y)
            loss_sparsity = sparsity_criterion(pred_k_logits, batch_k)
            loss = loss_estimate + loss_sparsity
            
            loss.backward()
            optimizer.step()
            
            _, predicted_k = torch.max(pred_k_logits, 1)
            total_correct += (predicted_k == batch_k).sum().item()
            total_samples += batch_k.size(0)
            
            running_loss += loss.item()
            running_estimate_loss += loss_estimate.item()
            running_sparsity_loss += loss_sparsity.item()
            batch_count += 1
            
            # Overwriting progress update
            sparsity_accuracy = 100 * total_correct / total_samples
            sys.stdout.write(
                f"\rEpoch {epoch+1}/{num_epochs} | "
                f"Loss: {running_loss/batch_count:.4f} | "
                f"Est: {running_estimate_loss/batch_count:.4f} | "
                f"Spar: {running_sparsity_loss/batch_count:.4f} | "
                f"Acc: {sparsity_accuracy:.2f}%"
            )
            sys.stdout.flush()
        
        epoch_losses.append(running_loss / batch_count)
        epoch_estimate_losses.append(running_estimate_loss / batch_count)
        epoch_sparsity_losses.append(running_sparsity_loss / batch_count)
        epoch_accuracies.append(sparsity_accuracy)

    print("\nTraining complete.")
    torch.save(model.state_dict(), model_path)

    # Plot loss and accuracy curves
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    
    axes[0].plot(range(1, num_epochs+1), epoch_losses, color='b')
    axes[0].set_title('Total Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(range(1, num_epochs+1), epoch_estimate_losses,color='g')
    axes[1].set_title('Estimate Loss')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    axes[2].plot(range(1, num_epochs+1), epoch_sparsity_losses,  color='r')
    axes[2].set_title('Sparsity Loss')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)

    axes[3].plot(range(1, num_epochs+1), epoch_accuracies, color='m')
    axes[3].set_title('Sparsity Prediction Accuracy')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy (%)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()
    
    return model







def evaluate_model(test_audio_dir: str,
                output_dir: str,
                target_fs_values: List[int],
                clipping_thresholds: List[float],
                time_clip: List[int],
                model_path: str,
                k_classes: int) -> Dict:
  
  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Initialize results storage
  results = {
      'file': [],
      'fs': [],
      'threshold': [],
      'duration': [],
      'sdr_original': [],
      'sdr_reconstructed': [],
      'sdr_improvement': [],
      'processing_time': [],
      'clipped_percentage': []
  }

  # Process each configuration
  total_configs = len(target_fs_values) * len(clipping_thresholds) * len(time_clip)
  pbar = tqdm(total=total_configs, desc="Evaluating configurations")
  
  for target_fs in target_fs_values:
      for clipping_threshold in clipping_thresholds:
          dir_name = f"fs_{target_fs}_threshold_{clipping_threshold:.2f}"
          full_dir_path = os.path.join(output_dir, dir_name)
          os.makedirs(full_dir_path, exist_ok=True)

          for tc in time_clip:
              wav_files = [f for f in os.listdir(test_audio_dir) if f.endswith(".wav")]
              n_files = len(wav_files)
              wav_files = wav_files[:n_files]
              
              for audio_file in wav_files:
                  print(f"\nProcessing: {audio_file}")
                  
                  # Load and preprocess audio
                  data, fs = sf.read(os.path.join(test_audio_dir, audio_file))
                  
                  if len(data.shape) > 1:
                      data = data[:, 0]
                  
                  # Clip to desired duration and normalize
                  data = data[:fs * tc]
                  data = data / np.max(np.abs(data))
                  
                  # Resample to target frequency
                  resampled_data = resample(data, int(target_fs * tc))

                  # Setup parameters
                  Ls = len(resampled_data)
                  win_len = np.floor(Ls/4)
                  win_shift = np.floor(win_len/4)
                  F_red = 2
                  
                  # ASPADE parameters
                  ps_s = 1
                  ps_r = 2
                  ps_epsilon = 0.1
                  ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)
                  
                  # Generate clipped signal
                  print("Generating clipped signal...")
                  
                  # Generate clipped signal
                  clipped_signal, masks, theta, sdr_original, clipped_percentage = \
                      clip_sdr_modified(resampled_data, clipping_threshold)
                  
                  # Perform reconstruction
                  start_time = time()
                  reconstructed_signal = spade_segmentation_eval(
                      clipped_signal, resampled_data, Ls, win_len, win_shift,
                      ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks, model_path, k_classes
                  )
                  processing_time = time() - start_time

                  # Calculate metrics
                  sdr_reconstructed = sdr(resampled_data, reconstructed_signal)
                  sdr_improvement = sdr_reconstructed - sdr_original
                  
                  # Save reconstructed audio
                  output_path = os.path.join(full_dir_path, f"reconstructed_{audio_file}")
                  sf.write(output_path, reconstructed_signal, target_fs)

                  # Store results
                  results['file'].append(audio_file)
                  results['fs'].append(target_fs)
                  results['threshold'].append(clipping_threshold)
                  results['duration'].append(tc)
                  results['sdr_original'].append(sdr_original)
                  results['sdr_reconstructed'].append(sdr_reconstructed)
                  results['sdr_improvement'].append(sdr_improvement)
                  results['processing_time'].append(processing_time)
                  results['clipped_percentage'].append(clipped_percentage)
                  
                  # Plot the reconstructed signal
                  plt.figure(figsize=(8, 4))
                  plt.plot(reconstructed_signal, color='green', linewidth=1.5)
                  plt.title("Reconstructed Signal (ML)")
                  plt.xlabel("Time")
                  plt.ylabel("Amplitude")
                  plt.ylim(-1, 1)  # Set y-axis limits with a margin
                  plt.grid(True, linestyle='--', alpha=0.6)
                  plt.show()
              
              pbar.update(1)
  
  pbar.close()
  
  # Convert results to DataFrame and save
  results_df = pd.DataFrame(results)
  # results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
  
  # Generate summary statistics
  summary = results_df.groupby(['fs', 'threshold', 'duration']).agg({
      'sdr_improvement': ['mean', 'std'],
      'processing_time': 'mean',
      'clipped_percentage': 'mean'
  }).round(2)
  
  return results_df, summary

