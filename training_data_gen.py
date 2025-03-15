import numpy as np
import os
import soundfile as sf
from scipy.signal import resample
from time import time
from tqdm import tqdm
from clip_sdr_modified import clip_sdr_modified
from spade_segmentation import spade_segmentation_train
from typing import Tuple, List, Optional, Dict


def training_data(audio_dir: str,
                      output_path: str,
                      target_fs_values: List[int],
                      clipping_thresholds: List[float],
                      time_clip: List[int]
                      ):
    """
    Complete training pipeline for ASPADE ML enhancement
    """
    
    # Collect training data with progress tracking
    print("\nCollecting training data...")
    training_data = []
    
    total_combinations = len(target_fs_values) * len(clipping_thresholds) * len(time_clip)
    pbar = tqdm(total=total_combinations, desc="Processing configurations")
    
    for target_fs in target_fs_values:
        for clipping_threshold in clipping_thresholds:
            dir_name = f"fs_{target_fs}_threshold_{clipping_threshold:.2f}"
            full_dir_path = os.path.join(output_path, dir_name)
            os.makedirs(full_dir_path, exist_ok=True)

            for tc in time_clip:
                wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
                n_files = len(wav_files)
                wav_files = wav_files[:n_files]

                for i, audio_file in enumerate(wav_files):
                    print(f"\nProcessing: {audio_file}")
                    data, fs = sf.read(os.path.join(audio_dir, audio_file))
                    
                    # Preprocessing steps
                    if len(data.shape) > 1:
                        data = data[:, 0]
                    
                    data = data[: fs * tc]
                    
                    data = data / max(np.abs(data))  
                    
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
                    clipped_signal, masks, theta, true_sdr, percentage = clip_sdr_modified(resampled_data, clipping_threshold)
                    print(f"Clipping stats - Threshold: {theta:.3f}, SDR: {true_sdr:.2f} dB, "
                          f"Clipped: {percentage:.2f}%, Duration: {tc}sec")
                    
                    # Reconstruction and timing
                    print("Performing reconstruction...")
                    start_time = time()
                    reconstructed_signal, metrics, intermediate_training_data = spade_segmentation_train(
                        clipped_signal, resampled_data, Ls, win_len, win_shift,
                        ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks
                    )
                    elapsed_time = time() - start_time
                    print(f"Reconstruction time: {elapsed_time:.2f}s")

                    training_data.extend(intermediate_training_data)

                
                pbar.update(1)
    
    pbar.close()
    
    return training_data