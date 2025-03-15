import numpy as np
import torch
import torch.nn as nn
import os
from typing import Tuple, List, Optional, Dict
from fra import frana,frsyn
from proj_time import proj_time
from hard_thresholding import hard_thresholding
from torch.utils.data import Dataset
from ML_model import ASPADEDataset,ASPADEModel


# Training version
def ml_aspade_train(data_clipped: np.ndarray,
                    masks: np.ndarray,
                    Ls: int,
                    max_it: int,
                    epsilon: float,
                    r: int,
                    s: int,
                    redundancy: float) -> Tuple[np.ndarray, dict, List[dict]]:

    max_it = int(max_it)
    x_hat = np.copy(data_clipped)
    zEst = frana(x_hat, redundancy)
    zEst_init = zEst
    u = np.zeros(len(zEst), dtype=complex)
    k = s
    cnt = 1
    bestObj = float('inf')

    
    # Initialize tracking
    obj_history = []
    sparsity_history = []
    # Dynamic sparsity parameters
    obj_his = np.zeros((3,1))   # Store last 3 objective values
    imp_thres = 1e-4    # Minimum improvement threshold
    max_sparsity = len(zEst) * 0.75  # Maximum sparsity limit (50% of coefficients)
    
    
    best_x_hat = None
    best_k = k
    
    for cnt in range(max_it):
        # Sparsity projection
        z_bar = hard_thresholding(zEst + u, k)
        objVal = np.linalg.norm(zEst - z_bar)
        
        obj_history.append(objVal)
        sparsity_history.append(k)
        
        # Store objective value history
        obj_his = np.roll(obj_his, 1)
        obj_his[0] = objVal
        
        
        if objVal < bestObj:
            data_rec = x_hat
            bestObj = objVal
            
            best_x_hat = np.copy(x_hat)
            best_k = k
        
        if cnt > 3:
            rel_improvement = (obj_his[2] - objVal) / obj_his[2]    # Calculate relative improvement
            
            if rel_improvement < imp_thres:
                k = min(k + 2 * s, max_sparsity)    # Slow convergence - increase sparsity more aggressively
            elif rel_improvement > 5 * imp_thres:
                k = k   # Fast convergence - maintain current sparsity
            else:
                if cnt % r == 0:
                    k = min(k + s, max_sparsity)

        if cnt > 1:
            adap_epsilon = epsilon * (1 + 0.1 * np.log(cnt))
        else:
            adap_epsilon = epsilon    # termination step with adaptive threshold

        if objVal <= adap_epsilon:
            break
        
        # Signal reconstruction
        b = z_bar - u
        syn = frsyn(b, redundancy)
        syn = syn[:Ls]
        x_hat = proj_time(syn, masks, data_clipped)
        
        # Update estimates
        zEst = frana(x_hat, redundancy)
        u = u + zEst - z_bar
        
        cnt+=1
    
    metrics = {
        'iterations': cnt + 1,
        'final_objective': objVal,
        'best_objective': bestObj,
        'objective_history': obj_history,
        'sparsity_history': sparsity_history,
        'final_sparsity': k,
        'best_sparsity': best_k/len(zEst),
        'best_estimate': zEst + u,
        'initial_estimate': zEst_init
    }

    
    return best_x_hat if best_x_hat is not None else x_hat, metrics



# Evaluation version 
def ml_aspade_eval(data_clipped: np.ndarray,
                   masks: np.ndarray,
                   Ls: int,
                   max_it: int,
                   epsilon: float,
                   r: int,
                   s: int,
                   redundancy: float,
                   model_path: str,
                   k_classes: int) -> Tuple[np.ndarray, dict]:
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model path {model_path} does not exist.")

    max_it = int(max_it)
    x_hat = np.copy(data_clipped)
    zEst_init = frana(x_hat, redundancy)

    zEst,k = predict_with_model(model_path, zEst_init)
    # k = int(k * len(zEst))
    # k = min(k,len(zEst)//2)
   


    # k = min(k,len(zEst)//2)


    u = np.zeros(len(zEst), dtype=complex)
    cnt = 1
    bestObj = float('inf')
    
    
    # Dynamic sparsity parameters
    obj_his = np.zeros((3,1))   # Store last 3 objective values
    imp_thres = 1e-4    # Minimum improvement threshold
    max_sparsity = len(zEst) * 0.75  # Maximum sparsity limit (50% of coefficients)

    while cnt <= max_it:
        # set all but k largest coefficients to zero (complex conjugate pairs are taken into consideration)
        z_bar = hard_thresholding(zEst + u, k)

        objVal = np.linalg.norm(zEst - z_bar)  # update termination function

        # Store objective value history
        obj_his = np.roll(obj_his, 1)
        obj_his[0] = objVal
        
        if objVal <= bestObj:
            data_rec = x_hat
            bestObj = objVal

        # Dynamic sparsity update based on convergence behavior

        if cnt > 3:
            rel_improvement = (obj_his[2] - objVal) / obj_his[2]    # Calculate relative improvement
            
            if rel_improvement < imp_thres:
                k = min(k + 2 * s, max_sparsity)    # Slow convergence - increase sparsity more aggressively
            elif rel_improvement > 5 * imp_thres:
                k = k   # Fast convergence - maintain current sparsity
            else:
                if cnt % r == 0:
                    k = min(k + s, max_sparsity)

        if cnt > 1:
            adap_epsilon = epsilon * (1 + 0.1 * np.log(cnt))
        else:
            adap_epsilon = epsilon    # termination step with adaptive threshold

        if objVal <= adap_epsilon:
            break

        # projection onto the set of feasible solutions    
        b = z_bar - u
        syn = frsyn(b, redundancy)
        syn = syn[:Ls]
        x_hat = proj_time(syn, masks, data_clipped)
        
        # dual variable update
        zEst = frana(x_hat, redundancy)
        u = u + zEst - z_bar
        
        cnt += 1    # iteration counter update

    return x_hat


def predict_with_model(model_path, initial_estimate, max_sparsity=1000):
    model = ASPADEModel(2000, 2000, max_sparsity)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        # Convert complex initial estimate to real representation
        real_input = np.hstack([initial_estimate.real, initial_estimate.imag])
        input_tensor = torch.tensor(real_input, dtype=torch.float32).unsqueeze(0)
        
        # Get predictions
        pred_output, pred_k_logits = model(input_tensor)
        
        # Convert back to complex representation
        n = pred_output.shape[1] // 2
        pred_real = pred_output[0, :n].numpy()
        pred_imag = pred_output[0, n:].numpy()
        pred_complex = pred_real + 1j * pred_imag
        
        # Get the sparsity as an integer
        _, predicted_k = torch.max(pred_k_logits, 1)
        pred_sparsity = predicted_k.item()  # This will be an integer
        
        return pred_complex, pred_sparsity



# def predict_with_model(model_path, initial_estimate):
#     model = ASPADEModel(2000, 2000)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     model.eval()
#     with torch.no_grad():
#         # Convert complex initial estimate to real representation
#         real_input = np.hstack([initial_estimate.real, initial_estimate.imag])
#         input_tensor = torch.tensor(real_input, dtype=torch.float32).unsqueeze(0)
        
#         # Get predictions
#         pred_output, pred_sparsity = model(input_tensor)
        
#         # Convert back to complex representation
#         n = pred_output.shape[1] // 2
#         pred_real = pred_output[0, :n].numpy()
#         pred_imag = pred_output[0, n:].numpy()
#         pred_complex = pred_real + 1j * pred_imag
        
#         return pred_complex, pred_sparsity.item()