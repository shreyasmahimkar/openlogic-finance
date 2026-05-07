import numpy as np
import scipy.linalg as la
from google.adk.tools import FunctionTool
from typing import Dict, Any

def compute_input_sensitivity_gradient(prediction: float, ground_truth: float, delta_f: float) -> float:
    # A_t implementation (Simplified for brevity based on BCE/MSE helper functions)
    return (ground_truth - prediction) * delta_f

def compute_target_sensitivity_gradient(prediction: float) -> float:
    # B_t implementation
    return 1.0  # Placeholder simplifying to constant diffusion scalar

def calculate_loss(prediction: float, ground_truth: float) -> float:
    return (prediction - ground_truth) ** 2

def stochastic_filter_update(agent_id: str, prediction: float, ground_truth: float, state: Any) -> float:
    """
    Executes the Euler-Maruyama discrete update for the Wonham-Shiryaev filter.
    Calculates At, Bt, Delta W, and updates the local belief pi.
    """
    # Initialize state variables safely
    num_experts = 3
    pi = state.get(f"pi_{agent_id}", np.ones(num_experts) / num_experts)
    # Default uniform transition matrix if not present
    Q_default = (1.0 / (num_experts - 1)) * (np.ones((num_experts, num_experts)) - np.eye(num_experts))
    # Correct diagonal to make rows sum to 0
    np.fill_diagonal(Q_default, -1.0)
    
    Q = state.get("global_Q_matrix", Q_default)
    
    prev_pred = state.get(f"prev_pred_{agent_id}", 0.5)
    prev_loss = state.get(f"prev_loss_{agent_id}", 0.25)
    
    # delta_f approximates the time-derivative of the expert's prediction
    delta_f = prediction - prev_pred
    
    # Calculate SDE components based on Helper Functions 
    A_t = compute_input_sensitivity_gradient(prediction, ground_truth, delta_f) 
    B_t = compute_target_sensitivity_gradient(prediction)
    
    # Since pi is a vector representing the belief over true underlying experts, A_bar_t is the expected value
    A_bar_t = np.dot(A_t * np.ones(num_experts), pi)
    
    # Innovations process formulation
    current_loss = calculate_loss(prediction, ground_truth)
    delta_loss = current_loss - prev_loss
    delta_W = (delta_loss - A_bar_t) / B_t
    
    # Update Belief State via Drift and Diffusion
    drift = np.dot(Q.T, pi)
    
    # Diffusion
    A_t_vec = A_t * np.ones(num_experts)
    diffusion = pi * (A_t_vec - A_bar_t) / B_t
    
    # Update equation
    new_pi = pi + drift + (diffusion * delta_W)
    
    # Enforce probability simplex constraints
    new_pi = np.clip(new_pi, 1e-6, 1.0)
    new_pi = new_pi / np.sum(new_pi)
    
    # Persist state back to ADK SessionState
    state.set(f"pi_{agent_id}", new_pi)
    state.set(f"score_{agent_id}", current_loss)
    state.set(f"prev_pred_{agent_id}", prediction)
    state.set(f"prev_loss_{agent_id}", current_loss)
    
    # Get all expert predictions so far this turn (can only calculate if others ran)
    all_expert_preds = state.get("all_expert_predictions", [0.5, 0.5, 0.5])
    return float(np.dot(new_pi, all_expert_preds))


stochastic_filter_update_tool = FunctionTool(func=stochastic_filter_update)

def robust_gibbs_aggregation(state: Any) -> float:
    """
    Executes Theorem 2: Softmin aggregation and outer-loop Q-Matrix update.
    """
    scores = [
        state.get("score_Llama_Expert", 0.5), 
        state.get("score_GPT4o_Expert", 0.5), 
        state.get("score_Mixtral_Expert", 0.5)
    ]
    lambda_param = state.get("lambda_hyperparam", 1.0)
    
    # PAC-Bayes Softmin aggregation
    exp_scores = np.exp(-lambda_param * np.array(scores))
    pi_bar = exp_scores / np.sum(exp_scores)
    
    # Final robust ensemble prediction
    predictions = [
        state.get("pred_llama", 0.5), 
        state.get("pred_gpt", 0.5), 
        state.get("pred_mixtral", 0.5)
    ]
    final_y = np.dot(pi_bar, predictions)
    
    # Bi-level robust Q-Matrix Update with Regularization Perturbation
    alpha = 0.05
    P = np.tile(pi_bar, (len(scores), 1))
    P_reg = (1 - alpha) * P + alpha * np.eye(len(scores))
    
    # Principal Matrix Logarithm to yield valid transition matrix intensity
    Q_new = np.maximum(0, la.logm(P_reg).real)
    np.fill_diagonal(Q_new, 0)
    # Ensure row sums = 0
    Q_new = Q_new - np.diag(np.sum(Q_new, axis=1))
    
    state.set("global_Q_matrix", Q_new)
    state.set("final_prediction", final_y)
    return float(final_y)

robust_gibbs_aggregation_tool = FunctionTool(func=robust_gibbs_aggregation)
