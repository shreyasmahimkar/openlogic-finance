import numpy as np
import scipy.linalg as la
from google.adk.tools import FunctionTool
from typing import Dict, Any
import time
from .block_convey.prismtrace_client import send_trace_async

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
    t0 = time.time()
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
    
    # Persist state safely (handles both custom SessionState and raw dictionaries)
    if hasattr(state, "set"):
        state.set(f"pi_{agent_id}", new_pi)
        state.set(f"score_{agent_id}", current_loss)
        state.set(f"prev_pred_{agent_id}", prediction)
        state.set(f"prev_loss_{agent_id}", current_loss)
    elif isinstance(state, dict):
        state[f"pi_{agent_id}"] = new_pi
        state[f"score_{agent_id}"] = current_loss
        state[f"prev_pred_{agent_id}"] = prediction
        state[f"prev_loss_{agent_id}"] = current_loss
    
    # Get all expert predictions so far this turn (can only calculate if others ran)
    all_expert_preds = state.get("all_expert_predictions", [0.5, 0.5, 0.5]) if hasattr(state, "get") else [0.5, 0.5, 0.5]
    ret_val = float(np.dot(new_pi, all_expert_preds))
    
    ms = int((time.time() - t0) * 1000)
    send_trace_async(
        user_input=f"{agent_id} pred: {prediction}, gt: {ground_truth}", 
        output=f"Filter updated. new_pi sum: {np.sum(new_pi)}", 
        model="math_guardrail",
        latency_ms=ms,
        step="stochastic_update",
        step_order=state.get("step_order", 0),
        session_id=state.get("session_id", "unknown")
    )
    if hasattr(state, "get") and state.get("step_order"):
        if hasattr(state, "set"):
            state.set("step_order", state.get("step_order") + 1)
        elif isinstance(state, dict):
            state["step_order"] = state.get("step_order") + 1
            
    return ret_val


stochastic_filter_update_tool = FunctionTool(func=stochastic_filter_update)

def robust_gibbs_aggregation(state: Any) -> float:
    """
    Executes Theorem 2: Softmin aggregation and outer-loop Q-Matrix update.
    """
    t0 = time.time()
    
    # ADK ParallelAgent outputs a list to the next Sequential step. 
    # If state is a list, we extract the predictions directly from it and assume uniform scores.
    if isinstance(state, list):
        scores = [0.5, 0.5, 0.5]
        lambda_param = 1.0
        predictions = []
        for item in state:
            try:
                # Try to parse the float from the expert's response
                predictions.append(float(str(item).strip()))
            except ValueError:
                predictions.append(0.5)
        
        # Pad or truncate to exactly 3 experts
        while len(predictions) < 3: predictions.append(0.5)
        predictions = predictions[:3]
        
        # Mock a state.set capability for the downstream tool
        class MockState:
            def set(self, k, v): pass
            def get(self, k, d=None): return d
        state_obj = MockState()
    else:
        scores = [
            state.get("score_Llama_Expert", 0.5) if hasattr(state, "get") else 0.5, 
            state.get("score_GPT4o_Expert", 0.5) if hasattr(state, "get") else 0.5, 
            state.get("score_Mixtral_Expert", 0.5) if hasattr(state, "get") else 0.5
        ]
        lambda_param = state.get("lambda_hyperparam", 1.0) if hasattr(state, "get") else 1.0
        
        predictions = [
            state.get("pred_llama", 0.5) if hasattr(state, "get") else 0.5, 
            state.get("pred_gpt", 0.5) if hasattr(state, "get") else 0.5, 
            state.get("pred_mixtral", 0.5) if hasattr(state, "get") else 0.5
        ]
        state_obj = state
    
    # PAC-Bayes Softmin aggregation
    exp_scores = np.exp(-lambda_param * np.array(scores))
    pi_bar = exp_scores / np.sum(exp_scores)
    
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
    
    if hasattr(state_obj, "set"):
        state_obj.set("global_Q_matrix", Q_new)
        state_obj.set("final_prediction", final_y)
    elif isinstance(state_obj, dict):
        state_obj["global_Q_matrix"] = Q_new
        state_obj["final_prediction"] = final_y
    
    ms = int((time.time() - t0) * 1000)
    session_id = state_obj.get("session_id", "live_adk_run")
    send_trace_async(
        user_input=f"Scores: {scores}",
        output=f"Aggregated prediction: {final_y}",
        model="gibbs_aggregation",
        latency_ms=ms,
        step="robust_aggregation",
        step_order=state_obj.get("step_order", 0),
        session_id=session_id
    )
    if hasattr(state_obj, "get") and state_obj.get("step_order"):
        if hasattr(state_obj, "set"):
            state_obj.set("step_order", state_obj.get("step_order") + 1)
        elif isinstance(state_obj, dict):
            state_obj["step_order"] = state_obj.get("step_order") + 1
            
    return float(final_y)

robust_gibbs_aggregation_tool = FunctionTool(func=robust_gibbs_aggregation)
