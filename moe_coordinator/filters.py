import numpy as np
import scipy.linalg
from typing import List, Dict

class MoEF_Filter:
    """
    Implements the Stochastic Filtering-Based Online Gating logic (MoE-F)
    from "Filtered not Mixed" (Saqur et al.).
    
    This optimal parallel filtering algorithm uses the Wonham-Shiryaev
    equations to update expert reliability based on continuous innovations,
    stochastic drift, and Q-Matrix dynamics.
    """
    def __init__(self, num_experts: int, lambda_reg: float = 1.0):
        self.N = num_experts
        # Lambda controls the "Entropic Regularization"
        self.lam = lambda_reg
        
        # Initialize running scores (loss) s_n
        self.expert_scores = np.zeros(self.N)
        # Initialize uniform mixture weights
        self.weights = np.ones(self.N) / self.N
        # Intensity matrix Q governing regime shifts
        self.Q = np.zeros((self.N, self.N))
        
    def _loss_fn(self, y_true: float, y_pred: float) -> float:
        """
        Calculates the Binary Cross Entropy (BCE) or MSE loss.
        """
        return (y_true - y_pred) ** 2
        
    def update_filter(self, y_true: float, expert_preds: List[float]) -> np.ndarray:
        """
        Executes Step 1 (Optimal Parallel Filtering) from the MoE-F framework.
        Updates the filter by calculating innovations, drift, and diffusion.
        """
        assert len(expert_preds) == self.N, "Mismatch between predictions and experts."
        
        losses = np.array([self._loss_fn(y_true, p) for p in expert_preds])
        self.expert_scores += losses
        
        # 1. Expected average loss (A_bar)
        expected_loss = np.dot(self.weights, losses)
        
        # 2. Innovations Process (Delta W)
        innovations = losses - expected_loss
        
        # 3. Calculate Q-Matrix Dynamics (Matrix Log of row-stochastic P)
        unnormalized_scores = np.exp(-self.lam * self.expert_scores)
        P_bar = unnormalized_scores / np.sum(unnormalized_scores)
        
        # Pseudo-transition matrix P
        P = np.tile(P_bar, (self.N, 1))
        # Add tiny strictly positive noise for matrix log stability
        P = (P + 1e-4) / np.sum(P + 1e-4, axis=1, keepdims=True)
        self.Q = scipy.linalg.logm(P).real
        
        # 4. Stochastic Drift (Q^T * pi)
        drift = np.dot(self.Q.T, self.weights)
        
        # 5. Diffusion Process (pi(A - A_bar)/B)
        diffusion = self.weights * innovations
        
        # Update weights (Euler-Maruyama step) using drift and diffusion
        # We subtract diffusion because higher error should decrease weight
        self.weights = self.weights + drift - diffusion * self.lam
        
        # Enforce probability simplex constraints
        self.weights = np.clip(self.weights, 1e-5, 1.0)
        self.weights /= np.sum(self.weights)
        
        return self.weights

    def predict_ensemble(self, new_expert_preds: List[float]) -> float:
        """
        Calculates the robust ensemble prediction.
        """
        assert len(new_expert_preds) == self.N, "Mismatch between predictions and experts."
        return float(np.dot(self.weights, new_expert_preds))

    def get_expert_rankings(self) -> Dict[int, float]:
        """Returns the current gating allocation for observability."""
        return {i: float(self.weights[i]) for i in range(self.N)}

    def get_state(self) -> Dict:
        """Exports ADK memory state."""
        return {
            "expert_scores": self.expert_scores.tolist(),
            "weights": self.weights.tolist(),
            "Q": self.Q.tolist()
        }

    def set_state(self, state: Dict):
        """Loads ADK memory state."""
        self.expert_scores = np.array(state.get("expert_scores", np.zeros(self.N)))
        self.weights = np.array(state.get("weights", np.ones(self.N) / self.N))
        self.Q = np.array(state.get("Q", np.zeros((self.N, self.N))))
