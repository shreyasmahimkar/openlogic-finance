import numpy as np
from typing import List, Dict

class MoEF_Filter:
    """
    Implements the Stochastic Filtering-Based Online Gating logic (MoE-F)
    from "Filtered not Mixed" (Saqur et al.).
    
    This filter tracks the running loss of N experts and uses a Gibbs/Softmin
    aggregation mechanism to yield robust dynamic mixture weights.
    """
    def __init__(self, num_experts: int, lambda_reg: float = 1.0):
        self.N = num_experts
        # Lambda controls the "Entropic Regularization" (how sharply we punish bad experts)
        self.lam = lambda_reg
        
        # Initialize running scores (loss) s_n for each expert to 0
        self.expert_scores = np.zeros(self.N)
        
        # Initialize uniform mixture weights
        self.weights = np.ones(self.N) / self.N
        
    def _loss_fn(self, y_true: float, y_pred: float) -> float:
        """
        Calculates the Binary Cross Entropy (BCE) or MSE loss.
        For simplicity on categorical tasks (Rise/Fall), we use a binary discrepancy.
        """
        # Simple MSE for predictions mapped to continuous space [0, 1]
        return (y_true - y_pred) ** 2
        
    def update_filter(self, y_true: float, expert_preds: List[float]):
        """
        Executes Step 2 (Robust Aggregation) from Algorithm 1 of the MoE-F paper.
        In an online setting, when ground truth y_true is revealed, we update
        the loss score of each expert and recalculate the softmin routing weights.
        """
        assert len(expert_preds) == self.N, "Mismatch between predictions and experts."
        
        for n in range(self.N):
            # Calculate instantaneous loss
            loss_t = self._loss_fn(y_true, expert_preds[n])
            # Accumulate into running score s_n
            self.expert_scores[n] += loss_t
            
        # Execute Softmin (Gibbs-aggregation) to compute new mixture matrix P_bar
        # \bar{\pi} <- e^{-\lambda s_n} / \sum e^{-\lambda s_i}
        unnormalized_scores = np.exp(-self.lam * self.expert_scores)
        self.weights = unnormalized_scores / np.sum(unnormalized_scores)
        
        return self.weights

    def predict_ensemble(self, new_expert_preds: List[float]) -> float:
        """
        Calculates the robust ensemble prediction: \sum \bar{\pi}_n \hat{Y}^{(n)}_t
        """
        assert len(new_expert_preds) == self.N, "Mismatch between predictions and experts."
        prediction = np.dot(self.weights, new_expert_preds)
        return prediction

    def get_expert_rankings(self) -> Dict[int, float]:
        """Returns the current gating allocation for observability."""
        return {i: float(self.weights[i]) for i in range(self.N)}
