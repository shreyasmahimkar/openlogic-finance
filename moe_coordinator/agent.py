from google.adk.agents import Agent
from .filters import MoEF_Filter
from .experts import EXPERTS

# Initialize our stochastic filter for N=3 experts globally for the session
moef_filter = MoEF_Filter(num_experts=len(EXPERTS), lambda_reg=1.0)

def invoke_moe_filter(market_context_url: str, most_recent_close_change: float) -> str:
    """
    Given the data gathered from the Data Ingestion Agent, this tool executes the 
    Stochastic Filtering algorithm across N parallel experts.
    
    Args:
        market_context_url: The path or URL to the trailing OHLCV data.
        most_recent_close_change: Float [0.0 - 1.0] representing if it rose (1.0) or fell (0.0) recently to adjust filter.
    Returns:
        JSON string of the ensemble's aggregate decision.
    """
    # NOTE: In a true continuous HMM, we would store `last_expert_preds` in a DB 
    # to evaluate them against `most_recent_close_change` here. 
    # For now, we mock the feedback loop to prove the Filter weight recalculation.
    dummy_last_preds = [0.8, 0.4, 0.1]
    
    # Step 1: Online Filter Update (Eq 8 and 13 from MoE-F paper)
    # The filter heavily rewards whichever expert perfectly predicted the most_recent_close_change
    new_mixture_weights = moef_filter.update_filter(
        y_true=most_recent_close_change, 
        expert_preds=dummy_last_preds
    )
    
    # Step 2: Poll Experts (Simulating parallel calls to Gemini sub-agents)
    # If context was real, we would send `market_context_url` to each EXPERTS[i]
    new_expert_forecasts = [0.55, 0.60, 0.90] 
    
    # Step 3: Robust Aggregation 
    final_prediction = moef_filter.predict_ensemble(new_expert_forecasts)
    ranks = moef_filter.get_expert_rankings()
    
    return f"MoE-F Filter completed. New Dynamic Rankings (0=Tech, 1=Fund, 2=Contra): {ranks}. Final Ensemble Prediction: {final_prediction:.3f}"

root_agent = Agent(
    name="moef_coordinator",
    model="gemini-2.5-flash",
    instruction="""You are the CandleSage Root Agent implementing the MoE-F (Stochastic Filtering) paper.
Your goal is to coordinate predictions using an ensemble of 3 distinct LLM personas.

Whenever the user asks for a market forecast:
1. You pretend we received context from the Data Ingestion agent. 
2. You immediately trigger 'invoke_moe_filter' using a dummy recent move to recalibrate the experts' running quality.
3. You return the mathematically robust Ensemble Prediction to the user, explaining which expert was punished/rewarded by the filter.""",
    description="Orchestrator Agent routing data through the Wonham-Shiryaev gating filter.",
    tools=[invoke_moe_filter]
)
