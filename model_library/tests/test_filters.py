from model_library.ml_zoo.filters import stochastic_filter_update, robust_gibbs_aggregation


class MockSessionState:
    def __init__(self, initial_state=None):
        self._state = initial_state or {}

    def get(self, key, default=None):
        return self._state.get(key, default)

    def set(self, key, val):
        self._state[key] = val


def test_stochastic_filter_update_with_dict():
    state = {}
    # Run the filter update with dict
    result = stochastic_filter_update(
        agent_id="test_agent", prediction=0.6, ground_truth=0.5, state=state
    )
    assert isinstance(result, float)
    assert "pi_test_agent" in state
    assert "score_test_agent" in state
    assert "prev_pred_test_agent" in state
    assert "prev_loss_test_agent" in state
    assert state["prev_pred_test_agent"] == 0.6


def test_stochastic_filter_update_with_object():
    state = MockSessionState()
    result = stochastic_filter_update(
        agent_id="test_agent", prediction=0.6, ground_truth=0.5, state=state
    )
    assert isinstance(result, float)
    assert state.get("pi_test_agent") is not None
    assert state.get("score_test_agent") is not None
    assert state.get("prev_pred_test_agent") == 0.6


def test_robust_gibbs_aggregation_with_dict():
    state = {
        "score_Llama_Expert": 0.1,
        "score_GPT4o_Expert": 0.2,
        "score_Mixtral_Expert": 0.3,
        "pred_llama": 0.7,
        "pred_gpt": 0.6,
        "pred_mixtral": 0.5,
    }
    result = robust_gibbs_aggregation(state)
    assert isinstance(result, float)
    assert "global_Q_matrix" in state
    assert "final_prediction" in state
    assert state["final_prediction"] == result


def test_robust_gibbs_aggregation_with_object():
    state = MockSessionState(
        {
            "score_Llama_Expert": 0.1,
            "score_GPT4o_Expert": 0.2,
            "score_Mixtral_Expert": 0.3,
            "pred_llama": 0.7,
            "pred_gpt": 0.6,
            "pred_mixtral": 0.5,
        }
    )
    result = robust_gibbs_aggregation(state)
    assert isinstance(result, float)
    assert state.get("global_Q_matrix") is not None
    assert state.get("final_prediction") == result


def test_read_market_indicators():
    from model_library.technical.indicators import read_market_indicators

    res = read_market_indicators("nonexistent.csv")
    assert "Error" in res

    # Verify it reads standard files
    res2 = read_market_indicators("assets/SPY_1mo.csv")
    assert "Close" in res2


def test_apply_semantic_news_filter():
    from model_library.technical.indicators import apply_semantic_news_filter

    res = apply_semantic_news_filter("assets/SPY_1mo.csv")
    assert "High-signal news chunks" in res


def test_robust_gibbs_aggregation_with_expert_keys():
    state = {
        "Llama_Expert": 0.9,
        "GPT4o_Expert": 0.1,
        "Mixtral_Expert": 0.15,
        "score_Llama_Expert": 0.5,
        "score_GPT4o_Expert": 0.5,
        "score_Mixtral_Expert": 0.5,
    }
    result = robust_gibbs_aggregation(state)
    assert isinstance(result, float)
    # y_final = (0.9 + 0.1 + 0.15) / 3 = 1.15 / 3 = 0.38333...
    assert abs(result - 0.38333333333333336) < 1e-6
