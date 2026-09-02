"""验证低置信度物理状态不会污染神经相位条件。"""


def test_tracked_cfo_requires_physical_state_confidence():
    import compare
    from agent.cir_estimator import PilotPhysicalState

    low = PilotPhysicalState(cfo_cycles_per_symbol=0.0007, confidence=0.34)
    high = PilotPhysicalState(cfo_cycles_per_symbol=0.0007, confidence=0.80)

    assert compare._tracked_cfo_or_none(low, threshold=0.35) is None
    assert compare._tracked_cfo_or_none(high, threshold=0.35) == 0.0007
