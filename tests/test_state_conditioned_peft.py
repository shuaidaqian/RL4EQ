# -*- coding: utf-8 -*-
"""状态 embedding 条件化 PEFT 的最小契约测试。"""

import torch

from agent.unfolded_equalizer import OnlinePhaseTrendAdapter


def test_state_conditioned_phase_adapter_zero初始化仍保持恒等映射():
    """映射尚未在线学习时，加入状态向量不能改变冻结模型输出。"""

    torch.manual_seed(7)
    adapter = OnlinePhaseTrendAdapter()
    rx_iq = torch.randn(2, 24, 2)
    state = torch.randn(2, 6)

    corrected = adapter(rx_iq, state_embedding=state)

    assert torch.allclose(corrected, rx_iq)


def test_state_conditioned_phase_adapter_has_adapt_pilot_gradient_path():
    """Adapt Pilot 损失应能通过状态向量调制参数，而不读取 Data 标签。"""

    torch.manual_seed(11)
    adapter = OnlinePhaseTrendAdapter()
    rx_iq = torch.randn(1, 32, 2)
    state = torch.tensor([[0.2, -0.1, 0.3, 0.4, 0.7, 0.1]])
    target = rx_iq.clone()
    target[..., 0] = target[..., 0] + 0.1

    loss = torch.mean((adapter(rx_iq, state_embedding=state) - target) ** 2)
    loss.backward()

    gradient = adapter.state_to_phase_cfo.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_state_conditioned_phase_adapter_rejects_wrong_embedding_shape():
    """状态维度错误时应立即报错，避免静默使用错误的条件。"""

    adapter = OnlinePhaseTrendAdapter()
    rx_iq = torch.zeros(1, 8, 2)

    try:
        adapter(rx_iq, state_embedding=torch.zeros(1, 5))
    except ValueError as error:
        assert "state_embedding" in str(error)
    else:
        raise AssertionError("错误的状态维度没有触发 ValueError。")
