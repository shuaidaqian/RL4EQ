# -*- coding: utf-8 -*-
"""实际推理输入快照与离线监督更新的边界。"""

import torch
from torch import nn

from agent.cir_estimator import CIRCondition
from training.continual_offline import InputRecorder, train_examples


class ToyEqualizer(nn.Module):
    """用于检查录制与梯度更新，不用于接收性能比较。"""
    config = None

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)

    def forward(self, rx, condition, *, adapt_symbols):
        logits = self.head(rx).squeeze(-1)
        return logits, logits.sigmoid()


def test_actual_inputs_are_copied_and_supervision_updates_only_after_rollout():
    torch.manual_seed(9)
    model = ToyEqualizer()
    recorder = InputRecorder(model)
    rx = torch.tensor([[[1., -.2], [.4, -.1], [-.2, -.7]]])
    condition = CIRCondition(torch.ones((1, 1), dtype=torch.complex64), torch.ones((1, 1)),
                             torch.ones(1), torch.ones(1), torch.zeros((1, 96)))
    adapt = torch.tensor([[1., 0., 0.]])
    before = {name:p.detach().clone() for name,p in model.named_parameters()}
    with torch.no_grad():
        output, _ = recorder(rx, condition, adapt_symbols=adapt)
    labels = torch.tensor([True, True, False])
    example = recorder.example(labels, torch.tensor([False, True, True]), 0)
    labels[:] = False
    rx[:] = 100
    condition.complex_cir[:] = 10
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    copied, _ = model(*example.args, **example.kwargs)
    torch.testing.assert_close(copied, output, rtol=0, atol=0)
    assert example.labels.tolist() == [True, True, False]
    assert example.args[1].complex_cir.item() == 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    rows = train_examples(model, optimizer, [example], order=[0])
    assert rows[0]['data_bits'] == 2
    assert any(not torch.equal(parameter, before[name]) for name,parameter in model.named_parameters())
