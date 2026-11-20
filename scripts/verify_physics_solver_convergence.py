# -*- coding: utf-8 -*-
"""用无随机数的有限帧线性方程检查物理解码器迭代误差，不运行研究会话。"""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from agent.cir_estimator import CIRCondition
from agent.unfolded_equalizer import UnfoldedEqualizer


def main():
    torch.set_num_threads(1)
    previous_backend = os.environ.get('RL4EQ_LINEAR_OPERATOR_BACKEND')
    os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = 'fft'
    try:
        rows = verify_equations()
    finally:
        if previous_backend is None:
            os.environ.pop('RL4EQ_LINEAR_OPERATOR_BACKEND', None)
        else:
            os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = previous_backend
    payload = dict(
        role='deterministic_solver_verification_not_receiver_performance',
        research_seeds_used=[], frame_len=1024, max_delay=116, adapt_symbols=224,
        unknown_symbols=800, operator_backend='fft', torch_version=torch.__version__,
        waveform='x[k]=sign(sin(17*k+0.2))；无噪声、无tail、无随机数',
        reference='float64显式有限帧矩阵；(Re(U^H U)+noise/2 I)z=Re(U^H(y-H_known*x_known))',
        interpretation='noise只是线性方程正则化参数；无仿真会话或BER。符号差是与精确线性松弛解的差，不是比特错误。模板不代表实际PDP。',
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('scripts/verify_physics_solver_convergence.py',
                         'agent/unfolded_equalizer.py', 'env/linear_operator.py')},
        rows=rows)
    output = ROOT / 'docs/results/continual_v2_solver_math_20261120/equation_examples.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(f'{len(rows)}个确定性线性方程求解比较完成；没有运行研究会话或汇总P8中途性能。')


def verify_equations():
    length, memory = 1024, 116
    index = torch.arange(length, dtype=torch.float64)
    symbols = torch.where(torch.sin(index * 17.0 + .2) > 0, 1., -1.)
    known_mask = index < 224
    columns = torch.arange(length)[~known_mask]
    delays = torch.arange(length)[:, None] - torch.arange(length)[None, :]
    rows = []
    for name in ('direct', 'two_path_notch', 'long_diffuse'):
        cir = torch.zeros(memory + 1, dtype=torch.complex128)
        if name == 'direct':
            cir[0] = 1
        elif name == 'two_path_notch':
            cir[0] = cir[70] = 2 ** -.5
        else:
            tap = torch.arange(memory + 1, dtype=torch.float64)
            cir = torch.exp(-tap / 45) * torch.exp(1j * (tap * .93 + tap.square() * .011))
            cir /= cir.norm()
        # 显式矩阵保留因果卷积的帧尾截断，不用循环卷积代替。
        matrix = torch.where((delays >= 0) & (delays <= memory), cir[delays.clamp(0, memory)], 0)
        unknown_matrix = matrix[:, columns]
        received = matrix @ symbols.to(torch.complex128)
        target = received - matrix[:, known_mask] @ symbols[known_mask].to(torch.complex128)
        rhs = (unknown_matrix.mH @ target).real
        gram = (unknown_matrix.mH @ unknown_matrix).real
        for noise in (1., 10 ** -.5, .1, 10 ** -1.5):
            normal = gram + noise * .5 * torch.eye(columns.numel(), dtype=torch.float64)
            exact = torch.linalg.solve(normal, rhs)
            reference_residual = float((normal @ exact - rhs).norm() / rhs.norm())
            assert reference_residual < 1e-10
            reference_logits = (2 * exact / noise).clamp(-40, 40)
            condition = CIRCondition(complex_cir=cir.to(torch.complex64)[None],
                support_probability=cir.abs().float()[None], noise_variance=torch.tensor([noise]),
                confidence=torch.ones(1), latent_residual=torch.zeros(1, 96))
            for iterations in (8, 16, 32, 64):
                holder = SimpleNamespace(config=SimpleNamespace(physics_warm_start_iterations=iterations))
                logits = UnfoldedEqualizer._real_bpsk_known_adapt_logits(
                    holder, received.to(torch.complex64)[None], condition,
                    torch.zeros(1, memory, dtype=torch.complex64),
                    symbols.float().to(torch.complex64)[None], known_mask[None])[0, columns].double()
                assert bool(torch.isfinite(logits).all())
                error = logits - reference_logits
                if name == 'direct':
                    assert float(error.abs().max()) < 1e-5
                visible = (2 * exact / noise).abs() < 39
                relative = (float((logits[visible] * noise * .5 - exact[visible]).norm()
                                  / exact[visible].norm()) if bool(visible.any()) else None)
                rows.append(dict(template=name, regularization_noise=noise, iterations=iterations,
                    reference_relative_equation_residual=reference_residual,
                    clipped_logits_max_abs_difference=float(error.abs().max()),
                    clipped_logits_rms_difference=float(error.square().mean().sqrt()),
                    unknown_sign_disagreements_with_exact=int(((logits >= 0) != (exact >= 0)).sum()),
                    nonclipped_reference_columns=int(visible.sum()),
                    nonclipped_relative_estimate_error=relative))
    return rows


if __name__ == '__main__':
    main()
