# -*- coding: utf-8 -*-
"""核验说明性长记忆 PDP 下帧尾截断对物理 BER 下界的影响，无随机数。"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.physical_ber_bounds import conditional_frame_genie_bound, expected_fading_frame_genie_bound


def main():
    dynamics_path = ROOT / 'docs/results/continual_v2_channel_dynamics_20261119/offline_estimate_10x11719.json'
    dynamics = json.loads(dynamics_path.read_text(encoding='utf8'))
    power = np.asarray(dynamics['stationary_power_by_tap'], dtype=float)
    templates = {'uniform_117_taps': np.ones(117) / 117,
                 'normalized_offline_average_template': power / power.sum()}
    data_mask = np.arange(1024) >= 256
    full_mask = data_mask & (np.arange(1024) <= 907)
    tail_mask = data_mask & ~full_mask
    rows = []
    for name, template in templates.items():
        for snr in [0, 5, 10, 15]:
            noise = 10 ** (-snr / 10)
            fixed = conditional_frame_genie_bound(np.sqrt(template), data_mask, noise)
            for elapsed in [0., .01, .1, .5, 1., 2., 10.]:
                whole, full, tail = [expected_fading_frame_genie_bound(template, mask, noise,
                    elapsed_over_coherence=elapsed) for mask in (data_mask, full_mask, tail_mask)]
                assert abs(whole['genie_ber_bound'] -
                    (652 * full['genie_ber_bound'] + 116 * tail['genie_ber_bound']) / 768) < 1e-11
                if elapsed == 0.:
                    assert abs(whole['genie_ber_bound'] - fixed['genie_ber_bound']) < 1e-12
                rows.append(dict(power_template=name, snr_db=snr, time_over_coherence=elapsed,
                    genie_ber_bound=whole['genie_ber_bound'], full_column_ber_bound=full['genie_ber_bound'],
                    truncated_column_ber_bound=tail['genie_ber_bound'], quadrature_error=whole['quadrature_error']))
    output = ROOT / 'docs/results/continual_v2_genie_bound_math_20261120/finite_frame_boundary_examples.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(role='deterministic_illustrative_finite_frame_bounds_not_receiver_performance',
        research_seeds_used=[], frame_len=1024, max_delay=116, data_bits=768,
        full_column_bits=652, truncated_column_bits=116,
        source_template_sha256=hashlib.sha256(dynamics_path.read_bytes()).hexdigest(),
        source_template_original_total_power=float(power.sum()),
        power_templates={name: template.tolist() for name, template in templates.items()},
        interpretation='均匀117tap与归一化离线平均模板均为显式数学假设，不是某个实际会话PDP；积分下界不能当实测BER或目标不可能性证明。',
        rows=rows)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('56个整体积分及对应完整/截断分量核验通过，无研究seed。')


if __name__ == '__main__':
    main()
