# -*- coding: utf-8 -*-
"""用复高斯AR(1)精确功率协方差检查说明性时间平均精度，不生成轨迹。"""

import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'docs/results/continual_v2_channel_dynamics_20261119/offline_estimate_10x11719.json'
    data = json.loads(source.read_text(encoding='utf8'))
    q = data['rho_estimate'] ** 2
    n = data['frames_per_seed'] - 1
    start = 2
    total_variance = 0.
    for offset in range(n):
        index = start + offset
        remaining = n - offset - 1
        geometric = q * (-math.expm1(remaining * math.log(q))) / (1 - q) if remaining else 0.
        total_variance += (1 - q ** (2 * index)) * (1 + 2 * geometric)
    relative_variance = total_variance / n ** 2
    for count in [2, 3, 5]:
        coefficient = .7
        direct = sum(coefficient ** abs(i - j) * (1 - coefficient ** (2 * min(i, j)))
            for i in range(1, count + 1) for j in range(1, count + 1)) / count ** 2
        grouped = sum((1 - coefficient ** (2 * i)) *
            (1 + 2 * sum(coefficient ** delay for delay in range(1, count - i + 1)))
            for i in range(1, count + 1)) / count ** 2
        assert abs(direct - grouped) < 1e-14
    result = dict(role='deterministic_moment_calculation_not_estimator_confidence_interval',
        research_seeds_used=[], frozen_dynamics_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        assumed_known_rho=data['rho_estimate'], power_average_indices=[start, start + n - 1],
        states_per_session=n, sessions=10, one_tap_mean_power_relative_variance=relative_variance,
        one_tap_relative_std=math.sqrt(relative_variance),
        ten_identical_independent_sessions_relative_std=math.sqrt(relative_variance / 10),
        interpretation='条件于固定单会话基准功率、固定初始幅度与已知rho的精确矩计算。没有计入会话PDP异质性或rho估计误差，不是当前D的置信区间或实际有效样本数。')
    output = root / 'docs/results/continual_v2_genie_bound_math_20261120/slow_power_precision_example.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('功率协方差双重求和与几何求和核验通过；无研究seed，不是实际D估计的区间。')


if __name__ == '__main__':
    main()
