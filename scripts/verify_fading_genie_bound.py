# -*- coding: utf-8 -*-
"""无随机数地核验说明性 PDP 的额外信息 BER 下界积分，不运行研究会话。"""

import argparse
import json
import math
from pathlib import Path

from scipy.integrate import quad
from scipy.special import ndtr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/results/continual_v2_genie_bound_math_20261120/deterministic_quadrature.json')
    args = parser.parse_args()
    vectors = {'single_tap': [1.0], 'two_equal_taps': [.5, .5], 'three_unequal_taps': [.7, .2, .1]}
    rows = []
    for name, power in vectors.items():
        for snr in [0, 5, 10, 15]:
            noise = 10 ** (-snr / 10)
            for time_over_coherence in [0., .01, .1, .5, 1., 2., 10.]:
                a = math.exp(-2 * time_over_coherence)
                v = 1 - a

                def integrand(theta):
                    sin2 = math.sin(theta) ** 2
                    if sin2 == 0:
                        return 0.
                    s = 1 / (noise * sin2)
                    log_laplace = sum(-math.log1p(s * p * v) - s * p * a / (1 + s * p * v) for p in power)
                    return math.exp(log_laplace) / math.pi

                value, error = quad(integrand, 0, math.pi / 2, epsabs=1e-13, epsrel=1e-10)
                if time_over_coherence == 0.:
                    assert abs(value - ndtr(-math.sqrt(2 / noise))) < 1e-11
                if name == 'single_tap' and time_over_coherence == 10.:
                    assert abs(value - .5 * (1 - math.sqrt(1 / (1 + noise)))) < 1e-10
                rows.append(dict(illustrative_power_vector=name, snr_db=snr,
                    time_over_coherence=time_over_coherence, genie_ber_bound=value, quadrature_error=error))
    for name in vectors:
        for snr in [0, 5, 10, 15]:
            values = [row['genie_ber_bound'] for row in rows if
                      row['illustrative_power_vector'] == name and row['snr_db'] == snr]
            assert all(b >= a - 1e-13 for a, b in zip(values, values[1:]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(role='deterministic_mathematical_examples_not_channel_experiment',
        research_seeds_used=[], power_vectors=vectors,
        interpretation='显式假设的说明性PDP；不是项目真实PDP或接收性能，不用于方法选取或确认。全能量单比特genie下界，未计帧尾截断。',
        rows=rows), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('84个无随机数积分通过AWGN/瑞利闭式及单调性核验，未生成或复用研究seed。')


if __name__ == '__main__':
    main()
