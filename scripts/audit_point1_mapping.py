# -*- coding: utf-8 -*-
"""只读审计第一研究点原始 PDP、扩频等效条件与当前 profile 的数值差异。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import loadmat
from scipy.signal import max_len_seq

from env.eme_channel_profiles import EMEChannelProfileConfig, sample_eme_profile
from env.experiment_config import build_comm_env_config
from evaluation.seed_registry import reserve_seeds


def pdp_statistics(delays, power):
    power = np.asarray(power, float)
    delays = np.asarray(delays, float)
    if delays.shape != power.shape or np.any(power < 0) or not np.isfinite(power).all() or power.sum() <= 0:
        raise ValueError('PDP 必须为非负有限同形数组，且总功率为正。')
    p = power/power.sum()
    mean = float(p@delays)
    return dict(mean_delay_ms=1000*mean, rms_delay_ms=1000*float(np.sqrt(p@((delays-mean)**2))),
        energy_after_1ms=float(p[delays>0.001].sum()), energy_after_5ms=float(p[delays>0.005].sum()),
        delay90_ms=1000*float(delays[np.searchsorted(np.cumsum(p), .9)]),
        effective_taps90=int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), .9)+1))


def equivalence_check(seed):
    """理想同步、重复扩频码和整数码片路径下验证解扩等效，不执行原 MATLAB。"""
    rng = np.random.default_rng(seed)
    code = 2*max_len_seq(7, state=[1, 0, 0, 0, 0, 0, 0], taps=[6])[0].astype(float)-1
    length, memory, count = len(code), 6, 64
    symbols = 2*rng.integers(0, 2, size=memory+count)-1
    chips = (symbols[:, None]*code[None]).flatten()
    taps = (rng.normal(size=memory+1)+1j*rng.normal(size=memory+1))/np.sqrt(2*(memory+1))
    actual = np.zeros(length*count, complex)
    expected = np.zeros(count, complex)
    for delay, tap in enumerate(taps):
        actual += tap*chips[(memory-delay)*length:(memory-delay+count)*length]
        expected += tap*symbols[memory-delay:memory-delay+count]
    despread = (actual.reshape(count, length)*code).mean(1)
    # 码片延迟不为整符号时，会分裂为两个相邻符号的相关系数。
    whole, remainder = 2, 13
    a = float(code[remainder:]@code[:-remainder]/length)
    b = float(code[:remainder]@code[-remainder:]/length)
    begin = (memory-whole)*length-remainder
    fractional = (chips[begin:begin+count*length].reshape(count, length)*code).mean(1)
    fractional_expected = a*symbols[memory-whole:memory-whole+count] + b*symbols[memory-whole-1:memory-whole-1+count]
    return dict(seed=seed, spreading_length=length, symbol_aligned_max_abs=float(np.max(np.abs(despread-expected))),
        fractional_chip_delay=remainder, fractional_a=a, fractional_b=b,
        fractional_max_abs=float(np.max(np.abs(fractional-fractional_expected))),
        white_chip_noise_to_symbol_variance_ratio=1/length,
        processing_gain_db=10*float(np.log10(length)),
        limitation='理想离散链路代数检查，未包含原 MATLAB 分数时延滤波器、群延迟、残余 CFO 或接收对齐误差')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--config', default='configs/eme_long_memory_v2_online_dd_tail_v1.json')
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--equivalence-seed', type=int, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('禁止覆盖已有物理映射审计。')
    source = Path(args.source_root)
    files = ['03_仿真程序/Multipath/channel_100.mat', '03_仿真程序/Multipath/Doppler_Spread_RLS_DFE01.m',
             '03_仿真程序/Multipath/spread.m', '03_仿真程序/Multipath/despread.m',
             '03_仿真程序/Copy_of_Multipath/channel_generator.m']
    source_hashes = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    mat = loadmat(source/files[0])
    delays, db = mat['Delay'].flatten(), mat['R'].flatten()
    if not np.allclose(delays*10000, np.round(delays*10000), rtol=0, atol=1e-8):
        raise ValueError('源 PDP 不满足当前审计的 100 微秒时延网格。')
    protocol = dict(vars(args), role='development_physical_audit',
        source_sha256=source_hashes, config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope='独立 profile 抽样与理想扩频等效检查，不执行通信 BER 实验，不作为确认')
    reserve_seeds(ROOT, experiment=args.experiment, seeds=args.seeds+[args.equivalence_seed],
                  purpose='development', details=protocol)
    matrix = []
    for seed in args.seeds:
        env = build_comm_env_config(config, level='B', snr_db=0, seed=seed, state_split='offline_train')
        profile = sample_eme_profile(EMEChannelProfileConfig(level=env.level,
            sample_rate_hz=env.sample_rate_hz, symbol_rate_hz=env.symbol_rate_hz, frame_len=env.frame_len,
            strong_path_count=env.strong_path_count, diffuse_energy_ratio=env.diffuse_energy_ratio,
            seed=seed, max_delay_seconds=env.max_delay_seconds))
        matrix.append(np.abs(profile.cir)**2)
    mean_power = np.mean(matrix, axis=0)
    current_delays = np.arange(mean_power.size)/config['symbol_rate_hz']
    selected = np.arange(96)[::3]
    result = dict(protocol=protocol, status='completed', first_point=dict(
        delay_seconds=delays.tolist(), average_path_gains_db=db.tolist(),
        interpretation='MATLAB RicianChannel AveragePathGains 为功率 dB，按 10^(R/10) 转换后归一化',
        all100=pdp_statistics(delays, 10**(db/10)),
        rls_script_selected32=pdp_statistics(delays[selected], 10**(db[selected]/10))),
        current=dict(delay_seconds=current_delays.tolist(), mean_initial_power=mean_power.tolist(),
                     profiles=len(matrix), statistics=pdp_statistics(current_delays, mean_power)),
        ideal_despreading=equivalence_check(args.equivalence_seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temporary.replace(output)
    print(json.dumps({key:result[key] if key=='ideal_despreading' else
        result[key].get('statistics', {name:result[key][name] for name in ['all100', 'rls_script_selected32']} if key=='first_point' else {})
        for key in ['first_point', 'current', 'ideal_despreading']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
