"""只读统计分析：Level B 长回波逐帧日志的统计功效与错误构成。

不训练、不修改现有源码、不读取 Data 标签用于任何控制。
统计单位是独立 session，而不是帧或 bit。
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

N_BITS = 768  # 1024 符号帧 - 256 prefix Pilot = 768 Data BPSK 符号
ALPHA = 0.05
POWER = 0.80
MAX_N = 2_000_000

REPO = Path(__file__).resolve().parents[3]
MAIN = Path("D:/Research/RL4EQ/logs/eme_pilot256_bandit_pilotsparse_main_60f_3s_final/frame_metrics.jsonl")

DATASETS = [
    ("main_60f", MAIN, "method"),
    ("adapt_replay_16f", REPO / "docs/results/adapt_replay_dev_16f_v1_20261106/frame_metrics.jsonl", "branch"),
    ("cumulative_radius_16f", REPO / "docs/results/cumulative_radius_dev_16f_v1_20261106/frame_metrics.jsonl", "branch"),
    ("persistent_adam_16f", REPO / "docs/results/persistent_adam_dev_16f_v1_20261106/frame_metrics.jsonl", "branch"),
    ("persistent_adam_32f", REPO / "docs/results/persistent_adam_dev_32f_v1_20261106/frame_metrics.jsonl", "branch"),
    ("strong_transfer_8f", REPO / "docs/results/strong_checkpoint_pilot_transfer_meta32_20261105/frame_metrics.jsonl", "branch"),
    ("trajectory_16f", REPO / "docs/results/trajectory_transfer_dev_16f_v1_20261105/frame_metrics.jsonl", "branch"),
    ("trajectory_or_16f", REPO / "docs/results/trajectory_transfer_or_gate_dev_16f_v1_20261106/frame_metrics.jsonl", "branch"),
    ("rls_32f", REPO / "docs/results/strong_offline_rls_onewindow_32f_dev_20261105/frame_metrics.jsonl", "method"),
    ("rls_64f", REPO / "docs/results/strong_offline_rls_onewindow_64f_dev_20261105/frame_metrics.jsonl", "method"),
]

# 只做补充核查，不并入主功效池：这些来源大多在 10/15 dB 没有接受更新（Δ 恒 0），
# 且 121xxx/761xx 种子族在多个版本间复用，不能当作新的独立 session。
EXTRA_DATASETS = [
    ("v27_32f", REPO / "docs/results/detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1_20261105/frame_metrics.jsonl.gz", "gz_branch"),
    ("full_head_softber_eval16", REPO / "docs/results/full_head_softber_train32_v1_20261106/eval16.json", "eval_rows"),
    ("full_head_softber_eval32_seed76200", REPO / "docs/results/full_head_softber_train32_v1_20261106/eval32_seed76200.json", "eval_rows"),
    ("full_head_adapt_replay_h4", REPO / "docs/results/full_head_adapt_replay_h4_dev_v1_20261106/eval.json", "eval_rows"),
    ("full_head_radius2", REPO / "docs/results/full_head_radius2_dev_v1_20261106/eval.json", "eval_rows"),
    ("full_head_supervised_train32", REPO / "docs/results/full_head_supervised_train32_v1_20261106/eval16.json", "eval_rows"),
    ("full_head_trajectory_train32", REPO / "docs/results/full_head_trajectory_train32_v1_20261106/eval16.json", "eval_rows"),
]


def load_dataset(name, path, kind):
    if not path.exists():
        return None, {"name": name, "path": str(path), "available": False}
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    records = []
    if kind == "method":
        frozen = {(r["snr_db"], r["seed"], r["frame"]): r for r in rows if r["method"] == "Frozen Offline NN"}
        online = {(r["snr_db"], r["seed"], r["frame"]): r for r in rows if r["method"] == "Pilot-Driven Online Adaptation"}
        keys = sorted(set(frozen) & set(online))
        for key in keys:
            records.append({
                "snr": float(key[0]), "seed": str(key[1]), "arm": "method",
                "frame": int(key[2]),
                "ef": int(round(frozen[key]["ber_data"] * N_BITS)),
                "eo": int(round(online[key]["ber_data"] * N_BITS)),
            })
        meta = {"name": name, "path": str(path), "available": True, "kind": kind,
                "pairs": len(keys), "frames": int(max(r["frame"] for r in rows))}
    else:
        for r in rows:
            if "ber_frozen" not in r or "ber_online" not in r:
                continue
            records.append({
                "snr": float(r["snr_db"]), "seed": str(r["seed"]), "arm": str(r["branch"]),
                "frame": int(r["frame"]),
                "ef": int(round(r["ber_frozen"] * N_BITS)),
                "eo": int(round(r["ber_online"] * N_BITS)),
            })
        meta = {"name": name, "path": str(path), "available": True, "kind": kind,
                "pairs": len(records), "frames": int(max(r["frame"] for r in rows)) + 1}
    meta["snrs"] = sorted(set(r["snr"] for r in records))
    meta["seeds"] = sorted(set(r["seed"] for r in records))
    meta["arms"] = sorted(set(r["arm"] for r in records))
    return records, meta


def load_extra(name, path, kind):
    """补充来源核查：返回 (snr, seed, arm, frame, ef, eo) 记录。"""
    if not path.exists():
        return []
    if kind == "gz_branch":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        out = []
        for r in rows:
            if "ber_frozen" not in r or "ber_online" not in r:
                continue
            arm = r.get("training") or r.get("arm") or "arm"
            out.append((float(r["snr_db"]), str(r["seed"]), str(arm), int(r["frame"]),
                        int(round(r["ber_frozen"] * N_BITS)), int(round(r["ber_online"] * N_BITS))))
        return out
    if kind == "eval_rows":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows", []) if isinstance(data, dict) else []
        out = []
        for r in rows:
            if r.get("branch") == "frozen":
                continue
            if "ber_frozen" not in r or "ber_online" not in r:
                continue
            out.append((float(r["snr_db"]), str(r["seed"]), str(r.get("branch")), int(r["frame_index"]),
                        int(round(r["ber_frozen"] * N_BITS)), int(round(r["ber_online"] * N_BITS))))
        return out
    return []


def summarize_extra(records):
    """按 SNR 汇总补充来源的逐帧 Δ 与逐 session 方差。"""
    by = defaultdict(list)
    for snr, seed, arm, frame, ef, eo in records:
        by[snr].append((seed, arm, frame, ef, eo))
    out = {}
    for snr, recs in sorted(by.items()):
        d = np.array([eo - ef for _, _, _, ef, eo in recs], float)
        sess = defaultdict(list)
        for seed, arm, frame, ef, eo in recs:
            sess[(seed, arm)].append(eo - ef)
        avars = [float(np.var(v, ddof=1)) for v in sess.values() if len(v) > 1 and np.var(v, ddof=1) > 0]
        F = int(max(len(v) for v in sess.values()))
        mean_var = float(np.mean(avars)) if avars else None
        out[str(snr)] = {
            "frames": int(len(d)),
            "sessions": len(sess),
            "frames_per_session": F,
            "mean_delta_ber": float(d.mean() / N_BITS),
            "fraction_zero": float(np.mean(d == 0)),
            "fraction_abs_le2": float(np.mean(np.abs(d) <= 2)),
            "max_abs_errors": int(np.abs(d).max()),
            "active_session_frame_var_errors2": avars,
            "mean_active_frame_var_errors2": mean_var,
            "mde_n3_using_active_var": mde_ber(3, mean_var / F) if mean_var is not None else None,
            "required_n_1e-5_using_active_var": required_n(1e-5, mean_var / F) if mean_var is not None else None,
        }
    return out


def quantiles(x, qs=(0, 5, 25, 50, 75, 95, 100)):
    if len(x) == 0:
        return {}
    return {f"q{q}": float(np.percentile(x, q)) for q in qs}


def power_t(n, effect_errors, var_session):
    """双侧 alpha=0.05、n 个独立 session 的配对 t 检验功效；effect 为每帧平均错误数变化。"""
    if effect_errors == 0:
        return ALPHA
    if var_session <= 0:
        return 1.0
    df = n - 1
    nc = effect_errors * math.sqrt(n) / math.sqrt(var_session)
    tcrit = stats.t.ppf(1 - ALPHA / 2, df)
    return float(1 - stats.nct.cdf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc))


def mde_ber(n, var_session):
    """给定 n 与 session 级方差，返回 80% power 的最小可检出 ΔBER。"""
    if var_session <= 0:
        return 0.0
    lo, hi = 0.0, 1000.0  # errors/frame
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_t(n, mid, var_session) < POWER:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 / N_BITS


def required_n(delta_ber, var_session):
    """给定 ΔBER 与 session 级方差，返回达到 80% power 所需独立 session 数。"""
    effect = delta_ber * N_BITS
    if effect <= 0 or var_session <= 0:
        return 0
    if power_t(MAX_N, effect, var_session) < POWER:
        return None
    lo, hi = 2, MAX_N
    while lo < hi:
        mid = (lo + hi) // 2
        if power_t(mid, effect, var_session) < POWER:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    all_records = []
    dataset_meta = []
    for name, path, kind in DATASETS:
        records, meta = load_dataset(name, path, kind)
        dataset_meta.append(meta)
        if records:
            for r in records:
                r["dataset"] = name
            all_records.extend(records)

    extra_summary = {}
    for name, path, kind in EXTRA_DATASETS:
        recs = load_extra(name, path, kind)
        extra_summary[name] = {
            "path": str(path),
            "available": bool(recs),
            "role": "supplementary_check_not_pooled",
            "by_snr": summarize_extra(recs) if recs else {},
        }

    # 按 session 聚合，并显式按 frame 排序
    sessions_raw = defaultdict(lambda: [])
    for r in all_records:
        key = (r["dataset"], r["snr"], r["arm"], r["seed"])
        sessions_raw[key].append(r)
    sessions = {}
    for key, recs in sessions_raw.items():
        recs = sorted(recs, key=lambda x: x["frame"])
        sessions[key] = {
            "ef": np.asarray([x["ef"] for x in recs], float),
            "d": np.asarray([x["eo"] - x["ef"] for x in recs], float),
        }

    by_snr = {}
    for snr in [0.0, 5.0, 10.0, 15.0]:
        per_dataset = []
        within_d_all = []   # (var, df)
        within_d_active = []  # 排除 d 恒为 0 的退化 session
        within_ef_all = []
        session_rows = []
        for name, _, _ in DATASETS:
            ss = {k: v for k, v in sessions.items() if k[0] == name and k[1] == snr}
            if not ss:
                continue
            e_means, d_means, d_vars, ef_vars = [], [], [], []
            F = 0
            for (ds, s, arm, seed), v in sorted(ss.items()):
                F = len(v["d"])
                e_means.append(float(v["ef"].mean() / N_BITS))
                d_means.append(float(v["d"].mean() / N_BITS))
                dv = float(np.var(v["d"], ddof=1)) if F > 1 else 0.0
                ev = float(np.var(v["ef"], ddof=1)) if F > 1 else 0.0
                d_vars.append(dv)
                ef_vars.append(ev)
                within_d_all.append((dv, max(F - 1, 1)))
                within_ef_all.append((ev, max(F - 1, 1)))
                if dv > 0:
                    within_d_active.append((dv, max(F - 1, 1)))
                session_rows.append({
                    "dataset": name, "snr_db": snr, "seed": seed, "arm": arm,
                    "frames": F, "ber_frozen": e_means[-1],
                    "ber_online": float((v["ef"] + v["d"]).mean() / N_BITS),
                    "mean_delta_ber": d_means[-1],
                    "var_frame_delta_errors2": dv,
                    "var_frame_frozen_errors2": ev,
                })
            # τ²：同一 dataset/arm 的 3 个种子；先按 arm 算，再在 dataset 内平均
            arm_tau = []
            for arm in sorted(set(k[2] for k in ss)):
                means = [float(v["d"].mean()) for (ds, s, a, seed), v in ss.items() if a == arm]
                vars_ = [float(np.var(v["d"], ddof=1)) for (ds, s, a, seed), v in ss.items() if a == arm]
                if len(means) >= 3:
                    F_arm = len(next(v for (ds, s, a, seed), v in ss.items() if a == arm)["d"])
                    between = float(np.var(means, ddof=1))
                    arm_tau.append(max(0.0, between - float(np.mean(vars_)) / F_arm))
            per_dataset.append({
                "dataset": name, "frames": F, "sessions": len(ss),
                "ber_frozen_mean": float(np.mean(e_means)),
                "session_ber_frozen_var": float(np.var(e_means, ddof=1)) if len(e_means) > 1 else None,
                "session_ber_frozen_sd": float(np.std(e_means, ddof=1)) if len(e_means) > 1 else None,
                "session_delta_mean": float(np.mean(d_means)),
                "session_delta_sd": float(np.std(d_means, ddof=1)) if len(d_means) > 1 else None,
                "frame_delta_var_mean": float(np.mean(d_vars)),
                "tau2_delta_mom": float(np.mean(arm_tau)) if arm_tau else None,
                "tau2_delta_mom_arm_values": [float(x) for x in arm_tau],
            })

        def pooled_var(pairs):
            if not pairs:
                return None
            vals = np.array([v for v, df in pairs], float)
            dfs = np.array([df for v, df in pairs], float)
            return float((vals * dfs).sum() / dfs.sum())

        sigma2_typical = pooled_var(within_d_active) if within_d_active else pooled_var(within_d_all)
        sigma2_active = sigma2_typical
        active_vars = np.array([v for v, df in within_d_active], float)
        sigma2_low = float(np.percentile(active_vars, 10)) if len(active_vars) else sigma2_active
        sigma2_median = float(np.median(active_vars)) if len(active_vars) else sigma2_active
        sigma2_high = float(np.percentile(active_vars, 90)) if len(active_vars) else sigma2_active
        sigma2_frozen = pooled_var(within_ef_all)
        p_frozen = float(np.mean([r["ber_frozen"] for r in session_rows]))
        sigma2_binom = N_BITS * p_frozen * (1 - p_frozen)
        tau_all = [r["tau2_delta_mom"] for r in per_dataset if r["tau2_delta_mom"] is not None]
        tau2_typical = float(np.median(tau_all)) if tau_all else 0.0
        tau2_upper = float(np.percentile(tau_all, 90)) if tau_all else 0.0
        # 主矩阵自身（10/15 dB 有实际在线更新）的帧级 Δ 方差
        main_active_vars = [float(np.var(v["d"], ddof=1)) for k, v in sessions.items()
                            if k[0] == "main_60f" and k[1] == snr and np.var(v["d"], ddof=1) > 0]
        sigma2_main = float(np.mean(main_active_vars)) if main_active_vars else None

        # 功效表
        scenarios = {
            "empirical_typical": (sigma2_typical, 0.0),
            "empirical_median": (sigma2_median, 0.0),
            "empirical_gentle_p10": (sigma2_low, 0.0),
            "empirical_aggressive_p90": (sigma2_high, 0.0),
            "binomial_reference": (sigma2_binom, 0.0),
            "empirical_typical_plus_tau": (sigma2_typical, tau2_typical),
            "empirical_typical_plus_tau_upper": (sigma2_typical, tau2_upper),
        }
        if sigma2_main is not None:
            scenarios["main_matrix_empirical"] = (sigma2_main, 0.0)
        mde_tables = {}
        for scen, (s2, t2) in scenarios.items():
            table = {}
            for F in [16, 32, 60, 128]:
                row = {}
                for n in [3, 5, 10, 20, 50, 100]:
                    row[str(n)] = mde_ber(n, s2 / F + t2)
                table[str(F)] = row
            mde_tables[scen] = table

        targets = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        req_tables = {}
        req_scenarios = ["empirical_gentle_p10", "empirical_median", "empirical_typical",
                         "empirical_aggressive_p90", "binomial_reference"]
        if sigma2_main is not None:
            req_scenarios.append("main_matrix_empirical")
        for scen in req_scenarios:
            s2, t2 = scenarios[scen]
            req_tables[scen] = {}
            for F in [16, 32, 60, 128]:
                req_tables[scen][str(F)] = {f"{d:.0e}": required_n(d, s2 / F + t2) for d in targets}

        # 用每个实验自身的 n、F、帧级 Δ 方差，评估它对该实验已观测 Δ 的功效
        observed_power = []
        for row in per_dataset:
            n_sess = row["sessions"]
            F = row["frames"]
            var_session = row["frame_delta_var_mean"] / F
            effect = abs(row["session_delta_mean"]) * N_BITS
            observed_power.append({
                "dataset": row["dataset"], "snr_db": snr, "n_sessions": n_sess, "frames": F,
                "observed_delta_ber": row["session_delta_mean"],
                "power_to_detect_observed_effect": power_t(n_sess, effect, var_session),
            })

        # 主矩阵逐 SNR 的帧级/会话级分布
        main_rows = [r for r in session_rows if r["dataset"] == "main_60f"]
        main_ef = np.concatenate([
            v["ef"] for (ds, s, a, seed), v in sessions.items() if ds == "main_60f" and s == snr
        ]) if main_rows else np.array([])
        main_d = np.concatenate([
            v["d"] for (ds, s, a, seed), v in sessions.items() if ds == "main_60f" and s == snr
        ]) if main_rows else np.array([])
        error_stats = {}
        if len(main_ef):
            mean_e = float(main_ef.mean())
            var_e = float(np.var(main_ef, ddof=1))
            p_main = mean_e / N_BITS
            binom_main = N_BITS * p_main * (1 - p_main)
            error_stats = {
                "frames": int(len(main_ef)),
                "errors_per_frame_mean": mean_e,
                "errors_per_frame_var": var_e,
                "errors_per_frame_sd": math.sqrt(var_e),
                "errors_per_frame_quantiles": quantiles(main_ef),
                "zero_error_frame_fraction": float(np.mean(main_ef == 0)),
                "variance_over_mean": float(var_e / mean_e) if mean_e > 0 else None,
                "p_main": p_main,
                "design_effect_vs_main_binomial": float(var_e / binom_main) if binom_main > 0 else None,
                "design_effect_vs_pooled_binomial": float(var_e / sigma2_binom) if sigma2_binom > 0 else None,
                "delta_errors_distribution": {
                    str(int(k)): float(v) for k, v in zip(*np.unique(main_d, return_counts=True))
                },
                "session_ber_frozen": [float((sessions[k]["ef"].mean() / N_BITS)) for k in sessions
                                       if k[0] == "main_60f" and k[1] == snr],
                "session_delta_ber": [float((sessions[k]["d"].mean() / N_BITS)) for k in sessions
                                      if k[0] == "main_60f" and k[1] == snr],
            }
        by_snr[str(snr)] = {
            "p_frozen": p_frozen,
            "sigma2_frozen_frame_errors2": sigma2_frozen,
            "sigma2_binomial_floor_errors2": sigma2_binom,
            "sigma2_delta_typical_errors2": sigma2_typical,
            "sigma2_delta_active_typical_errors2": sigma2_active,
            "sigma2_delta_median_errors2": sigma2_median,
            "sigma2_delta_gentle_p10_errors2": sigma2_low,
            "sigma2_delta_aggressive_p90_errors2": sigma2_high,
            "sigma2_main_matrix_errors2": sigma2_main,
            "tau2_delta_typical_errors2": tau2_typical,
            "tau2_delta_upper90_errors2": tau2_upper,
            "min_nonzero_session_delta_ber": {str(F): 1.0 / (N_BITS * F) for F in [16, 32, 60, 128]},
            "session_net_delta_error_sd_by_F": {
                "empirical_median": {str(F): math.sqrt(sigma2_median * F) for F in [16, 32, 60, 128]},
                "empirical_typical": {str(F): math.sqrt(sigma2_typical * F) for F in [16, 32, 60, 128]},
                "binomial_reference": {str(F): math.sqrt(sigma2_binom * F) for F in [16, 32, 60, 128]},
            },
            "target_1e-5_net_errors_by_F": {str(F): 1e-5 * N_BITS * F for F in [16, 32, 60, 128]},
            "mde_existing_main_design_n3_f60": (mde_ber(3, sigma2_main / 60) if sigma2_main is not None else None),
            "mde_existing_dev_design_n3_f16_typical": mde_ber(3, sigma2_typical / 16),
            "mde_existing_dev_design_n3_f32_typical": mde_ber(3, sigma2_typical / 32),
            "power_mde_ber": mde_tables,
            "required_sessions": req_tables,
            "observed_power_of_each_experiment": observed_power,
            "per_dataset": per_dataset,
            "main_matrix_error_composition": error_stats,
        }

    # 汇总所有可用配对实验的帧级 Δ 分布
    delta_composition = {}
    for snr in [0.0, 5.0, 10.0, 15.0]:
        d = np.concatenate([v["d"] for k, v in sessions.items() if k[1] == snr])
        if not len(d):
            continue
        counts = {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))}
        delta_composition[str(snr)] = {
            "frames": int(len(d)),
            "fraction_zero": float(np.mean(d == 0)),
            "fraction_abs1": float(np.mean(np.abs(d) == 1)),
            "fraction_abs_le2": float(np.mean(np.abs(d) <= 2)),
            "fraction_abs_le5": float(np.mean(np.abs(d) <= 5)),
            "max_abs": int(np.abs(d).max()),
            "counts": counts,
        }

    # 历史 error_geometry 只读聚合（不同 checkpoint/运行，仅作上下文，不可与本矩阵混合）
    legacy = None
    legacy_path = REPO / "docs/results/error_geometry_stage_20261101.json"
    if legacy_path.exists():
        g = json.load(legacy_path.open(encoding="utf-8"))
        legacy = {
            "source": str(legacy_path),
            "role": "historical_diagnostic_different_checkpoint_do_not_pool",
            "checkpoint": g.get("checkpoint"),
            "seeds": g.get("seeds"),
            "frames": g.get("frames"),
            "per_snr_data_regions": [
                {
                    "snr_db": r["snr_db"], "region": r["region"], "bits": r["bits"],
                    "errors": r["errors"], "mean_wrong_abs_logit": r.get("mean_wrong_abs_logit"),
                    "wrong_abs_logit_ge_5": r.get("wrong_abs_logit_ge_5"),
                    "wrong_abs_logit_ge_20": r.get("wrong_abs_logit_ge_20"),
                }
                for r in g.get("per_region", [])
                if r.get("region") in ("data_head", "data_middle", "data_tail")
            ],
        }

    out = {
        "schema_version": "power-error-composition-v1",
        "role": "read_only_cpu_statistical_analysis",
        "data_labels_used_for_control": False,
        "training_performed": False,
        "gpu_used": False,
        "n_bits_per_frame": N_BITS,
        "alpha_two_sided": ALPHA,
        "target_power": POWER,
        "statistical_unit": "independent session（同一 seed/session 内的帧只降低会话内噪声，不能替代独立 session）",
        "datasets": dataset_meta,
        "additional_paired_sources_not_pooled": extra_summary,
        "per_snr": by_snr,
        "delta_composition_all_datasets": delta_composition,
        "legacy_error_geometry_context": legacy,
        "available": {
            "per_frame_error_counts": "由 BER_data*768 精确恢复；已报告均值/方差/分位/零错误比例/方差均值比/设计效应",
            "session_level_paired_delta": "由逐帧 Frozen/Online BER 计算；统计单位为独立 session",
            "mde_and_required_sessions": "非中心 t、alpha=0.05 双侧、80% power；按多个帧级方差场景报告",
        },
        "unavailable": {
            "per_symbol_frozen_logits": "现有 frame_metrics.jsonl 只保存逐帧 BER，无逐符号 logit；docs/results 与 D:/Research/RL4EQ/logs 中未找到逐符号 logit 数组",
            "wrong_logit_threshold_fractions_0p1_0p5_1p0_2p0": "不可用：无逐符号 logit，无法计算阈值占比",
            "error_symbol_spatial_histogram": "不可用：无逐符号错误位置",
            "error_run_length_distribution": "不可用：无逐符号判决序列",
        },
        "notes": [
            "ΔBER 定义为 BER_online - BER_frozen；正值表示在线更差。",
            "每帧最小非零 BER 变化为 1/768=0.001302；session 级最小非零变化为 1/(768*F)。",
            "设计效应 = 逐帧错误数方差 / (768*p*(1-p))；远大于 1 表示 bit 级二项置信区间严重过窄。",
            "binomial_reference 是未配对的 bit 级二项参考，不是严格下界；配对差方差可能低于或高于它。",
            "τ² 由 3 个种子按矩估计，极不稳定，仅作敏感性上界；主表默认 τ²=0。",
        ],
    }
    out_path = Path(__file__).with_name("power_error_composition.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_path)
    for snr in ["0.0", "5.0", "10.0", "15.0"]:
        s = out["per_snr"][snr]
        print(snr, "p=", round(s["p_frozen"], 5),
              "sig2 typ/lo/hi/binom=",
              round(s["sigma2_delta_typical_errors2"], 3),
              round(s["sigma2_delta_gentle_p10_errors2"], 3),
              round(s["sigma2_delta_aggressive_p90_errors2"], 3),
              round(s["sigma2_binomial_floor_errors2"], 3),
              "tau2 typ/up=", round(s["tau2_delta_typical_errors2"], 3), round(s["tau2_delta_upper90_errors2"], 3))
        for scen in ["binomial_reference", "empirical_gentle_p10", "empirical_median", "empirical_typical", "empirical_aggressive_p90"]:
            print("  ", scen, "n=3 F=32 MDE", round(s["power_mde_ber"][scen]["32"]["3"], 6),
                  "| n for 1e-4 F=128:", s["required_sessions"][scen]["128"]["1e-04"],
                  "| n for 1e-5 F=128:", s["required_sessions"][scen]["128"]["1e-05"])


if __name__ == "__main__":
    main()
