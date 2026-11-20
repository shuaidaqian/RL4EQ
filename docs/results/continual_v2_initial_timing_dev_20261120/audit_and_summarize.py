# -*- coding: utf-8 -*-
"""从仓库根目录复核已完成的P7数据并重建诊断汇总，不运行实验或修改种子登记。"""

import json,hashlib,math
from pathlib import Path
from collections import defaultdict
from scipy.stats import t
out=Path("docs/results/continual_v2_initial_timing_dev_20261120")
p=out/"diagnostics_3x64.json"
r=json.loads(p.read_text(encoding="utf-8"))
mp=Path("docs/protocols/continual_v2_initial_timing_diagnostics_dev_20261120.json")
m=json.loads(mp.read_text(encoding="utf-8"))
assert r["status"]=="completed" and r["row_count"]==3840
assert r["protocol"]["stream_rows_schema_version"]==2
assert r["protocol"]["seeds"]==m["seeds"] and r["protocol"]["snrs"]==m["snrs"] and r["protocol"]["frames"]==64
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(mp)==r["protocol"]["manifest_sha256"]
assert sha(r["protocol"]["config"])==r["protocol"]["config_sha256"]
for path,digest in r["protocol"]["sources"].items(): assert sha(path)==digest,path
for path,digest in r["protocol"]["checkpoint_sha256"].items(): assert sha(path)==digest
arms=[v["name"] for v in m["variants"]]
expected={(s,k,f) for s in m["snrs"] for k in m["seeds"] for f in range(64)}
groups=defaultdict(dict)
rows=0
with (out/r["rows_file"]).open(encoding="utf-8") as stream:
 for line in stream:
  row=json.loads(line); key=(row["snr_db"],row["seed"],row["frame_index"]); arm=row["arm"]
  assert key in expected and arm in arms
  assert arm not in groups[key],("duplicate",key,arm)
  assert type(row["bits"]) is int and row["bits"]==768 and type(row["errors"]) is int and 0<=row["errors"]<=768
  assert set(m["diagnostics"])<=set(row)
  assert row["pilot_observations"]==108
  assert row["model_calls"]==1 and row["neural_forwards"]==0 and row["data_updates"]==0
  assert row["pilot_updates"]==int(arm!="acquisition_known_adapt_real_physics")
  for k in ("prior_covariance_trace","decode_covariance_trace","posterior_covariance_trace","pilot_innovation_mse","pilot_innovation_variance_mean","noise_variance_used"):
   assert isinstance(row[k],(int,float)) and math.isfinite(row[k]) and row[k]>=0,(key,arm,k)
  assert row["noise_variance_used"]>0 and row["pilot_innovation_variance_mean"]>0
  assert abs(row["posterior_covariance_trace"]-row["decode_covariance_trace"])<1e-6
  assert row["decode_covariance_trace"]<=row["prior_covariance_trace"]+1e-6
  assert not set(row)&{"bits_array","data_labels","true_cir","true_cfo","true_phase","tx_symbols"}
  groups[key][arm]=row; rows+=1
assert rows==3840 and set(groups)==expected
for key,group in groups.items():
 assert set(group)==set(arms)
 assert len({v["rx_sha256"] for v in group.values()})==1
 assert len({v["noise_variance_used"] for v in group.values()})==1
acq=[(a["snr_db"],a["seed"]) for a in r["acquisitions"]]
assert len(acq)==12 and len(set(acq))==12 and set(acq)=={(s,k) for s in m["snrs"] for k in m["seeds"]}
def ci(v):
 n=len(v); mean=sum(v)/n
 half=float(t.ppf(.975,n-1))*math.sqrt(sum((x-mean)**2 for x in v)/(n-1)/n)
 return [mean,mean-half,mean+half]
metrics=["prior_covariance_trace","decode_covariance_trace","pilot_innovation_mse","pilot_innovation_variance_mean","elapsed_ms"]
snrs={}
for snr in m["snrs"]:
 windows=[]
 for start,end in [*m["pre_registered_windows"],[0,64]]:
  arm_values={a:[] for a in arms}
  diagnostic={a:{k:[] for k in metrics} for a in arms}
  for seed in m["seeds"]:
   for arm in arms:
    values=[groups[(snr,seed,f)][arm] for f in range(start,end)]
    arm_values[arm].append(sum(v["errors"] for v in values)/sum(v["bits"] for v in values))
    for field in metrics: diagnostic[arm][field].append(sum(v[field] for v in values)/len(values))
  gains={"|".join(pair):[x-y for x,y in zip(arm_values[pair[0]],arm_values[pair[1]])] for pair in m["pairs"]}
  windows.append(dict(start=start,end_exclusive=end,arm_ber_mean_ci95_t={a:ci(v) for a,v in arm_values.items()},pair_gain_mean_ci95_t={k:ci(v) for k,v in gains.items()},
   session_ber=arm_values,session_pair_gains=gains,diagnostic_mean_ci95_t={a:{k:ci(v) for k,v in values.items()} for a,values in diagnostic.items()}))
 first=[]
 dynamic=r["protocol"]["variants"][-1]
 rho=dynamic["state_rho"]; power=sum(dynamic["stationary_power_by_tap"])
 for seed in m["seeds"]:
  old=groups[(snr,seed,0)]["recursive_pilot_ar1_initial0"]; fixed=groups[(snr,seed,0)]["recursive_pilot_ar1"]
  predicted_delta=(rho*rho-1)*old["prior_covariance_trace"]+(1-rho*rho)*power
  actual_delta=fixed["prior_covariance_trace"]-old["prior_covariance_trace"]
  assert math.isclose(actual_delta,predicted_delta,abs_tol=1e-6)
  first.append(dict(seed=seed,predicted_prior_trace_delta=predicted_delta,actual_prior_trace_delta=actual_delta,old_errors=old["errors"],fixed_errors=fixed["errors"]))
 snrs[str(snr)]=dict(windows=windows,first_frame_transition_check=first)
audit=dict(status="completed_development_audit",sessions=12,rows=3840,paired_frames=768,session_arm_groups=60,frames_per_group=64,schema_version=2,pilot_observations=108,
 duplicates=0,missing=0,hash_mismatches=0,invalid_fields=0,manifest_sha256=sha(mp),result_sha256=sha(p),rows_sha256=sha(out/r["rows_file"]),
 confidence_intervals="跨3个独立会话的未校正描述性95% t区间；未定义等价容差，不作无影响或饱和结论",
 snrs=snrs)
(out/"diagnostics_3x64_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

print("P7全部3840行、768个配对帧及schema 2诊断审计通过，已生成diagnostics_3x64_audit.json。")

