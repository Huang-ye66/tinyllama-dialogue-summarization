"""Summarize paired long-dialogue metrics, gates, and confidence intervals."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(os.environ.get("TINYLLAMA_PROJECT_ROOT", ROOT)).expanduser().resolve()
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
def main():
 p=argparse.ArgumentParser();p.add_argument("--split",choices=["val","test"],required=True);a=p.parse_args()
 d=json.loads((RUN/f"reports/long_dialogue/eval-{a.split}.json").read_text(encoding="utf-8"));r=d["records"];n=len(r)
 def mean(mode,path):return sum(x[mode][path[0]][path[1]] for x in r)/n
 dl=np.array([x["hierarchical"]["rouge"]["rougeL"]-x["direct"]["rouge"]["rougeL"] for x in r]);dr=np.array([x["hierarchical"]["rouge"]["rouge1_recall"]-x["direct"]["rouge"]["rouge1_recall"] for x in r]);dc=np.array([x["hierarchical"]["error_proxy"]["critical"]-x["direct"]["error_proxy"]["critical"] for x in r])
 rng=np.random.default_rng(42);idx=rng.integers(0,n,size=(10000,n));base_critical=mean("direct",("error_proxy","critical"));relative=dc.mean()/base_critical if base_critical else 0
 engineering=all(x["hierarchical"]["input_coverage"]==1 and x["hierarchical"]["summary"] and x["hierarchical"]["peak_memory_gb"]<=5.5 for x in r) and dl.mean()>=-.01
 quality=dr.mean()>=.03 or relative<=-.05
 result={"state":"complete","split":a.split,"count":n,"groups":{k:sum(x["eval_type"]==k for x in r) for k in ("real_long","synthetic_stress")},"direct":{"rougeL":mean("direct",("rouge","rougeL")),"rouge1_recall":mean("direct",("rouge","rouge1_recall")),"critical":base_critical,"mean_seconds":sum(x["direct"]["total_seconds"] for x in r)/n},"hierarchical":{"rougeL":mean("hierarchical",("rouge","rougeL")),"rouge1_recall":mean("hierarchical",("rouge","rouge1_recall")),"critical":mean("hierarchical",("error_proxy","critical")),"mean_seconds":sum(x["hierarchical"]["total_seconds"] for x in r)/n,"max_peak_memory_gb":max(x["hierarchical"]["peak_memory_gb"] for x in r),"min_coverage":min(x["hierarchical"]["input_coverage"] for x in r)},"delta":{"rougeL":float(dl.mean()),"rougeL_ci95":[float(x) for x in np.quantile(dl[idx].mean(1),[.025,.975])],"rouge1_recall":float(dr.mean()),"critical":float(dc.mean()),"critical_relative":float(relative)},"decision":{"engineering_gate_passed":bool(engineering),"quality_gate_passed":bool(engineering and quality),"rules":{"rougeL_guardrail":">= -0.01","recall_gain":">= 0.03 OR","critical_relative":"<= -0.05"}},"bootstrap_samples":10000,"test_split_read":a.split=="test"}
 save(RUN/f"reports/long_dialogue/summary-{a.split}.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
