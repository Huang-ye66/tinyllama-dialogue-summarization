"""Finalize group analysis and a 30-item blinded human-AI review sheet."""
from __future__ import annotations
import csv,json,os,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(os.environ.get("TINYLLAMA_PROJECT_ROOT", ROOT)).expanduser().absolute()
REPORT=RUN/"reports/long_dialogue"
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
data=json.loads((REPORT/"eval-test.json").read_text(encoding="utf-8"));records=data["records"]
summary=json.loads((REPORT/"summary-test.json").read_text(encoding="utf-8"));rng_np=np.random.default_rng(42);breakdown={}
for kind in ("real_long","synthetic_stress"):
 part=[x for x in records if x["eval_type"]==kind];n=len(part)
 dl=np.array([x["hierarchical"]["rouge"]["rougeL"]-x["direct"]["rouge"]["rougeL"] for x in part]);dr=np.array([x["hierarchical"]["rouge"]["rouge1_recall"]-x["direct"]["rouge"]["rouge1_recall"] for x in part]);dc=np.array([x["hierarchical"]["error_proxy"]["critical"]-x["direct"]["error_proxy"]["critical"] for x in part]);base=np.mean([x["direct"]["error_proxy"]["critical"] for x in part]);idx=rng_np.integers(0,n,size=(10000,n))
 breakdown[kind]={"count":n,"rougeL_delta":float(dl.mean()),"rougeL_ci95":[float(x) for x in np.quantile(dl[idx].mean(1),[.025,.975])],"rouge1_recall_delta":float(dr.mean()),"critical_relative_change":float(dc.mean()/base)}
summary["by_group"]=breakdown;save(REPORT/"summary-test.json",summary)
real=sorted([x for x in records if x["eval_type"]=="real_long"],key=lambda x:x["input_tokens"],reverse=True)[:15]
syn=[x for x in records if x["eval_type"]=="synthetic_stress"];random.Random(42).shuffle(syn);chosen=real+syn[:15];random.Random(142).shuffle(chosen)
manifest=[];key={}
for i,row in enumerate(chosen,1):
 order=["direct","hierarchical"];random.Random(42+i).shuffle(order)
 item={"item_id":i,"source_index":row["source_index"],"eval_type":row["eval_type"],"dialogue":row["dialogue"],"reference":row["reference"],"candidate_a":row[order[0]]["summary"],"candidate_b":row[order[1]]["summary"],"a_factual_error":None,"a_person_error":None,"a_omission":None,"a_completeness":None,"b_factual_error":None,"b_person_error":None,"b_omission":None,"b_completeness":None,"best_candidate":None,"reviewer_notes":"","review_identity":"human-AI collaborative evaluation"}
 manifest.append(item);key[str(i)]={"candidate_a":order[0],"candidate_b":order[1]}
save(REPORT/"human_eval_blinded_30.json",{"review_identity":"human-AI collaborative evaluation","rubric":{"factual_error":"0 none, 1 minor, 2 major or fabricated","person_error":"0 correct, 1 ambiguous, 2 wrong attribution","omission":"0 no key omission, 1 minor omission, 2 core event omitted","completeness":"1 poor to 5 accurate, complete, and concise"},"items":manifest})
save(REPORT/"human_eval_key_30.json",{"seed":42,"key":key})
with (REPORT/"human_eval_blinded_30.csv").open("w",newline="",encoding="utf-8-sig") as h:
 w=csv.DictWriter(h,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
print(json.dumps({"group_analysis":breakdown,"blind_items":len(manifest),"real":sum(x["eval_type"]=="real_long" for x in manifest),"synthetic":sum(x["eval_type"]=="synthetic_stress" for x in manifest)},ensure_ascii=False,indent=2))
