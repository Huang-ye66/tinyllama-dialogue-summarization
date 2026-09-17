"""Compare direct truncation and hierarchical generation on a prepared split."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(os.environ.get("TINYLLAMA_PROJECT_ROOT", ROOT)).expanduser().absolute()
sys.path.insert(0,str(RUN))
from rouge_score import rouge_scorer
from litgpt import LLM
from scripts.core.reliability import speakers,NUMBER_RE
from scripts.long_dialogue.hierarchical import summarize_dialogue

def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
def score(row,pred,scorer):
 rs=scorer.score(row["output"],pred);people=speakers(row["input"])
 ref_people={x.casefold() for x in people if x.casefold() in row["output"].casefold()};pred_people={x.casefold() for x in people if x.casefold() in pred.casefold()}
 ref_num=set(NUMBER_RE.findall(row["output"]));pred_num=set(NUMBER_RE.findall(pred));om=max(0.0,1-rs["rouge1"].recall)
 return {"rouge":{k:rs[k].fmeasure for k in ("rouge1","rouge2","rougeL")}|{"rouge1_recall":rs["rouge1"].recall},"error_proxy":{"speaker":int(bool(ref_people.symmetric_difference(pred_people))),"number":int(bool(ref_num.symmetric_difference(pred_num))),"omission":om,"critical":int(bool(ref_people.symmetric_difference(pred_people)))+int(bool(ref_num.symmetric_difference(pred_num)))+om}}
def main():
 p=argparse.ArgumentParser();p.add_argument("--split",choices=["val","test"],required=True);a=p.parse_args()
 rows=json.loads((RUN/f"data/long_dialogue/{a.split}/combined.json").read_text(encoding="utf-8"));out=RUN/f"reports/long_dialogue/eval-{a.split}.json"
 previous=json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"records":[]};records=previous.get("records",[]);done={x["source_index"] for x in records}
 llm=LLM.load(str(RUN/"outputs/models/resource_optimized_r8/final"),distribute=None);llm.distribute(accelerator="cuda",precision="bf16-true");llm.model.eval();scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
 started=time.perf_counter()
 for i,row in enumerate(rows,1):
  if row["source_index"] in done:continue
  direct=summarize_dialogue(llm,row["input"],"direct");hier=summarize_dialogue(llm,row["input"],"hierarchical")
  record={"source_index":row["source_index"],"eval_type":row["eval_type"],"input_tokens":row["input_tokens"],"dialogue":row["input"],"reference":row["output"],"direct":direct|score(row,direct["summary"],scorer),"hierarchical":hier|score(row,hier["summary"],scorer)}
  records.append(record);save(out,{"split":a.split,"records":records,"complete":False,"test_split_read":a.split=="test"});print(f"{len(records)}/{len(rows)}",flush=True)
 result={"split":a.split,"count":len(records),"records":records,"complete":len(records)==len(rows),"seconds_this_run":time.perf_counter()-started,"test_split_read":a.split=="test"}
 save(out,result)
if __name__=="__main__":main()
