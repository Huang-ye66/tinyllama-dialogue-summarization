"""Prepare real and deterministic synthetic long-dialogue evaluation sets."""
from __future__ import annotations
import argparse,json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(os.environ.get("TINYLLAMA_PROJECT_ROOT", ROOT)).expanduser().absolute()
sys.path.insert(0,str(RUN))
from litgpt import LLM
from scripts.core.reliability import extract_facts

def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
def main():
 p=argparse.ArgumentParser();p.add_argument("--split",choices=["val","test"],required=True);a=p.parse_args()
 rows=json.loads((RUN/f"data/samsum/{a.split}.json").read_text(encoding="utf-8"))
 llm=LLM.load(str(RUN/"outputs/models/resource_optimized_r8/final"),distribute=None);tok=llm.preprocessor.tokenizer
 real=[{**x,"eval_type":"real_long","input_tokens":len(tok.encode(x["input"]))} for x in rows if len(tok.encode(x["input"]))>400]
 rng=random.Random(42);order=list(range(len(rows)));rng.shuffle(order);used=set();synthetic=[]
 for anchor in order:
  if len(synthetic)>=30:break
  chosen=[];names=set()
  for idx in [anchor]+order:
   if idx in used or idx in chosen:continue
   speakers={x.casefold() for x in extract_facts(rows[idx]["input"])["speakers"]}
   if not speakers or names&speakers:continue
   chosen.append(idx);names|=speakers
   if len(chosen)==3:break
  if len(chosen)<3:continue
  dialogue="\n\n".join(f"[Conversation {j+1}]\n{rows[idx]['input']}" for j,idx in enumerate(chosen))
  if len(tok.encode(dialogue))<=400:continue
  reference=" ".join(rows[idx]["output"] for idx in chosen)
  synthetic.append({"source_index":f"synthetic-{a.split}-{len(synthetic)+1:02d}","component_ids":[rows[idx]["source_index"] for idx in chosen],"instruction":"Summarize all three conversations in one concise paragraph.","input":dialogue,"output":reference,"eval_type":"synthetic_stress","input_tokens":len(tok.encode(dialogue))})
  used.update(chosen)
 if len(synthetic)!=30:raise RuntimeError(f"Only built {len(synthetic)} synthetic items")
 out=RUN/"data/long_dialogue"/a.split
 save(out/"real_long.json",real);save(out/"synthetic_stress.json",synthetic);save(out/"combined.json",real+synthetic)
 print(json.dumps({"split":a.split,"real_long":len(real),"synthetic":len(synthetic),"combined":len(real)+len(synthetic),"min_tokens":min(x["input_tokens"] for x in real+synthetic),"test_split_read":a.split=="test"},indent=2))
if __name__=="__main__":main()
