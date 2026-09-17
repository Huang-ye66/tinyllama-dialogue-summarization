"""Regression smoke test for the built-in long-dialogue example."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from litgpt import LLM
from scripts.long_dialogue.hierarchical import summarize_dialogue

BLOCK=[
 "Alice: We need to plan Friday's product launch at 9:00 AM.",
 "Bob: I will prepare the customer list and contact 25 clients.",
 "Carol: The website still needs the payment page fixed before Thursday.",
 "Alice: Bob, please confirm the client list by Wednesday afternoon.",
 "Bob: I can finish it by 3:00 PM on Wednesday.",
 "David: I will test the payment page with Carol and record every failed case.",
 "Carol: The current problem affects card payments above $100.",
 "David: We should also test refunds and mobile browsers.",
 "Alice: Good. The launch remains Friday only if all critical tests pass.",
 "Bob: Marketing needs the final link by Thursday evening.",
 "Carol: I will send the fixed build at noon on Thursday.",
 "David: Testing will take about four hours after the build arrives.",
 "Alice: If a critical payment bug remains, we postpone the launch to Monday.",
 "Bob: I will tell marketing about the backup date.",
 "Carol: I also need access to the production logs.",
 "David: I can grant that access today.",
]
dialogue="\n".join(BLOCK*4)
root=Path(__file__).resolve().parents[2]
run_root=Path(os.environ.get("TINYLLAMA_PROJECT_ROOT", root)).expanduser().absolute()
checkpoint=run_root/"outputs/models/resource_optimized_r8/final"
llm=LLM.load(str(checkpoint),distribute=None)
llm.distribute(accelerator="cuda",precision="bf16-true")
llm.model.eval()
mode=sys.argv[1] if len(sys.argv)>1 else "hierarchical"
result=summarize_dialogue(llm,dialogue,mode)
out=root/f"reports/output_reliability/ui_long_example_smoke_{mode}.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({
 "summary":result["summary"],
 "risk":result["reliability"]["risk"],
 "risk_codes":[x["code"] for x in result["reliability"]["reasons"]],
 "chunks":result["chunk_count"],
 "deduplicated_turns":result["deduplicated_turns"],
 "coverage":result["input_coverage"],
 "seconds":result["total_seconds"],
 "peak_memory_gb":result["peak_memory_gb"],
 "final_tokens":result["final_tokens"],
},ensure_ascii=False,indent=2))
