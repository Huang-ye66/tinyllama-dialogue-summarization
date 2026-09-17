"""Run the locked integration smoke on the longest validation dialogue."""
import json
import os
import sys
from pathlib import Path

root=Path(os.environ.get("TINYLLAMA_PROJECT_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
sys.path.insert(0,str(root))

from litgpt import LLM
from scripts.long_dialogue.hierarchical import summarize_dialogue

rows=json.loads((root/"data/samsum/val.json").read_text(encoding="utf-8"))
llm=LLM.load(str(root/"outputs/models/resource_optimized_r8/final"),distribute=None)
longest=max(rows,key=lambda x:len(llm.preprocessor.tokenizer.encode(x["input"])))
llm.distribute(accelerator="cuda",precision="bf16-true")
llm.model.eval()
first=summarize_dialogue(llm,longest["input"],"hierarchical")
second=summarize_dialogue(llm,longest["input"],"hierarchical")
assert first["summary"]==second["summary"]
assert first["input_coverage"]==1.0 and first["chunk_count"]>=2
assert first["peak_memory_gb"]<=5.5 and first["summary"]
result={
    "source_index":longest["source_index"],
    "input_tokens":len(llm.preprocessor.tokenizer.encode(longest["input"])),
    "deterministic":True,
    "first":first,
    "second_total_seconds":second["total_seconds"],
    "test_split_read":False,
}
out=root/"reports/long_dialogue/integration_smoke.json"
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({
    "source_index":result["source_index"],
    "input_tokens":result["input_tokens"],
    "deterministic":True,
    "chunks":first["chunk_count"],
    "coverage":first["input_coverage"],
    "peak_memory_gb":first["peak_memory_gb"],
    "seconds":first["total_seconds"],
    "summary":first["summary"],
},ensure_ascii=False,indent=2))
