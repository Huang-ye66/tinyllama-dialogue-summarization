"""Interactive TinyLlama QLoRA demo with hierarchical long-dialogue summarization."""
from __future__ import annotations
import gc,os,time
from pathlib import Path
from typing import Any
import streamlit as st
from scripts.long_dialogue.hierarchical import summarize_dialogue

REPO_ROOT=Path(__file__).resolve().parent
RUN_ROOT=Path(os.environ.get("TINYLLAMA_PROJECT_ROOT", REPO_ROOT)).expanduser().resolve()
MODELS={
 "R8 · Resource-optimized (recommended)":{"path":RUN_ROOT/"outputs/models/resource_optimized_r8/final","description":"All-module QLoRA rank 8; about 75% smaller than rank 32 with comparable quality across three seeds.","role":"Deployed optimized model"},
 "R32 · Quality baseline":{"path":RUN_ROOT/"outputs/models/quality_baseline_r32/final","description":"All-module QLoRA rank 32 used as the quality and resource baseline.","role":"Reference baseline"},
}
PROCESSING={"Auto":"auto","Direct (head-and-tail truncation for long input)":"direct","Hierarchical long-dialogue":"hierarchical"}
EXAMPLE="""Leticia: Would any of you have $10 I could borrow? I lost my wallet.
Lora: Sure. Do you need anything else?
Leticia: No, thank you. I will pay you back tomorrow.
Miranda: I can help you look for it after class."""
LONG_EXAMPLE="\n".join([
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
]*4)

st.set_page_config(page_title="TinyLlama Dialogue Summarization",page_icon="📝",layout="wide")

def release_model()->None:
 model=st.session_state.pop("loaded_llm",None);st.session_state.pop("loaded_label",None)
 if model is not None:del model
 gc.collect()
 try:
  import torch
  if torch.cuda.is_available():torch.cuda.empty_cache()
 except ImportError:pass

def load_model(label:str):
 if st.session_state.get("loaded_label")==label and st.session_state.get("loaded_llm") is not None:return st.session_state["loaded_llm"],0.0,True
 release_model();checkpoint=MODELS[label]["path"]
 if not checkpoint.exists():raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
 import torch
 if not torch.cuda.is_available():raise RuntimeError("No CUDA GPU detected; this demo was validated on a 6 GB GPU.")
 from litgpt import LLM
 started=time.perf_counter();llm=LLM.load(str(checkpoint),distribute=None);llm.distribute(accelerator="cuda",precision="bf16-true");llm.model.eval()
 st.session_state["loaded_llm"]=llm;st.session_state["loaded_label"]=label
 return llm,time.perf_counter()-started,False

def generate_one(label:str,dialogue:str,processing:str,display_label:str|None=None)->dict[str,Any]:
 llm,load_seconds,cache_hit=load_model(label)
 result=summarize_dialogue(llm,dialogue,processing)
 return result|{"model_label":label,"display_label":display_label or label,"load_seconds":load_seconds,"cache_hit":cache_hit}

def show_result(result:dict[str,Any])->None:
 st.subheader(result["display_label"]);st.caption(MODELS[result["model_label"]]["role"])
 if result["truncated"]:st.warning(f"Direct mode retained about {result['input_coverage']:.0%} of the input; middle content was truncated.")
 elif result["actual_mode"]=="hierarchical":st.success(f"Processed {result['chunk_count']} chunks with {result['input_coverage']:.0%} input coverage.")
 if result.get("deduplication_applied"):st.info(f"Collapsed {result['deduplicated_turns']} exactly repeated turns while retaining source provenance.")
 if result.get("attribution_repairs"):st.info(f"Repaired or removed {len(result['attribution_repairs'])} high-confidence attribution conflicts using source-turn evidence.")
 st.markdown("**Generated summary**");st.write(result["summary"])
 check=result["reliability"]
 if check["risk"]=="high":st.error("Rule-based consistency risk: high")
 elif check["risk"]=="medium":st.warning("Rule-based consistency risk: medium")
 else:st.success("Rule-based consistency risk: low")
 messages={"empty_summary":"The model returned an empty summary.","unsupported_person":"The summary contains a person or entity not found in the source.","unsupported_number":"The summary contains a number, time, or amount not found in the source.","possible_number_omission":"Some source numbers are absent from the summary; review whether they are important.","duplicate_event":"The summary may repeat the same event.","possible_attribution_conflict":"Similar events are assigned to different people; attribution may be incorrect.","incomplete_sentence":"The summary ends with an incomplete sentence.","input_truncated":"The input was truncated before generation."}
 for reason in check["reasons"]:
  values="、".join(reason.get("values",[]));st.caption("• "+messages.get(reason["code"],reason["message"])+(f"（{values}）" if values else ""))
 c1,c2,c3,c4=st.columns(4)
 c1.metric("Total latency",f"{result['total_seconds']:.2f}s");c2.metric("Final tokens",result["final_tokens"]);c3.metric("Peak VRAM",f"{result['peak_memory_gb']:.2f}GB");c4.metric("Input coverage",f"{result['input_coverage']:.0%}")
 cache="Model cache hit" if result["cache_hit"] else f"Initial load: {result['load_seconds']:.2f}s"
 st.caption(f"{cache} | Pipeline: {'hierarchical' if result['actual_mode']=='hierarchical' else 'direct'} | {result['generated_tokens_total']} generated tokens across all stages")
 if result["chunk_summaries"]:
  with st.expander(f"View {len(result['chunk_summaries'])} local summaries"):
   for i,(chunk,summary) in enumerate(zip(result["chunks"],result["chunk_summaries"]),1):
    st.markdown(f"**Chunk {i} · {chunk['token_count']} tokens**")
    st.write(summary)
 with st.expander("View source fact table"):st.json(result["source_facts"])

st.title("TinyLlama Dialogue Summarization")
st.caption("Short inputs are summarized directly. Long dialogues use complete-turn chunking, local summaries, and recursive merging.")
with st.sidebar:
 st.header("Project results");st.metric("R8 adapter reduction","74.9%");st.metric("Three-seed mean ROUGE-L change","-0.00101");st.metric("Test ROUGE-L change","-0.00206")
 st.success("Low-rank resource optimization validated across three seeds.")
 st.success("Hierarchical processing: +0.0303 ROUGE-L and 100% coverage on real long dialogues.")
 with st.expander("Negative experiment record"):
  st.write("Differential learning rates did not pass the quality gate.");st.write("Factual contrastive training improved strong-margin discrimination but not the primary quality gate.")
 if st.button("Release model VRAM",width="stretch"):release_model();st.success("Model memory released.")

if "dialogue" not in st.session_state:st.session_state.dialogue=""
b1,b2,b3=st.columns([1,1,2])
with b1:
 if st.button("Load short example",width="stretch"):st.session_state.dialogue=EXAMPLE
with b2:
 if st.button("Load long example",width="stretch"):st.session_state.dialogue=LONG_EXAMPLE
with b3:st.caption("The first run loads the model; subsequent runs reuse the cached model.")
dialogue=st.text_area("English dialogue",key="dialogue",height=260,placeholder="Alice: Are you free tomorrow?\nBob: Yes, let's meet at 3 pm.")
view=st.radio("Comparison",["Single model","R8 vs. R32","Direct vs. hierarchical"],horizontal=True)
model_label=st.selectbox("Model",list(MODELS),disabled=view!="Single model")
processing_label=st.selectbox("Input processing",list(PROCESSING),disabled=view=="Direct vs. hierarchical")
if view=="Single model":st.info(MODELS[model_label]["description"])
elif view=="R8 vs. R32":st.info("Models are loaded sequentially with the same processing mode to stay within the 6 GB VRAM target.")
else:st.info("The same rank-8 model compares head-and-tail truncation with 100% complete-turn coverage.")

if st.button("Generate summary",type="primary",disabled=not dialogue.strip(),width="stretch"):
 progress=st.progress(0,text="Preparing model...");results=[]
 try:
  if view=="Single model":jobs=[(model_label,PROCESSING[processing_label],model_label,False)]
  elif view=="R8 vs. R32":jobs=[(x,PROCESSING[processing_label],x,True) for x in MODELS]
  else:
   r8=next(iter(MODELS));jobs=[(r8,"direct","R8 · Direct truncation",False),(r8,"hierarchical","R8 · Hierarchical",False)]
  for i,(label,mode,title,release_after) in enumerate(jobs):
   progress.progress(i/len(jobs),text=f"Running {title}");results.append(generate_one(label,dialogue,mode,title))
   if release_after:release_model()
  progress.progress(1.0,text="Generation complete")
  for column,result in zip(st.columns(len(results)),results):
   with column:show_result(result)
 except Exception as exc:
  st.error(f"Generation failed: {exc}")
  with st.expander("Technical details"):st.exception(exc)
 finally:progress.empty()

st.divider();st.subheader("Experiments and conclusions")
st.dataframe([
 {"Module":"Low-rank resource optimization","Goal":"Adapter compression","Status":"Passed","Conclusion":"74.9% smaller adapter with comparable quality across three seeds"},
 {"Module":"Differential learning rates","Goal":"Training efficiency","Status":"Did not pass","Conclusion":"Similar ROUGE with a worse critical-error proxy"},
 {"Module":"Factual contrastive training","Goal":"Fact discrimination","Status":"Did not pass","Conclusion":"Stronger margin discrimination without primary-metric improvement"},
 {"Module":"Hierarchical summarization","Goal":"Long-input coverage","Status":"Passed automatic evaluation","Conclusion":"Real long dialogues: ROUGE-L +0.0303, critical-error proxy -14.96%, 100% coverage"},
],hide_index=True,width="stretch")
st.caption("Rule-based warnings support human review. Real and synthetic long-dialogue results are reported separately.")
