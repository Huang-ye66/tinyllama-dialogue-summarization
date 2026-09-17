"""Speaker- and fact-preserving hierarchical summarization for long dialogues."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from scripts.core.reliability import assess_summary, extract_facts

DIRECT_INSTRUCTION = "Summarize the following dialogue in one concise paragraph."
CHUNK_INSTRUCTION = (
    "Summarize this dialogue section in at most 60 tokens. Preserve who did what, "
    "speaker attribution, decisions, amounts, dates and times. Do not invent facts."
)
MERGE_INSTRUCTION = (
    "Merge the section summaries into one concise dialogue summary in at most 100 tokens. "
    "Preserve speaker attribution and important numbers or times. Use only supported facts. "
    "Never transfer an action from one person to another, remove repeated events, and finish every sentence."
)
COMPLETE_INSTRUCTION = (
    "Rewrite the draft as a concise summary using complete sentences only. Preserve its supported facts, "
    "remove repetitions, and do not add or reassign actions."
)
MAX_PROMPT_TOKENS = 412
CHUNK_INPUT_TOKENS = 280
LONG_TURN_OVERLAP = 20


@dataclass
class Chunk:
    text: str
    turn_indices: list[int]
    token_count: int


def collapse_repeated_dialogue(dialogue: str, min_block_turns: int = 4) -> tuple[str, dict[int, list[int]]]:
    """Collapse an exact whole-dialogue repeated block while retaining source-index provenance."""
    indexed = [(i, line.strip()) for i, line in enumerate(dialogue.splitlines()) if line.strip()]
    lines = [line for _, line in indexed]
    total = len(lines)
    for block_size in range(min_block_turns, total // 2 + 1):
        if total % block_size:
            continue
        copies = total // block_size
        if copies >= 2 and all(lines[start:start + block_size] == lines[:block_size] for start in range(0, total, block_size)):
            provenance = {i:[indexed[i + copy * block_size][0] for copy in range(copies)] for i in range(block_size)}
            return "\n".join(lines[:block_size]), provenance
    return dialogue.strip(), {i:[source_i] for i, (source_i, _) in enumerate(indexed)}


def _encode(tokenizer: Any, text: str):
    return tokenizer.encode(text)


def _token_len(tokenizer: Any, text: str) -> int:
    return len(_encode(tokenizer, text))


def _decode(tokenizer: Any, token_ids: Any) -> str:
    return tokenizer.decode(token_ids).strip()


def _split_long_turn(tokenizer: Any, turn: str, budget: int) -> list[str]:
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", turn) if x.strip()]
    pieces: list[str] = []
    for sentence in sentences or [turn]:
        ids = _encode(tokenizer, sentence)
        if len(ids) <= budget:
            pieces.append(sentence)
            continue
        start = 0
        while start < len(ids):
            end = min(start + budget, len(ids))
            pieces.append(_decode(tokenizer, ids[start:end]))
            if end == len(ids):
                break
            start = max(start + 1, end - LONG_TURN_OVERLAP)
    groups: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = " ".join(current + [piece])
        if current and _token_len(tokenizer, candidate) > budget:
            groups.append(" ".join(current));current = [piece]
        else:
            current.append(piece)
    if current:
        groups.append(" ".join(current))
    return groups


def chunk_dialogue(tokenizer: Any, dialogue: str, budget: int = CHUNK_INPUT_TOKENS) -> list[Chunk]:
    if not dialogue.strip():
        raise ValueError("Dialogue is empty.")
    turns = [(i, line.strip()) for i, line in enumerate(dialogue.splitlines()) if line.strip()]
    pieces: list[tuple[int, str]] = []
    for index, turn in turns:
        if _token_len(tokenizer, turn) <= budget:
            pieces.append((index, turn))
        else:
            pieces.extend((index, part) for part in _split_long_turn(tokenizer, turn, budget))
    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    for piece in pieces:
        candidate = "\n".join(x[1] for x in current + [piece])
        if current and _token_len(tokenizer, candidate) > budget:
            text = "\n".join(x[1] for x in current)
            chunks.append(Chunk(text, list(dict.fromkeys(x[0] for x in current)), _token_len(tokenizer, text)))
            current = [piece]
        else:
            current.append(piece)
    if current:
        text = "\n".join(x[1] for x in current)
        chunks.append(Chunk(text, list(dict.fromkeys(x[0] for x in current)), _token_len(tokenizer, text)))
    expected = {i for i, _ in turns}
    covered = {i for chunk in chunks for i in chunk.turn_indices}
    if covered != expected:
        raise RuntimeError(f"Turn coverage mismatch: missing={sorted(expected-covered)} extra={sorted(covered-expected)}")
    return chunks


def _format_prompt(llm: Any, instruction: str, input_text: str) -> str:
    from litgpt.prompts import Alpaca, Default
    llm.prompt_style = Default()
    return Alpaca().apply(instruction, input=input_text)


def _direct_prompt(llm: Any, dialogue: str) -> tuple[str, bool, float]:
    clean = dialogue.strip()
    prompt = _format_prompt(llm, DIRECT_INSTRUCTION, clean)
    if _token_len(llm.preprocessor.tokenizer, prompt) <= MAX_PROMPT_TOKENS:
        return prompt, False, 1.0
    original = clean
    keep = max(200, int(len(clean) * 0.8))
    while keep > 200:
        head = int(keep * 0.7)
        tail = keep - head
        retained = original[:head] + "\n[... dialogue truncated ...]\n" + original[-tail:]
        prompt = _format_prompt(llm, DIRECT_INSTRUCTION, retained)
        if _token_len(llm.preprocessor.tokenizer, prompt) <= MAX_PROMPT_TOKENS:
            return prompt, True, min(1.0, keep / max(1, len(original)))
        keep = int(keep * 0.8)
    raise ValueError("Dialogue remains too long after direct-mode truncation.")


def source_facts(dialogue: str) -> dict[str, list[str]]:
    facts = extract_facts(dialogue)
    speakers = {x.casefold() for x in facts["speakers"]}
    ignored = {"The", "This", "That", "These", "Those", "A", "An", "I", "He", "She", "We", "They"}
    mentioned = []
    for value in re.findall(r"\b[A-Z][A-Za-z'-]{1,30}\b", dialogue):
        if value not in ignored and value.casefold() not in speakers and value not in mentioned:
            mentioned.append(value)
    return {**facts, "mentioned_people": mentioned}


def _facts_text(dialogue: str) -> str:
    facts = source_facts(dialogue)
    speakers = ", ".join(facts["speakers"]) or "none detected"
    people = ", ".join(facts["mentioned_people"]) or "none detected"
    numbers = ", ".join(x.strip() for x in facts["numbers"]) or "none detected"
    return f"Dialogue speakers: {speakers}\nMentioned people: {people}\nNumbers/times/amounts from source: {numbers}"


def _generate(llm: Any, prompt: str, max_new_tokens: int) -> tuple[str, float, int]:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        text = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0, top_k=1).strip()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return text, elapsed, _token_len(llm.preprocessor.tokenizer, text)
    finally:
        clear_cache = getattr(getattr(llm, "model", None), "clear_kv_cache", None)
        if callable(clear_cache):
            clear_cache()
            if hasattr(llm, "kv_cache_initialized"):
                llm.kv_cache_initialized = False
            if hasattr(llm, "prev_generated_seq_length"):
                llm.prev_generated_seq_length = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_incomplete(text: str) -> bool:
    return bool(text.strip()) and not bool(re.search(r"[.!?][\"')\]]?\s*$", text.strip()))


def _complete_summary(llm: Any, summary: str, timings: list[float]) -> tuple[str, int]:
    if not _is_incomplete(summary):
        return summary.strip(), 0
    prompt = _format_prompt(llm, COMPLETE_INSTRUCTION, summary.strip())
    repaired, elapsed, tokens = _generate(llm, prompt, 100)
    timings.append(elapsed)
    if not _is_incomplete(repaired):
        return repaired.strip(), tokens
    complete = re.findall(r".+?[.!?](?:[\"')\]]?)(?:\s+|$)", repaired.strip())
    if complete:
        return " ".join(x.strip() for x in complete).strip(), tokens
    complete = re.findall(r".+?[.!?](?:[\"')\]]?)(?:\s+|$)", summary.strip())
    return (" ".join(x.strip() for x in complete).strip() if complete else repaired.strip()), tokens


def _sentence_units(text: str) -> list[str]:
    return [x.strip() for x in re.findall(r"[^.!?]+[.!?]", text.strip()) if x.strip()]


def _action_words(text: str, speakers: list[str]) -> set[str]:
    stop = {"the","and","with","will","would","could","should","this","that","from","into","about","every","also","needs","need"}
    speaker_words = {x.casefold() for name in speakers for x in name.split()}
    return {x.casefold() for x in re.findall(r"[A-Za-z][A-Za-z'-]*", text)
            if x.casefold() not in stop and x.casefold() not in speaker_words and len(x) > 2}


def _primary_speaker(sentence: str, speakers: list[str]) -> str | None:
    lower = sentence.casefold()
    matches = []
    for speaker in speakers:
        match = re.search(rf"\b{re.escape(speaker.casefold())}\b", lower)
        if match:
            matches.append((match.start(), speaker))
    return min(matches)[1] if matches else None


def repair_attribution_conflicts(dialogue: str, summary: str) -> tuple[str, list[dict[str, Any]]]:
    """Correct explicit unsupported actions, then remove high-confidence duplicate assignments."""
    speakers = extract_facts(dialogue)["speakers"]
    source = {speaker:[] for speaker in speakers}
    source_raw = {speaker:set() for speaker in speakers}
    for line in dialogue.splitlines():
        match = re.match(r"^\s*([^:]{1,40}):\s*(.+)$", line)
        if not match:
            continue
        owner = next((x for x in speakers if x.casefold() == match.group(1).strip().casefold()), None)
        utterance = match.group(2)
        if owner:
            source[owner].append(_action_words(utterance, speakers))
            source_raw[owner].update(x.casefold() for x in re.findall(r"[A-Za-z][A-Za-z'-]*", utterance))
        # Explicit directives such as "Bob, please confirm..." also support Bob's assignment.
        for target in speakers:
            if re.match(rf"^\s*{re.escape(target)}(?:,|\s+(?:will|must|should|needs? to|please))", utterance, re.I):
                source[target].append(_action_words(utterance, speakers))
                source_raw[target].update(x.casefold() for x in re.findall(r"[A-Za-z][A-Za-z'-]*", utterance))

    sentences = _sentence_units(summary)
    repairs: list[dict[str, Any]] = []

    # Correct only explicit "X will VERB" claims when X has no evidence for VERB and another speaker does.
    for index, sentence in enumerate(sentences):
        assigned = _primary_speaker(sentence, speakers)
        if not assigned:
            continue
        action = re.match(
            rf"^\s*{re.escape(assigned)}\s+(?:will|must|should|can|could|plans? to|needs? to|is going to)\s+([A-Za-z'-]+)",
            sentence, re.I,
        )
        if not action:
            continue
        verb = action.group(1).casefold()
        if verb in source_raw.get(assigned, set()):
            continue
        words = _action_words(sentence, speakers)
        candidates = []
        for candidate in speakers:
            if candidate == assigned or verb not in source_raw.get(candidate, set()):
                continue
            support = max((len(words & item) / max(1, len(words)) for item in source.get(candidate, [])), default=0.0)
            candidates.append((support, candidate))
        if not candidates:
            continue
        best_support, best = max(candidates)
        if best_support >= 0.50:
            sentences[index] = re.sub(rf"^\s*{re.escape(assigned)}\b", best, sentence, count=1, flags=re.I)
            repairs.append({"sentence":index + 1,"from":assigned,"to":best,"action":verb,"reason":"unsupported_explicit_attribution"})

    removed: set[int] = set()
    for i, left in enumerate(sentences):
        if i in removed:
            continue
        for j in range(i + 1, len(sentences)):
            if j in removed:
                continue
            right = sentences[j]
            left_words, right_words = _action_words(left, speakers), _action_words(right, speakers)
            union = left_words | right_words
            similarity = len(left_words & right_words) / len(union) if union else 0.0
            left_speaker, right_speaker = _primary_speaker(left, speakers), _primary_speaker(right, speakers)
            if not left_speaker or not right_speaker:
                continue
            if left_speaker == right_speaker and similarity >= 0.50:
                drop = i if len(left_words) <= len(right_words) else j
                keep = j if drop == i else i
                removed.add(drop)
                repairs.append({"removed_sentence":drop + 1,"kept_sentence":keep + 1,"reason":"near_duplicate_event"})
                if drop == i:
                    break
                continue
            if similarity < 0.68 or left_speaker == right_speaker:
                continue
            def support(sentence_words: set[str], speaker: str) -> float:
                return max((len(sentence_words & words) / max(1, len(sentence_words)) for words in source.get(speaker, [])), default=0.0)
            left_support = support(left_words, left_speaker)
            right_support = support(right_words, right_speaker)
            if left_support <= 0.30 and right_support >= 0.60:
                removed.add(i);repairs.append({"removed_sentence":i + 1,"kept_sentence":j + 1,"reason":"unsupported_duplicate_attribution"})
                break
            if right_support <= 0.30 and left_support >= 0.60:
                removed.add(j);repairs.append({"removed_sentence":j + 1,"kept_sentence":i + 1,"reason":"unsupported_duplicate_attribution"})
    repaired = " ".join(sentence for i, sentence in enumerate(sentences) if i not in removed)
    return (repaired or summary.strip()), repairs

def _fit_merge_input(llm: Any, summaries: list[str], facts: str) -> str:
    numbered = "\n".join(f"Section {i+1}: {x}" for i, x in enumerate(summaries))
    full = facts + "\n" + numbered
    if _token_len(llm.preprocessor.tokenizer, _format_prompt(llm, MERGE_INSTRUCTION, full)) <= MAX_PROMPT_TOKENS:
        return full
    # Facts are evidence hints; section summaries remain the primary merge input.
    fact_lines = facts.splitlines()
    while fact_lines:
        fact_lines.pop()
        candidate = "\n".join(fact_lines + [numbered])
        if _token_len(llm.preprocessor.tokenizer, _format_prompt(llm, MERGE_INSTRUCTION, candidate)) <= MAX_PROMPT_TOKENS:
            return candidate
    return numbered


def _merge_recursive(llm: Any, summaries: list[str], facts: str, timings: list[float]) -> tuple[str, int]:
    if not summaries:
        raise RuntimeError("No section summaries were generated.")
    if len(summaries) == 1:
        merge_input = _fit_merge_input(llm, summaries, facts)
        result, elapsed, tokens = _generate(llm, _format_prompt(llm, MERGE_INSTRUCTION, merge_input), 100)
        timings.append(elapsed)
        return result, tokens
    merge_input = _fit_merge_input(llm, summaries, facts)
    if _token_len(llm.preprocessor.tokenizer, _format_prompt(llm, MERGE_INSTRUCTION, merge_input)) <= MAX_PROMPT_TOKENS:
        result, elapsed, tokens = _generate(llm, _format_prompt(llm, MERGE_INSTRUCTION, merge_input), 100)
        timings.append(elapsed)
        return result, tokens
    groups: list[list[str]] = []
    current: list[str] = []
    for summary in summaries:
        candidate = current + [summary]
        candidate_input = "\n".join(f"Section {i+1}: {x}" for i, x in enumerate(candidate))
        if current and _token_len(llm.preprocessor.tokenizer, _format_prompt(llm, MERGE_INSTRUCTION, candidate_input)) > 320:
            groups.append(current)
            current = [summary]
        else:
            current = candidate
    if current:
        groups.append(current)
    if len(groups) == 1:
        raise RuntimeError("Unable to reduce merge prompt within token budget.")
    reduced: list[str] = []
    generated_tokens = 0
    for group in groups:
        group_input = _fit_merge_input(llm, group, "")
        item, elapsed, tokens = _generate(llm, _format_prompt(llm, MERGE_INSTRUCTION, group_input), 80)
        timings.append(elapsed);generated_tokens += tokens;reduced.append(item)
    final, tokens = _merge_recursive(llm, reduced, facts, timings)
    return final, generated_tokens + tokens


def summarize_dialogue(llm: Any, dialogue: str, mode: str = "auto") -> dict[str, Any]:
    if mode not in {"auto", "direct", "hierarchical"}:
        raise ValueError("mode must be auto, direct, or hierarchical")
    if not dialogue.strip():
        raise ValueError("Dialogue is empty.")
    import torch
    tokenizer = llm.preprocessor.tokenizer
    cleaned_dialogue, provenance = collapse_repeated_dialogue(dialogue)
    deduplicated_turns = sum(len(x) - 1 for x in provenance.values())
    full_prompt = _format_prompt(llm, DIRECT_INSTRUCTION, cleaned_dialogue)
    actual = "direct" if mode == "direct" or (mode == "auto" and _token_len(tokenizer, full_prompt) <= MAX_PROMPT_TOKENS) else "hierarchical"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if actual == "direct":
        prompt, truncated, coverage = _direct_prompt(llm, cleaned_dialogue)
        summary, elapsed, tokens = _generate(llm, prompt, 100)
        chunks: list[Chunk] = []
        partials: list[str] = []
        timings = [elapsed]
        generated_tokens = tokens
    else:
        chunks = chunk_dialogue(tokenizer, cleaned_dialogue)
        partials=[];timings=[];generated_tokens=0
        for chunk in chunks:
            prompt = _format_prompt(llm, CHUNK_INSTRUCTION, chunk.text)
            if _token_len(tokenizer, prompt) > MAX_PROMPT_TOKENS:
                raise RuntimeError("Chunk prompt exceeds model budget.")
            text, elapsed, tokens = _generate(llm, prompt, 60)
            if not text:
                raise RuntimeError("A section summary is empty.")
            partials.append(text);timings.append(elapsed);generated_tokens += tokens
        if len(partials) == 1:
            summary = partials[0]
        else:
            summary, merge_tokens = _merge_recursive(llm, partials, _facts_text(cleaned_dialogue), timings)
            generated_tokens += merge_tokens
        truncated=False;coverage=1.0
    summary, repair_tokens = _complete_summary(llm, summary, timings)
    generated_tokens += repair_tokens
    summary, attribution_repairs = repair_attribution_conflicts(cleaned_dialogue, summary)
    if not summary:
        raise RuntimeError("Final summary is empty.")
    peak = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0
    return {
        "summary":summary,
        "requested_mode":mode,
        "actual_mode":actual,
        "chunk_count":len(chunks) if chunks else 1,
        "chunks":[{"text":x.text,"turn_indices":x.turn_indices,"token_count":x.token_count} for x in chunks],
        "chunk_summaries":partials,
        "input_coverage":coverage,
        "truncated":truncated,
        "source_facts":source_facts(cleaned_dialogue),
        "deduplicated_turns":deduplicated_turns,
        "deduplication_applied":deduplicated_turns > 0,
        "attribution_repairs":attribution_repairs,
        "stage_seconds":timings,
        "total_seconds":time.perf_counter()-started,
        "generated_tokens_total":generated_tokens,
        "final_tokens":_token_len(tokenizer,summary),
        "peak_memory_gb":peak,
        "reliability":assess_summary(cleaned_dialogue,summary,truncated),
    }
