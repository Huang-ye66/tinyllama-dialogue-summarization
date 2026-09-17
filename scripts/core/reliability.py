"""Explainable source-consistency checks for dialogue summaries."""
from __future__ import annotations
import re
from typing import Any
SPEAKER_RE = re.compile(r"(?m)^\s*([A-Z][A-Za-z .'-]{0,39}?):")
NUMBER_RE = re.compile(r"(?<!\w)(?:[$]|\u00a3|\u20ac)?\s*\d+(?:[.,:]\d+)*(?:\s*(?:am|pm|AM|PM|%|dollars?|euros?|pounds?|hours?|minutes?|days?|weeks?))?(?!\w)")
CAP_RE = re.compile(r"\b[A-Z][A-Za-z'-]{1,30}\b")
IGNORE = {"The","A","An","This","That","These","Those","It","They","He","She","We","I","Tomorrow","Today","Yesterday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"}
def _norm_number(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold().replace(",", "")
def extract_facts(dialogue: str) -> dict[str, list[str]]:
    return {"speakers": list(dict.fromkeys(x.strip() for x in SPEAKER_RE.findall(dialogue))), "numbers": list(dict.fromkeys(NUMBER_RE.findall(dialogue)))}

def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]

def _content_words(text: str) -> set[str]:
    return {x.casefold() for x in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if x not in IGNORE and len(x) > 2}

def _speaker_for_sentence(sentence: str, speakers: list[str]) -> str | None:
    lower = sentence.casefold()
    matches = []
    for speaker in speakers:
        match = re.search(rf"\b{re.escape(speaker.casefold())}\b", lower)
        if match:
            matches.append((match.start(), speaker))
    return min(matches)[1] if matches else None

def speakers(dialogue: str) -> list[str]:
    return extract_facts(dialogue)["speakers"]

def assess_summary(dialogue: str, summary: str, truncated: bool = False) -> dict[str, Any]:
    facts = extract_facts(dialogue)
    dialogue_words = {x.casefold() for x in re.findall(r"[A-Za-z][A-Za-z'-]*", dialogue)}
    unsupported_people = []
    for token in CAP_RE.findall(summary):
        if token not in IGNORE and token.casefold() not in dialogue_words and token not in unsupported_people:
            unsupported_people.append(token)
    source_numbers = {_norm_number(x): x for x in facts["numbers"]}
    summary_numbers = {_norm_number(x): x for x in NUMBER_RE.findall(summary)}
    unsupported_numbers = [raw for key, raw in summary_numbers.items() if key not in source_numbers]
    omitted_numbers = [raw for key, raw in source_numbers.items() if key not in summary_numbers]
    sentences = _sentences(summary)
    incomplete = bool(summary.strip()) and not bool(re.search(r"[.!?][\"')\]]?\s*$", summary.strip()))
    duplicate_pairs = []
    attribution_conflicts = []
    for i, left in enumerate(sentences):
        left_words = _content_words(left)
        for j, right in enumerate(sentences[i + 1:], i + 1):
            right_words = _content_words(right)
            union = left_words | right_words
            similarity = len(left_words & right_words) / len(union) if union else 0.0
            if similarity >= 0.82:
                duplicate_pairs.append([i + 1, j + 1])
            left_speaker = _speaker_for_sentence(left, facts["speakers"])
            right_speaker = _speaker_for_sentence(right, facts["speakers"])
            if similarity >= 0.68 and left_speaker and right_speaker and left_speaker != right_speaker:
                attribution_conflicts.append({"sentences":[i + 1, j + 1], "people":[left_speaker, right_speaker]})
    reasons = []
    if not summary.strip(): reasons.append({"code":"empty_summary","severity":"high","message":"The model returned an empty summary."})
    if unsupported_people: reasons.append({"code":"unsupported_person","severity":"high","values":unsupported_people,"message":"The summary contains capitalized names or entities not found in the dialogue."})
    if unsupported_numbers: reasons.append({"code":"unsupported_number","severity":"high","values":unsupported_numbers,"message":"The summary contains numbers, times, or amounts not found in the dialogue."})
    if omitted_numbers: reasons.append({"code":"possible_number_omission","severity":"medium","values":omitted_numbers,"message":"Some dialogue numbers are absent from the summary; review whether they are important."})
    if duplicate_pairs: reasons.append({"code":"duplicate_event","severity":"medium","pairs":duplicate_pairs,"message":"The summary may repeat the same event."})
    if attribution_conflicts: reasons.append({"code":"possible_attribution_conflict","severity":"high","conflicts":attribution_conflicts,"message":"Very similar events may be attributed to different speakers."})
    if incomplete: reasons.append({"code":"incomplete_sentence","severity":"high","message":"The summary ends with an incomplete sentence."})
    if truncated: reasons.append({"code":"input_truncated","severity":"medium","message":"The input was truncated before generation."})
    severity = "high" if any(x["severity"] == "high" for x in reasons) else "medium" if reasons else "low"
    return {"risk":severity,"reasons":reasons,"source_facts":facts,"flags":{"unsupported_person":bool(unsupported_people),"unsupported_number":bool(unsupported_numbers),"possible_number_omission":bool(omitted_numbers),"duplicate_event":bool(duplicate_pairs),"possible_attribution_conflict":bool(attribution_conflicts),"incomplete_sentence":incomplete,"input_truncated":bool(truncated)}}
