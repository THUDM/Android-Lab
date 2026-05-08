from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from evaluation.evaluation import TextOnlyTask
from evaluation.utils import extract_bounds
from recorder.json_recoder import get_compressed_xml
### RANK PHASE###
#RANK PROMPT 1
RTV_SYSTEM_PROMPT = """You are an Android GUI action selector.

Given the task, the current XML screen description, and a finite candidate action
set, choose the next action. Do not invent elements or coordinates outside the
candidate set. When asked to choose a candidate, answer with one letter only.
"""
#RANK PROMPT 2
RANK_USER_TEMPLATE = """Goal: {instruction}
Current XML:
{xml_text}

Recent actions:
{recent_actions}

The following candidate next operations are labeled by letters:

{candidate_block}

Some candidates contain placeholders such as `<TEXT_TO_FILL>` or
`<MESSAGE_TO_FILL>`. Treat these as fillable action templates: choose them when
typing text or finishing with a message is the correct next operation. The exact
text/message will be generated only after this letter is selected.

Which single letter labels the best next operation?

Finish rule:
- If the task goal is already satisfied on the current screen, choose the
  `finish(...)` candidate.
- If the task asks for an answer and the answer is visible/known, choose the
  `finish(...)` candidate.
- Do not choose `finish(...)` before the task is satisfied.

Answer with one letter only."""
### RANK END ###

### VERIFY PHASE ###
#VERIFY PROMPT 1

VERIFY_SYSTEM_PROMPT = """You are a verifier for Android action selection.
Given the current goal, action history, and UI state, decide whether one
candidate action is the correct next step.
Answer exactly Yes or No."""

#VERIFY PROMPT 2
VERIFY_USER_TEMPLATE = """Goal:
{instruction}

Current XML:
{xml_text}

Recent actions:
{recent_actions}

Ranked candidate set:
{candidate_block}

Candidate to verify:
{candidate_label}. {candidate_description}
code: {candidate_code}

Some candidates contain placeholders such as `<TEXT_TO_FILL>` or
`<MESSAGE_TO_FILL>`. Treat these as fillable action templates: answer Yes if
this template is the single best next operation among the ranked candidate set;
the exact text/message will be generated only after the candidate is selected.

Is the candidate to verify the single best next step among the ranked candidate set?
Answer exactly Yes or No."""
### VERIFY END ###

### FILL PHASE ###
FILL_SYSTEM = """You fill one missing text slot for an Android automation step.
Do not choose actions. Do not output code. Return only the exact text."""

FILL_TYPE_TAIL = """The chosen next operation is to type text into this UI element:
{element_desc}
bounds={bbox}

Return the exact text that should replace <TEXT_TO_FILL>. Return text only."""

FILL_FINISH_TAIL = """The chosen next operation is `finish(message=...)`.

Return the concise message or answer that should replace <MESSAGE_TO_FILL>.
Return message text only."""
### FILL END ###

BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
HASH_RE = re.compile(r"\[n[0-9a-f]{4}\]\s*")
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Candidate:
    label: str
    kind: str
    description: str
    code: str
    action: Dict[str, Any]
    target_text: str = ""
    bbox: Optional[List[int]] = None
    needs_fill: Optional[str] = None
    code_template: Optional[str] = None


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _label_for_index(index: int) -> str:
    if index < len(ALPHABET):
        return ALPHABET[index]
    index -= len(ALPHABET)
    return ALPHABET[index // len(ALPHABET)] + ALPHABET[index % len(ALPHABET)]


def _parse_bounds(bounds_str: str) -> Optional[List[int]]:
    match = BOUNDS_RE.search(bounds_str or "")
    if not match:
        return None
    return [int(match.group(i)) for i in range(1, 5)]


def _is_clickable(desc: str) -> bool:
    return ";click" in (desc or "")


def _is_editable(desc: str) -> bool:
    desc = (desc or "").lower()
    return "edittext" in desc or "autocompletetextview" in desc or "searchview" in desc


def _describe_element(desc: str) -> str:
    if not desc:
        return ""
    tail = HASH_RE.sub("", desc).split(";")[-1].strip()
    tail = tail.split(":")[0].strip()
    tokens = tail.split()
    if len(tokens) >= 2 and tokens[0] == tokens[1]:
        tokens = [tokens[0]] + tokens[2:]
    return " ".join(tokens).strip()


def _py_literal(value: str) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _first_line_text(value: str) -> str:
    text = str(value or "").strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
    return text.strip().splitlines()[0].strip().strip('"').strip("'") if text.strip() else ""


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...[truncated middle of XML]...\n"
    keep = max_chars - len(marker)
    head = int(keep * 0.7)
    tail = keep - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def _recent_actions(history: Sequence[Dict[str, str]], limit: int = 4) -> str:
    actions: List[str] = []
    for msg in history:
        if msg.get("role") == "assistant":
            actions.append(str(msg.get("content", "")).strip().splitlines()[-1])
    return "\n".join(actions[-limit:]) if actions else "None"


def _slim_xml(xml_text: str, max_chars: int) -> str:
    lines: List[str] = []
    pending: Optional[str] = None
    for raw in (xml_text or "").splitlines():
        line = HASH_RE.sub("", raw).rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("bounds:") and pending:
            bounds = stripped.split(":", 1)[1].strip()
            pending = f"{pending} @{bounds}"
            continue
        if pending:
            lines.append(_with_visible_label(pending))
        pending = line
    if pending:
        lines.append(_with_visible_label(pending))
    return _truncate_middle("\n".join(lines), max_chars)


def _with_visible_label(line: str) -> str:
    clean = HASH_RE.sub("", line).strip()
    if clean.startswith("[") or "{" in clean or "}" in clean:
        return clean
    label = clean.split(";")[-1].split(":")[0].strip()
    label = " ".join(label.split())
    return f"[{label or 'no-label'}] {clean}"


class RankThenVerifyTask(TextOnlyTask):
    """Unified No-CoT Generate, Rank-then-Verify, and exhaustive Yes/No verify.

    rtv_k=1 executes the top ranked candidate, matching No-CoT Generate.
    1 < rtv_k < candidate_count ranks first, then verifies each top-k candidate.
    rtv_k=max verifies every candidate, matching full-candidate verify.
    """

    CONTROL_ACTIONS = [
        ("control", "Press Back", 'do(action="Back")', {"action": "Back"}),
        ("control", "Press Home", 'do(action="Home")', {"action": "Home"}),
        ("control", "Press Enter", 'do(action="Enter")', {"action": "Enter"}),
        (
            "control",
            "Swipe up",
            'do(action="Swipe", direction="up", dist="medium")',
            {"action": "Swipe", "direction": "up", "dist": "medium"},
        ),
        (
            "control",
            "Swipe down",
            'do(action="Swipe", direction="down", dist="medium")',
            {"action": "Swipe", "direction": "down", "dist": "medium"},
        ),
        (
            "control",
            "Swipe left",
            'do(action="Swipe", direction="left", dist="medium")',
            {"action": "Swipe", "direction": "left", "dist": "medium"},
        ),
        (
            "control",
            "Swipe right",
            'do(action="Swipe", direction="right", dist="medium")',
            {"action": "Swipe", "direction": "right", "dist": "medium"},
        ),
    ]

    def set_system_prompt(self, instruction: str) -> None:
        self.instruction = instruction
        self.record.history = [{
            "role": "system",
            "content": RTV_SYSTEM_PROMPT.format(instruction=instruction),
        }]

    @property
    def params(self) -> Dict[str, Any]:
        return getattr(self.agent, "kwargs", {}) or {}

    def _k_value(self, total: int) -> int:
        raw = self.params.get("rtv_k", 3)
        if isinstance(raw, str) and raw.strip().lower() in {"max", "all", "full"}:
            return total
        try:
            return max(1, min(total, int(raw)))
        except Exception:
            return min(total, 3)

    def _latest_xml_text(self) -> str:
        raw = self.record.get_latest_xml() or "(empty screen)"
        return _slim_xml(raw, int(self.params.get("rtv_xml_max_chars", 16000)))

    def _latest_activity(self) -> str:
        if getattr(self.record, "contents", None):
            return str(self.record.contents[-1].get("current_activity", "unknown"))
        return "unknown"

    def _enumerate_candidates(self) -> List[Candidate]:
        try:
            latest = self.record.contents[-1]
            xml_path = latest["xml"] if latest.get("xml") not in {"ERROR", None} else latest.get("ac_xml")
            xml_tree = json.loads(get_compressed_xml(xml_path, type="json")) if xml_path else None
            elements = extract_bounds(xml_tree) if xml_tree else []
        except Exception:
            elements = []

        raw: List[Dict[str, Any]] = []
        seen_bounds = set()
        for elem in elements:
            desc = str(elem.get("key", "")).strip()
            bounds = _parse_bounds(str(elem.get("value", "")))
            if bounds is None or tuple(bounds) in seen_bounds:
                continue
            if not _is_clickable(desc):
                continue
            seen_bounds.add(tuple(bounds))
            text = _describe_element(desc)
            tag = f"[{text}]" if text else "[no-label]"
            raw.append({
                "kind": "tap",
                "description": f"Tap {tag} at {bounds}",
                "code": f'do(action="Tap", element={bounds})  # target="{text}"',
                "action": {"action": "Tap", "element": bounds},
                "target_text": text,
                "bbox": bounds,
            })
            if _is_editable(desc):
                raw.append({
                    "kind": "type",
                    "description": f"Type text into {tag} at {bounds}",
                    "code": f'do(action="Type", element={bounds}, text="<TEXT_TO_FILL>")',
                    "code_template": f'do(action="Type", element={bounds}, text={{text}})',
                    "needs_fill": "type",
                    "action": {"action": "Type", "element": bounds, "text": "<FILL>"},
                    "target_text": text,
                    "bbox": bounds,
                })

        if _bool(self.params.get("rtv_include_type_action"), False):
            raw.append({
                "kind": "type",
                "description": "Type the needed text into the focused input field",
                "code": 'do(action="Type", text="<TEXT_TO_FILL>")',
                "code_template": 'do(action="Type", text={text})',
                "needs_fill": "type",
                "action": {"action": "Type", "text": "<FILL>"},
            })

        if _bool(self.params.get("rtv_include_control_actions"), True):
            for kind, desc, code, action in self.CONTROL_ACTIONS:
                raw.append({
                    "kind": kind,
                    "description": desc,
                    "code": code,
                    "action": action,
                })

        raw.append({
            "kind": "finish",
            "description": "Finish the task with a concise answer/status message",
            "code": 'finish(message="<MESSAGE_TO_FILL>")',
            "code_template": 'finish(message={text})',
            "needs_fill": "finish",
            "action": {"operation": "finish", "message": "<FILL>"},
        })

        max_candidates = min(int(self.params.get("rtv_max_candidates", len(ALPHABET))), len(ALPHABET))
        if max_candidates > 0 and len(raw) > max_candidates:
            fixed = [x for x in raw if x["kind"] in {"control", "finish"}]
            dynamic = [x for x in raw if x["kind"] not in {"control", "finish"}]
            raw = dynamic[:max(0, max_candidates - len(fixed))] + fixed

        return [Candidate(label=_label_for_index(i), **item) for i, item in enumerate(raw)]

    def _candidate_block(self, candidates: Sequence[Candidate]) -> str:
        return "\n".join(
            f"{candidate.label}. {candidate.description}\n   code: {candidate.code}"
            for candidate in candidates
        )

    def _score_choice(self, xml_text: str, candidates: Sequence[Candidate], template: str) -> List[Candidate]:
        if len(candidates) <= 1:
            return list(candidates)
        if len(candidates) > len(ALPHABET):
            winners: List[Candidate] = []
            for start in range(0, len(candidates), len(ALPHABET)):
                chunk = [
                    Candidate(label=ALPHABET[i], **{
                        "kind": c.kind,
                        "description": c.description,
                        "code": c.code,
                        "action": c.action,
                        "target_text": c.target_text,
                        "bbox": c.bbox,
                        "needs_fill": c.needs_fill,
                        "code_template": c.code_template,
                    })
                    for i, c in enumerate(candidates[start:start + len(ALPHABET)])
                ]
                winners.extend(self._score_choice(xml_text, chunk, template)[:1])
            return self._score_choice(xml_text, winners, template)

        relabeled = [
            Candidate(label=ALPHABET[i], **{
                "kind": c.kind,
                "description": c.description,
                "code": c.code,
                "action": c.action,
                "target_text": c.target_text,
                "bbox": c.bbox,
                "needs_fill": c.needs_fill,
                "code_template": c.code_template,
            })
            for i, c in enumerate(candidates)
        ]
        prompt = template.format(
            instruction=self.instruction,
            xml_text=xml_text,
            recent_actions=_recent_actions(self.record.history),
            candidate_block=self._candidate_block(relabeled)
        )
        messages = [self.record.history[0], {"role": "user", "content": prompt}]
        scores = self.agent.score_letters(messages, [c.label for c in relabeled])
        by_label = {score["letter"]: score for score in scores}
        ranked_pairs = sorted(
            zip(relabeled, candidates),
            key=lambda pair: by_label.get(pair[0].label, {}).get("prob", 0.0),
            reverse=True,
        )
        return [original for _, original in ranked_pairs]

    def _rank_candidates(self, xml_text: str, candidates: Sequence[Candidate], k: int) -> List[Candidate]:
        return self._score_choice(xml_text, candidates, RANK_USER_TEMPLATE)[:k]

    def _verify_messages(
        self,
        xml_text: str,
        candidate: Candidate,
        ranked_candidates: Sequence[Candidate],
    ) -> List[Dict[str, str]]:
        prompt = VERIFY_USER_TEMPLATE.format(
            instruction=self.instruction,
            xml_text=xml_text,
            recent_actions=_recent_actions(self.record.history),
            candidate_block=self._candidate_block(ranked_candidates),
            candidate_label=candidate.label,
            candidate_description=candidate.description,
            candidate_code=candidate.code,
        )
        return [{"role": "system", "content": VERIFY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    def _verify_candidates(self, xml_text: str, candidates: Sequence[Candidate]) -> List[tuple[Candidate, Dict[str, float]]]:
        fail = {
            "yes_logit": float("-inf"),
            "no_logit": float("-inf"),
            "yes_prob": 0.0,
            "no_prob": 1.0,
        }

        def _score_one(candidate: Candidate) -> tuple[Candidate, Dict[str, float]]:
            if hasattr(self.agent, "score_yes_no"):
                try:
                    return candidate, self.agent.score_yes_no(self._verify_messages(xml_text, candidate, candidates))
                except Exception as exc:
                    score = dict(fail)
                    score["error"] = str(exc)
                    return candidate, score
            try:
                rsp = self.agent.act(self._verify_messages(xml_text, candidate, candidates))
                yes = 1.0 if _first_line_text(rsp).lower().startswith("yes") else 0.0
                return candidate, {"yes_logit": yes, "no_logit": 1.0 - yes, "yes_prob": yes, "no_prob": 1.0 - yes}
            except Exception as exc:
                score = dict(fail)
                score["error"] = str(exc)
                return candidate, score

        workers = max(1, min(int(self.params.get("rtv_verify_workers", 8)), len(candidates)))
        if workers <= 1:
            scored = [_score_one(candidate) for candidate in candidates]
        else:
            scored: List[tuple[Candidate, Dict[str, float]]] = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_score_one, candidate) for candidate in candidates]
                for future in as_completed(futures):
                    scored.append(future.result())
        return sorted(scored, key=lambda item: item[1].get("yes_logit", float("-inf")), reverse=True)

    def _fill_candidate(self, xml_text: str, candidate: Candidate) -> Candidate:
        if not candidate.needs_fill or not candidate.code_template:
            return candidate

        if candidate.needs_fill == "type":
            tail = FILL_TYPE_TAIL.format(
                element_desc=candidate.description,
                bbox=candidate.bbox if candidate.bbox is not None else "focused input",
            )
        else:
            tail = FILL_FINISH_TAIL
        prompt = f"""Goal:
{self.instruction}

Current activity:
{self._latest_activity()}

Current XML:
{xml_text}

Recent actions:
{_recent_actions(self.record.history)}

Chosen candidate:
{candidate.label}. {candidate.description}
code template: {candidate.code}

{tail}"""
        messages = [{"role": "system", "content": FILL_SYSTEM}, {"role": "user", "content": prompt}]
        if hasattr(self.agent, "generate_text"):
            filled = self.agent.generate_text(
                messages,
                max_tokens=int(self.params.get("rtv_fill_max_tokens", 128)),
                temperature=0,
            )
        else:
            filled = self.agent.act(messages)
        filled = _first_line_text(filled)
        code = candidate.code_template.format(text=_py_literal(filled))
        action = dict(candidate.action)
        if candidate.needs_fill == "type":
            action["text"] = filled
        else:
            action["message"] = filled
        return Candidate(
            label=candidate.label,
            kind=candidate.kind,
            description=candidate.description,
            code=code,
            action=action,
            target_text=candidate.target_text,
            bbox=candidate.bbox,
        )

    def run_step(self, round_count: int) -> None:
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility)
        xml_text = self._latest_xml_text()
        candidates = self._enumerate_candidates()
        if not candidates:
            final = Candidate(
                label="A",
                kind="control",
                description="Press Back",
                code='do(action="Back")',
                action={"action": "Back"},
            )
            ranked = [final]
        else:
            k = self._k_value(len(candidates))
            ranked = list(candidates) if k >= len(candidates) else self._rank_candidates(xml_text, candidates, k)
            verified = [] if k == 1 else self._verify_candidates(xml_text, ranked)
            final = ranked[0] if k == 1 else verified[0][0]

        final = self._fill_candidate(xml_text, final)
        summary = [
            "## Rank-then-Verify",
            f"Ranked: {', '.join(c.label for c in ranked)}",
            "Verify: skipped" if len(ranked) <= 1 else "Verify yes logits: " + ", ".join(
                f"{candidate.label}={score.get('yes_logit', 0.0):.3f}"
                for candidate, score in verified
            ),
            f"Chosen: {final.label} {final.description}",
            "",
            "```python",
            final.code,
            "```",
        ]
        rsp = "\n".join(summary)
        exe_res = self.page_executor(final.code)
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1
