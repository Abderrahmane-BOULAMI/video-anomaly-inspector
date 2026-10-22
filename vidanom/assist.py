"""Assisted diagnosis: turn a detected event into evidence, retrieve the matching knowledge, ask a model.

The evidence is built from measurements, never from the model's imagination: times, durations, which
measurement left its normal range and by how much, and two key frames (before, during). A multimodal LLM
(Gemini on Vertex AI) can then write the operator message from that evidence and the retrieved knowledge.
Without cloud access, an offline backend fills the same JSON structure from the knowledge base, so the
pipeline and its tests run anywhere.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .synth import FPS

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "diagnostics.yaml"
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "one sentence for the operator"},
        "likely_cause": {"type": "string"},
        "checks": {"type": "array", "items": {"type": "string"}},
        "urgency": {"type": "string", "enum": ["planifiable", "moyenne", "haute", "critique"]},
        "kb_id": {"type": "string"},
        "confidence_note": {"type": "string"},
    },
    "required": ["summary", "likely_cause", "checks", "urgency", "kb_id"],
}
# plain-language description of each measurement, used to build the retrieval query
MEASURE_TEXT = {
    "speed": "belt speed drops, conveyor stopped",
    "max_abs_angle": "part rotated on the belt, angle out of tolerance",
    "hue_off": "object colour not matching the part",
    "min_fill": "object shape not matching the part",
    "mach_mean": "machine area brighter",
    "mach_std": "contrast change in front of the machine housing, grey plume",
    "dark_frac": "large part of the frame black, lens covered",
    "brightness": "image dark",
    "sharpness": "low sharpness",
    "n_obj": "unexpected number of objects on the belt",
}
KIND_TEXT = {
    "jam": "belt speed drops to zero, parts stop moving",
    "misaligned": "part rotated on the belt",
    "foreign_object": "unknown object on the belt, colour not matching the part",
    "smoke": "grey plume in front of the machine housing",
    "camera_fault": "large part of the frame black, lens covered",
    "wear": "fan hub wobble increasing",
}


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        self.entries = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.docs = [f"{e['title']} {e['symptom']} {' '.join(e['causes'])}" for e in self.entries]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(self.docs)
        self.M = self.vec.transform(self.docs)

    def search(self, query: str, k: int = 2) -> list[tuple[dict, float]]:
        sims = cosine_similarity(self.vec.transform([query]), self.M)[0]
        order = np.argsort(-sims)[:k]
        return [(self.entries[i], float(sims[i])) for i in order]


@dataclass
class Evidence:
    event_id: int
    start_s: float
    end_s: float
    kind: str | None          # typed detector output, None for a generic detection
    top_measure: str | None   # generic detector: measurement furthest outside its range
    excess: float | None
    keyframes: list           # BGR images
    out_of_range: tuple = ()  # generic detector: every measurement outside its range, worst first

    def text(self) -> str:
        if self.kind:
            return KIND_TEXT[self.kind]
        return ", ".join(MEASURE_TEXT[m] for m in (self.out_of_range or (self.top_measure,)) if m in MEASURE_TEXT)

    def as_dict(self) -> dict:
        return {"event_id": self.event_id, "start_s": round(self.start_s, 1), "end_s": round(self.end_s, 1),
                "duration_s": round(self.end_s - self.start_s, 1), "detector_label": self.kind,
                "top_measure": self.top_measure, "excess_vs_normal_range": None if self.excess is None
                else round(self.excess, 2), "observation": self.text()}


def build_prompt(ev: Evidence, kb_hits: list[tuple[dict, float]]) -> str:
    kb = [{"kb_id": e["id"], "title": e["title"], "causes": e["causes"], "checks": e["checks"],
           "urgency": e["urgency"], "similarity": round(s, 2)} for e, s in kb_hits]
    return (
        "Tu assistes une équipe de maintenance d'une usine automobile. Une anomalie a été détectée sur le flux "
        "vidéo d'un poste. Les deux images jointes montrent le poste avant et pendant l'événement.\n"
        "Règles : n'utilise que les mesures et la base de connaissances fournies ; si les images contredisent "
        "les mesures, dis-le ; ne propose aucune action dangereuse ; réponds en français, au format JSON demandé.\n"
        f"Mesures : {json.dumps(ev.as_dict(), ensure_ascii=False)}\n"
        f"Base de connaissances (extraits les plus proches) : {json.dumps(kb, ensure_ascii=False)}\n"
    )


class OfflineBackend:
    """Deterministic answer from the best knowledge entry (no model call)."""
    name = "offline"

    def diagnose(self, ev: Evidence, kb_hits, prompt: str) -> dict:
        e, sim = kb_hits[0]
        return {"summary": f"{e['title']} détecté(e) de {ev.start_s:.0f} s à {ev.end_s:.0f} s : {ev.text()}.",
                "likely_cause": e["causes"][0], "checks": e["checks"], "urgency": e["urgency"], "kb_id": e["id"],
                "confidence_note": f"similarité avec la fiche {sim:.2f}, réponse sans modèle de langage"}


class VertexGeminiBackend:
    """Gemini on Vertex AI through the google-genai SDK. Needs GOOGLE_CLOUD_PROJECT and credentials.

    Not exercised by the tests (no cloud access in CI); the offline backend produces the same structure.
    """
    name = "vertex-gemini"

    def __init__(self, model: str = "gemini-2.5-flash", location: str = "europe-west1"):
        from google import genai  # imported lazily: optional dependency
        from google.genai import types
        self.types = types
        self.client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=location)
        self.model = model

    def diagnose(self, ev: Evidence, kb_hits, prompt: str) -> dict:
        parts = [self.types.Part.from_bytes(data=cv2.imencode(".png", im)[1].tobytes(), mime_type="image/png")
                 for im in ev.keyframes]
        resp = self.client.models.generate_content(
            model=self.model, contents=[*parts, prompt],
            config=self.types.GenerateContentConfig(response_mime_type="application/json",
                                                    response_schema=OUTPUT_SCHEMA, temperature=0.1))
        return json.loads(resp.text)


def validate(answer: dict) -> list[str]:
    """Reject an answer that does not follow the schema or cites an unknown knowledge entry."""
    errs = [f"missing {k}" for k in OUTPUT_SCHEMA["required"] if k not in answer]
    if answer.get("urgency") not in OUTPUT_SCHEMA["properties"]["urgency"]["enum"]:
        errs.append("bad urgency")
    ids = {e["id"] for e in KnowledgeBase().entries}
    if answer.get("kb_id") not in ids:
        errs.append("unknown kb_id")
    return errs


def keyframes(stream, spans: list[tuple[int, int]], before: int = 20) -> list[list]:
    """For each (start, end) span: the frame shortly before it and the frame in its middle, in one pass."""
    want = {}
    for j, (s, e) in enumerate(spans):
        want.setdefault(max(0, s - before), []).append((j, 0))
        want.setdefault((s + e) // 2, []).append((j, 1))
    out = [[None, None] for _ in spans]
    last = max(want) if want else -1
    for i, fr in enumerate(stream.frames()):
        for j, k in want.get(i, []):
            out[j][k] = fr
        if i >= last:
            break
    return out


def to_b64png(img) -> str:
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def frames_to_s(f: int) -> float:
    return f / FPS
