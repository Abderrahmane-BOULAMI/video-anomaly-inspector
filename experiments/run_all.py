"""Reproduce every number of the README (about 4 minutes on a laptop).

    python experiments/run_all.py
"""
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vidanom import assist, wear  # noqa: E402
from vidanom.detect import (EnvelopeDetector, ForestDetector, PixelPCADetector, RuleDetector,  # noqa: E402
                            score_events, segments)
from vidanom.features import extract  # noqa: E402
from vidanom.synth import FPS, TYPES, generate  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
LIMIT_PX = 4.0
QUALITY = ("dark_frac",)  # share of near-black pixels: a covered or failing camera
KIND_TO_KB = {"jam": "KB-01", "misaligned": "KB-02", "foreign_object": "KB-03", "smoke": "KB-04",
              "camera_fault": "KB-05", "wear": "KB-06"}


def evaluate(stream, df, thumbs, rules, env, forest, pca):
    lab = rules.predict(df)
    typed = score_events(stream.events, {k: lab == k for k in TYPES})
    generic = pd.concat([
        score_events(stream.events, {"any": lab != "normal"}).assign(detector="typed rules"),
        score_events(stream.events, {"any": env.predict(df)}).assign(detector="learned envelope"),
        score_events(stream.events, {"any": forest.predict(df)}).assign(detector="Isolation Forest"),
        score_events(stream.events, {"any": pca.predict(thumbs)}).assign(detector="PCA on pixels"),
    ])
    amp = wear.amplitude(df)
    onset = wear.cusum_onset(amp)
    alert = wear.maintenance_alert(amp, LIMIT_PX, onset=onset)
    w = {"true_start_s": stream.wear_start / FPS, "onset_detected_s": onset,
         "true_limit_crossing_s": float(np.argmax(stream.wobble_px >= LIMIT_PX) / FPS),
         "measured_limit_crossing_s": wear.first_crossing(amp, LIMIT_PX), "alert": alert}
    return lab, typed, generic, amp, w


def diagnose_events(stream, df, env, kb):
    """Generic detections -> evidence -> retrieval -> offline answer; score the retrieved entry."""
    flags = env.predict(df)
    ex = env.excess(df)
    segs = segments(flags)
    frames = assist.keyframes(stream, segs)
    rows, prompts = [], []
    backend = assist.OfflineBackend()
    for i, ((a, b), kf) in enumerate(zip(segs, frames)):
        mean_ex = ex.iloc[a:b + 1].mean().sort_values(ascending=False)
        out = tuple(m for m, v in mean_ex.items() if v > env.margin)[:3]
        # image-quality gate: when the image itself is degraded, the other measurements are not trustworthy
        quality = tuple(m for m in out if m in QUALITY)
        out = quality or out
        ev = assist.Evidence(i, a / FPS, b / FPS, None, out[0] if out else mean_ex.index[0],
                             float(mean_ex[out[0]] if out else mean_ex.iloc[0]), kf, out)
        hits = kb.search(ev.text(), 2)
        prompt = assist.build_prompt(ev, hits)
        ans = backend.diagnose(ev, hits, prompt)
        truth = [e.kind for e in stream.events if a <= e.end and b >= e.start]
        rows.append({"segment": i, "start_s": a / FPS, "end_s": b / FPS, "out_of_range": ", ".join(out),
                     "true_kind": truth[0] if truth else "none (false alarm)", "kb_id": ans["kb_id"],
                     "kb_correct": bool(truth) and ans["kb_id"] == KIND_TO_KB[truth[0]],
                     "schema_errors": len(assist.validate(ans)), "answer": ans, "keyframes": kf})
        if i < 2:
            prompts.append({"prompt": prompt, "offline_answer": ans})
    return rows, prompts


def timeline(stream, lab, env_flags, path):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    colors = {"jam": "#d1495b", "misaligned": "#edae49", "foreign_object": "#00798c", "smoke": "#8d6a9f",
              "camera_fault": "#30638e"}
    t = np.arange(stream.n_frames) / FPS
    for row, (name, arr) in enumerate([("truth", stream.labels), ("typed rules", lab)]):
        for k, c in colors.items():
            m = arr == k
            ax.fill_between(t, row + 0.1, row + 0.9, where=m, color=c, step="mid", label=k if row == 0 else None)
    ax.fill_between(t, 2.1, 2.9, where=env_flags, color="#555555", step="mid")
    ax.axvline(stream.wear_start / FPS, color="#999999", ls="--", lw=1)
    ax.text(stream.wear_start / FPS + 3, 3.05, "fan wear starts", fontsize=8, color="#666666")
    ax.set_yticks([0.5, 1.5, 2.5], ["truth", "typed rules", "learned envelope"])
    ax.set_xlabel("time (s)")
    ax.set_ylim(0, 3.3)
    ax.legend(ncol=5, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.35), frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def wear_plot(amp, w, path):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(amp["t_s"], amp["amp_px"], "o-", ms=3, color="#00798c", label="measured wobble (10 s windows)")
    ax.axhline(LIMIT_PX, color="#d1495b", lw=1, label="alarm limit")
    if w["onset_detected_s"]:
        ax.axvline(w["onset_detected_s"], color="#999999", ls="--", lw=1, label="degradation detected (CUSUM)")
    if w["alert"]:
        ax.axvline(w["alert"]["alert_s"], color="#edae49", lw=2, label="maintenance alert (trend)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hub wobble (px)")
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def annotated_gif(stream, lab, env, df, path, spans):
    """Short GIF of a few events with the detector's label drawn on the frame."""
    import imageio.v2 as imageio
    want = set()
    for a, b in spans:
        want.update(range(a, b, 3))
    ex = env.explain(df)
    flags = env.predict(df)
    imgs = []
    for i, fr in enumerate(stream.frames()):
        if i in want:
            img = fr.copy()
            k = lab[i]
            col = (60, 200, 60) if k == "normal" else (60, 60, 230)
            cv2.rectangle(img, (0, 0), (319, 22), (20, 20, 20), -1)
            cv2.putText(img, f"t={i / FPS:5.1f}s  rules: {k}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            if flags[i]:
                cv2.putText(img, f"envelope: {ex.iloc[i]}", (5, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if i > max(want):
            break
    imageio.mimsave(path, imgs, duration=0.15, loop=0)


def report_html(rows, w, path):
    tr = []
    for r in rows:
        kf = "".join(f'<img src="data:image/png;base64,{assist.to_b64png(im)}" width="240">' for im in r["keyframes"])
        a = r["answer"]
        tr.append(f"<tr><td>{r['start_s']:.0f} à {r['end_s']:.0f} s</td><td>{kf}</td>"
                  f"<td>{html.escape(r['out_of_range'])}</td><td><b>{html.escape(a['summary'])}</b><br>"
                  f"Cause probable : {html.escape(a['likely_cause'])}<br>Vérifications : "
                  f"{html.escape('; '.join(a['checks']))}<br>Urgence : {a['urgency']} ({a['kb_id']})</td>"
                  f"<td>{html.escape(r['true_kind'])}</td></tr>")
    al = w["alert"]
    doc = f"""<!doctype html><meta charset="utf-8"><title>Rapport d'anomalies vidéo</title>
<style>body{{font-family:system-ui;margin:24px;color:#1d2a33}}td,th{{border-bottom:1px solid #ddd;padding:6px;
vertical-align:top;font-size:13px}}table{{border-collapse:collapse}}img{{margin-right:4px}}</style>
<h1>Rapport d'anomalies, poste M-12 (vidéo simulée)</h1>
<p>Usure du ventilateur : dégradation détectée à {w['onset_detected_s']:.0f} s ; alerte de maintenance à
{al['alert_s']:.0f} s (franchissement prévu vers {al['predicted_cross_s']:.0f} s) ; limite franchie à
{w['measured_limit_crossing_s']:.0f} s.</p>
<table><tr><th>Période</th><th>Avant / pendant</th><th>Mesures hors plage</th><th>Diagnostic assisté
(hors ligne)</th><th>Vérité terrain</th></tr>{''.join(tr)}</table>"""
    Path(path).write_text(doc, encoding="utf-8")


def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    normal = generate(2000, seed=1, anomalies=False, wear=False)
    dn, tn = extract(normal.frames(), thumbs=True)
    rules, env = RuleDetector().fit(dn), EnvelopeDetector().fit(dn)
    forest, pca = ForestDetector().fit(dn), PixelPCADetector().fit(tn)
    summary = {"normal_clip_s": normal.n_frames / FPS,
               "false_frames_on_normal_clip": {"rules": int((rules.predict(dn) != "normal").sum()),
                                               "envelope": int(env.predict(dn).sum())}}
    kb = assist.KnowledgeBase()
    for name, seed in (("A", 0), ("B", 2)):
        st = generate(6000, seed=seed)
        df, th = extract(st.frames(), thumbs=True)
        lab, typed, generic, amp, w = evaluate(st, df, th, rules, env, forest, pca)
        summary[f"clip_{name}"] = {"seconds": st.n_frames / FPS, "events": len(st.events),
                                   "typed_rules": typed.to_dict("records"),
                                   "generic": generic.to_dict("records"), "wear": w}
        rows, prompts = diagnose_events(st, df, env, kb)
        true_rows = [r for r in rows if not r["true_kind"].startswith("none")]
        summary[f"clip_{name}"]["assisted_diagnosis"] = {
            "segments": len(rows), "on_true_events": len(true_rows),
            "kb_top1_correct": sum(r["kb_correct"] for r in true_rows), "schema_errors": sum(r["schema_errors"] for r in rows),
            "table": [{k: v for k, v in r.items() if k not in ("answer", "keyframes")} for r in rows]}
        if name == "A":
            timeline(st, lab, env.predict(df), FIG / "fig1_timeline.png")
            wear_plot(amp, w, FIG / "fig2_wear.png")
            report_html(rows, w, RES / "report.html")
            (RES / "prompt_examples.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1, default=str),
                                                     encoding="utf-8")
            pick = {}
            for e in st.events:
                pick.setdefault(e.kind, e)
            spans = [(pick[k].start - 15, min(pick[k].start + 45, pick[k].end + 10)) for k in
                     ("jam", "foreign_object", "smoke") if k in pick]
            annotated_gif(st, lab, env, df, FIG / "demo.gif", spans)
    summary["runtime_s"] = round(time.time() - t0)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
