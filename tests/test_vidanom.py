import numpy as np
import pandas as pd
import pytest

from vidanom import assist, wear
from vidanom.detect import EnvelopeDetector, ForestDetector, RuleDetector, score_events, segments
from vidanom.features import belt_objects, belt_speed, extract
from vidanom.synth import BELT_Y0, BELT_Y1, FPS, SPEED, Event, generate


@pytest.fixture(scope="module")
def normal():
    st = generate(500, seed=11, anomalies=False, wear=False)
    df, _ = extract(st.frames())
    return st, df


@pytest.fixture(scope="module")
def clip():
    st = generate(1600, seed=3, wear=False)
    df, _ = extract(st.frames())
    return st, df


def test_rendering_is_deterministic_and_labels_match_events():
    a = generate(40, seed=5, anomalies=False).frame_array()
    b = generate(40, seed=5, anomalies=False).frame_array()
    assert np.array_equal(a, b) and a.shape == (40, 240, 320, 3)
    st = generate(3000, seed=5)
    for e in st.events:
        assert (st.labels[e.start:e.end + 1] == e.kind).all()
    assert (st.labels == "normal").sum() == 3000 - sum(e.end - e.start + 1 for e in st.events)


def test_belt_speed_measures_motion(normal):
    st, df = normal
    assert df["speed"].median() == pytest.approx(SPEED, abs=0.3)


def test_parts_are_found_upright(normal):
    st, df = normal
    fr = next(iter(st.frames()))
    objs = belt_objects(fr)
    assert all(abs(o["angle"]) < 8 for o in objs)
    assert df["n_obj"].median() >= 3


def test_rules_quiet_on_normal_and_find_every_event(normal, clip):
    rules = RuleDetector().fit(normal[1])
    assert (rules.predict(normal[1]) != "normal").sum() == 0
    st, df = clip
    lab = rules.predict(df)
    kinds = sorted({e.kind for e in st.events})
    res = score_events(st.events, {k: lab == k for k in kinds})
    assert (res["found"] == res["events"]).all()
    assert res["false_alarms"].sum() == 0


def test_envelope_explains_jam_by_speed(normal, clip):
    env = EnvelopeDetector().fit(normal[1])
    st, df = clip
    jam = [e for e in st.events if e.kind == "jam"]
    if jam:
        e = jam[0]
        assert env.explain(df).iloc[e.start + 10:e.end - 5].mode()[0] == "speed"


def test_isolation_forest_is_blind_to_features_constant_in_training(normal):
    """The documented limitation: a measurement that never moved in normal footage is never split on."""
    f = ForestDetector().fit(normal[1])
    df = normal[1].copy()
    df.loc[:, "max_abs_angle"] = 35.0  # every part rotated
    env = EnvelopeDetector().fit(normal[1])
    assert env.predict(df).mean() > 0.9
    assert f.predict(df).mean() < env.predict(df).mean()


def test_segments_merge_and_filter():
    flags = np.zeros(40, bool)
    flags[[2, 3, 4, 7, 8, 30]] = True
    assert segments(flags, min_len=3, gap=5) == [(2, 8)]


def test_score_events_counts_false_alarms_and_delay():
    flags = np.zeros(100, bool)
    flags[15:25] = True
    flags[60:70] = True
    r = score_events([Event("jam", 10, 30)], {"jam": flags}).iloc[0]
    assert r.found == 1 and r.false_alarms == 1 and r.mean_delay_s == pytest.approx(5 / FPS)


def test_wear_onset_and_alert_before_limit():
    rng = np.random.default_rng(0)
    t = np.arange(0, 600, 10.0) + 10
    a = np.where(t < 300, 0.5, 0.5 + 6 * (np.clip(t - 300, 0, None) / 300) ** 1.5) + rng.normal(0, 0.05, len(t))
    amp = pd.DataFrame({"t_s": t, "amp_px": a})
    onset = wear.cusum_onset(amp)
    assert 300 <= onset <= 360
    al = wear.maintenance_alert(amp, 4.0, onset=onset)
    assert al["alert_s"] < wear.first_crossing(amp, 4.0)


def test_amplitude_of_circle():
    th = np.linspace(0, 40 * np.pi, 200)
    df = pd.DataFrame({"hub_x": 60 + 3 * np.cos(th), "hub_y": 52 + 3 * np.sin(th)})
    assert wear.amplitude(df, window_s=20).amp_px.iloc[0] == pytest.approx(3 / np.sqrt(2) * np.sqrt(2), rel=0.05)


def test_knowledge_retrieval_matches_each_kind():
    kb = assist.KnowledgeBase()
    expect = {"jam": "KB-01", "misaligned": "KB-02", "foreign_object": "KB-03", "smoke": "KB-04",
              "camera_fault": "KB-05", "wear": "KB-06"}
    for kind, kb_id in expect.items():
        assert kb.search(assist.KIND_TEXT[kind], 1)[0][0]["id"] == kb_id


def test_offline_answer_follows_schema_and_validator_rejects_bad_ones():
    kb = assist.KnowledgeBase()
    ev = assist.Evidence(0, 10.0, 16.0, None, "speed", 3.2, [np.zeros((10, 10, 3), np.uint8)] * 2, ("speed",))
    hits = kb.search(ev.text(), 2)
    prompt = assist.build_prompt(ev, hits)
    assert "speed" in prompt and "KB-01" in prompt
    ans = assist.OfflineBackend().diagnose(ev, hits, prompt)
    assert assist.validate(ans) == [] and ans["kb_id"] == "KB-01"
    assert "unknown kb_id" in assist.validate({**ans, "kb_id": "KB-99"})
    assert "bad urgency" in assist.validate({**ans, "urgency": "whenever"})
