from agentzones.knowledge import FixStore


def test_add_merge_search_persist(tmp_path):
    s = FixStore(tmp_path / "f.json")  # no verifier: accepts any well-formed attestation
    f = s.add_verified(problem="tool_ca_bundle", summary="point pip at system CA",
                       steps=["pip config set global.cert x"], zone="a", sign=lambda d: "sig",
                       symptoms="CERTIFICATE_VERIFY_FAILED pip")
    other = {**f, "attestations": [{"zone": "b", "ts": 1.0, "sig": "x"}]}
    _, n, _ = s.merge(other)
    assert n == 1 and s.fixes[f["id"]]["verified_by"] == ["a", "b"]
    assert s.search("tool_ca_bundle")[0]["id"] == f["id"]
    assert s.search("clock_skew") == []
    assert s.search(text="pip certificate")[0]["id"] == f["id"]
    assert FixStore(tmp_path / "f.json").fixes[f["id"]]["verified_by"] == ["a", "b"]


def test_teacher_answer_to_soft_labels():
    from proxy_doctor.catalog import CLASSES
    from proxy_doctor.distill.teacher import to_probs
    p = to_probs({"problem": "tool_ca_bundle", "confidence": 0.6, "alternatives": {"untrusted_corp_ca": 0.4,
                                                                                    "bogus": 0.9}})
    assert abs(sum(p) - 1) < 1e-9
    assert p[CLASSES.index("tool_ca_bundle")] == 0.6 and p[CLASSES.index("untrusted_corp_ca")] == 0.4
    assert to_probs({}) == [1 / len(CLASSES)] * len(CLASSES)
