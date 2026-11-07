from src.evaluation import attach_gap_truth, gap_detection_report, planted_dark_windows


def test_all_detectable_planted_blackouts_found(synth, processed):
    gt = attach_gap_truth(processed["gaps"], synth["truth_reports"], processed["id_map"])
    dw = planted_dark_windows(synth["truth_reports"], processed["id_map"])
    rep = gap_detection_report(gt, dw, 60)
    assert rep["planted_dark_windows_detectable"] > 0
    assert rep["planted_blackout_recall"] == 1.0


def test_benign_gaps_exist(synth, processed):
    gt = attach_gap_truth(processed["gaps"], synth["truth_reports"], processed["id_map"])
    # the detector must not be mistaken for a suspicion label: benign gaps are common
    assert (gt.truth_cause != "dark_intentional").sum() > (gt.truth_cause == "dark_intentional").sum()
