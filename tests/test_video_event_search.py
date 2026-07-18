from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest


WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "outputs" / "video-event-search-template" / "video_event_search.py"

spec = importlib.util.spec_from_file_location("video_event_search_module", SCRIPT_PATH)
ves = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ves
assert spec.loader is not None
spec.loader.exec_module(ves)


def fake_cfg(**overrides):
    values = {
        "score_threshold": 0.75,
        "candidate_padding": 1.0,
        "minimum_candidate_windows": 2,
        "max_candidate_windows": 2,
        "local_scan_interval": 1.0,
        "minimum_event_duration": 5.0,
        "minimum_positive_samples": 2,
        "lexical_match_terms": (),
        "lexical_caption_terms": (),
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class VideoEventSearchTests(unittest.TestCase):
    def test_extract_json_and_scan_times(self):
        self.assertEqual(ves.extract_json_object("```json\n{\"a\": 1}\n```"), {"a": 1})
        self.assertEqual(ves.scan_times(10.25, 2.0), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        self.assertEqual(ves.local_scan_times(0.0, 3.0, 1.0, 10.0), [0.0, 1.0, 2.0, 3.0])

    def test_load_config_accepts_current_schema_and_rejects_old_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"not a real video")

            input_path = tmp_path / "input.json"
            ves.write_json_atomic(input_path, {
                "video_path": "video.mp4",
                "output_directory": ".",
                "condition": "find this",
                "search": {
                    "strategy": "retrieve_verify",
                    "score_threshold": 0.8,
                    "minimum_candidate_windows": 1,
                    "max_candidate_windows": 1,
                },
            })

            cfg = ves.load_config(input_path, None, None)
            self.assertEqual(cfg.strategy, "retrieve_verify")
            self.assertEqual(cfg.score_threshold, 0.8)
            self.assertEqual(cfg.minimum_candidate_windows, 1)

            bad_path = tmp_path / "bad.json"
            ves.write_json_atomic(bad_path, {
                "video_path": "video.mp4",
                "output_directory": ".",
                "condition": "find this",
                "caption_interval_seconds": 30,
            })
            with self.assertRaisesRegex(ves.ConfigError, "old candidate_binary schema"):
                ves.load_config(bad_path, None, None)

    def test_retrieval_selection_uses_threshold_then_top_fallback(self):
        cfg = fake_cfg(score_threshold=0.75, minimum_candidate_windows=2, max_candidate_windows=2)
        samples = [
            {"time": 0.0, "retrieval_likelihood_score": 0.1, "retrieval_raw_score": 0.1, "retrieval_rank": 4, "selected_for_verification": False, "selection_reason": ""},
            {"time": 2.0, "retrieval_likelihood_score": 0.9, "retrieval_raw_score": 0.9, "retrieval_rank": 1, "selected_for_verification": False, "selection_reason": ""},
            {"time": 4.0, "retrieval_likelihood_score": 0.2, "retrieval_raw_score": 0.2, "retrieval_rank": 3, "selected_for_verification": False, "selection_reason": ""},
            {"time": 8.0, "retrieval_likelihood_score": 0.6, "retrieval_raw_score": 0.6, "retrieval_rank": 2, "selected_for_verification": False, "selection_reason": ""},
        ]

        windows = ves.select_retrieval_windows(cfg, 10.0, samples)

        self.assertEqual(len(windows), 2)
        self.assertTrue(any(window["peak_time"] == 2.0 for window in windows))
        self.assertTrue(any("top retrieval fallback" in sample["selection_reason"] for sample in samples if sample["selected_for_verification"]))

    def test_verification_normalization_requires_evidence_and_threshold(self):
        cfg = fake_cfg(score_threshold=0.75)
        accepted = ves.normalize_verification_result({
            "neutral_caption": "visible event",
            "event_phase": "match",
            "confidence_score": 0.8,
            "is_match": True,
            "required_visual_evidence": True,
        }, cfg)
        self.assertTrue(accepted["is_match"])

        below_threshold = ves.normalize_verification_result({
            "neutral_caption": "visible event",
            "event_phase": "match",
            "confidence_score": 0.7,
            "is_match": True,
            "required_visual_evidence": True,
        }, cfg)
        self.assertFalse(below_threshold["is_match"])

        missing_evidence = ves.normalize_verification_result({
            "neutral_caption": "visible event",
            "event_phase": "match",
            "confidence_score": 0.9,
            "is_match": True,
            "required_visual_evidence": False,
        }, cfg)
        self.assertFalse(missing_evidence["is_match"])

    def test_lexical_override_is_constrained_by_caption_terms(self):
        cfg = fake_cfg(
            score_threshold=0.75,
            lexical_match_terms=("butterfly", "trunk", "near"),
            lexical_caption_terms=("butterfly", "trunk"),
        )

        corrected = ves.normalize_verification_result({
            "neutral_caption": "A butterfly is near a tree trunk.",
            "event_phase": "near_miss",
            "confidence_score": 0.8,
            "is_match": False,
            "required_visual_evidence": False,
            "evidence": "The butterfly is near the trunk.",
            "negative_evidence": "",
        }, cfg)
        self.assertTrue(corrected["is_match"])
        self.assertEqual(corrected["event_phase"], "match")
        self.assertIn("lexical_match_terms", corrected["override_reason"])

        not_corrected = ves.normalize_verification_result({
            "neutral_caption": "A tree trunk is visible.",
            "event_phase": "near_miss",
            "confidence_score": 0.8,
            "is_match": False,
            "required_visual_evidence": False,
            "evidence": "The condition mentions a butterfly near the trunk.",
            "negative_evidence": "",
        }, cfg)
        self.assertFalse(not_corrected["is_match"])

    def test_occurrence_grouping_and_minimum_positive_samples(self):
        cfg = fake_cfg(local_scan_interval=1.0, minimum_event_duration=5.0, minimum_positive_samples=2)
        samples = [
            {"time": 10.0, "is_match": True, "confidence_score": 0.8, "caption": "a", "evidence": "x"},
            {"time": 11.0, "is_match": True, "confidence_score": 0.9, "caption": "b", "evidence": "y"},
            {"time": 16.0, "is_match": True, "confidence_score": 0.7, "caption": "c", "evidence": "z"},
        ]

        occurrences = ves.preliminary_occurrences(samples, cfg, 20.0)

        self.assertEqual(len(occurrences), 2)
        self.assertEqual(occurrences[0]["status"], "accepted")
        self.assertEqual(occurrences[1]["status"], "insufficient_positive_samples")

    def test_prepare_output_dir_overwrites_generated_files_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"not a real video")
            input_path = tmp_path / "input.json"
            ves.write_json_atomic(input_path, {
                "video_path": "video.mp4",
                "output_directory": ".",
                "condition": "find this",
                "search": {"strategy": "retrieve_verify"},
            })
            cfg = ves.load_config(input_path, None, None)

            for generated_name in ves.GENERATED_OUTPUT_FILES:
                (tmp_path / generated_name).write_text("old\n", encoding="utf-8")
            evidence_file = tmp_path / "evidence" / "occurrence_999" / "representative.jpg"
            evidence_file.parent.mkdir(parents=True)
            evidence_file.write_text("old", encoding="utf-8")
            keep_file = tmp_path / "keep-me.txt"
            keep_file.write_text("keep", encoding="utf-8")

            ves.prepare_output_dir(cfg)

            self.assertTrue((tmp_path / "config.snapshot.json").exists())
            self.assertFalse((tmp_path / "output.json").exists())
            self.assertFalse((tmp_path / "event-search.json").exists())
            self.assertFalse((tmp_path / "evidence").exists())
            self.assertEqual(keep_file.read_text(encoding="utf-8"), "keep")

    def test_progress_output_is_complete_json_without_condition_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"not a real video")
            input_path = tmp_path / "input.json"
            ves.write_json_atomic(input_path, {
                "video_path": "video.mp4",
                "output_directory": ".",
                "condition": "private condition text",
                "search": {
                    "strategy": "retrieve_verify",
                    "score_threshold": 0.75,
                    "minimum_positive_samples": 1,
                },
            })
            cfg = ves.load_config(input_path, None, None)
            state = ves.new_state()
            state["candidate_windows"] = [{"index": 1, "start_time": 1.0, "end_time": 3.0}]
            state["retrieval_samples"] = []
            state["evaluations"][ves.sample_key(2.0)] = {
                "time": 2.0,
                "phase": "local_scan",
                "candidate_window_index": 1,
                "neutral_caption": "visible event",
                "caption": "visible event",
                "event_phase": "match",
                "confidence_score": 0.9,
                "score_threshold": 0.75,
                "passed_threshold": True,
                "is_match": True,
                "required_visual_evidence": True,
                "evidence": "visible",
                "negative_evidence": "",
                "override_reason": "",
                "parse_ok": True,
                "frame_path": "work/frame.jpg",
            }

            ves.write_progress_output(
                cfg,
                duration=10.0,
                cache_dir=tmp_path / "cache",
                state=state,
                incomplete_reason=None,
                progress={
                    "stage": "local_scan",
                    "last_evaluated_time": 2.0,
                    "last_evaluated_phase": "local_scan",
                    "sample_count": 1,
                    "candidate_window_count": 1,
                    "updated_at_unix": 1.0,
                },
            )

            output = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "running")
            self.assertEqual(output["progress"]["stage"], "local_scan")
            self.assertNotIn("condition", output)
            self.assertNotIn("private condition text", json.dumps(output, ensure_ascii=False))

    def test_public_text_artifacts_do_not_expose_personal_absolute_paths(self):
        result = subprocess.run(
            ["git", "-C", str(WORKSPACE_ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        text_suffixes = {".md", ".py", ".sh", ".json", ".yml", ".yaml", ".gitignore"}
        offenders = []
        user_name = "hiroto" + "saito"
        denied = ("/Users/" + user_name, "//Users/" + user_name)
        for rel_path in result.stdout.splitlines():
            path = WORKSPACE_ROOT / rel_path
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(marker in text for marker in denied):
                offenders.append(rel_path)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
