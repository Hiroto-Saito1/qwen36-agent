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
        "condition": "find this",
        "language": "Japanese",
        "endpoint": "http://127.0.0.1:8081/v1",
        "model": "test-model",
        "score_threshold": 0.75,
        "candidate_padding": 1.0,
        "minimum_candidate_windows": 2,
        "max_candidate_windows": 2,
        "local_scan_interval": 1.0,
        "minimum_event_duration": 5.0,
        "minimum_positive_samples": 2,
        "verification_image_max_edge_pixels": 960,
        "max_evaluations": None,
        "request_retries": 2,
        "request_timeout": 1,
        "temperature": 0.1,
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
            self.assertEqual(cfg.verification_image_max_edge_pixels, 640)
            self.assertEqual(cfg.retrieval_queries, ())
            self.assertEqual(cfg.query_plan, {})

            bad_path = tmp_path / "bad.json"
            ves.write_json_atomic(bad_path, {
                "video_path": "video.mp4",
                "output_directory": ".",
                "condition": "find this",
                "caption_interval_seconds": 30,
            })
            with self.assertRaisesRegex(ves.ConfigError, "old candidate_binary schema"):
                ves.load_config(bad_path, None, None)

            old_semantic_path = tmp_path / "old-semantic.json"
            ves.write_json_atomic(old_semantic_path, {
                "video_path": "video.mp4",
                "output_directory": ".",
                "condition": "find this",
                "search": {
                    "query_texts": ["extra search text"],
                    "required_visual_checks": ["thing must be visible"],
                },
            })
            with self.assertRaisesRegex(ves.ConfigError, "Put all event semantics in 'condition'"):
                ves.load_config(old_semantic_path, None, None)

    def test_verification_resize_dimensions(self):
        self.assertEqual(ves.scaled_dimensions(1920, 1080, 960), (960, 540))
        self.assertEqual(ves.scaled_dimensions(1080, 1920, 960), (540, 960))
        self.assertEqual(ves.scaled_dimensions(800, 600, 960), (800, 600))
        self.assertEqual(ves.scaled_dimensions(1920, 1080, None), (1920, 1080))

    def test_condition_query_plan_uses_model_result_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(b"not a real video")
            input_path = tmp_path / "input.json"
            ves.write_json_atomic(input_path, {
                "video_path": "video.mp4",
                "output_directory": ".",
                "condition": "蝶が木の幹の近くに見えている",
                "search": {"strategy": "retrieve_verify"},
            })
            cfg = ves.load_config(input_path, None, None)
            cache_path = tmp_path / "query-cache.json"
            calls = []

            def fake_post_json(url, payload, timeout):
                calls.append((url, payload, timeout))
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({
                                    "queries": [
                                        "butterfly near a tree trunk",
                                        "butterfly near a tree trunk",
                                        "small butterfly near a tree",
                                        "extra ignored query",
                                    ]
                                })
                            }
                        }
                    ]
                }

            original_post_json = ves.post_json
            original_cache_path = ves.condition_query_cache_path
            ves.post_json = fake_post_json
            ves.condition_query_cache_path = lambda cfg: cache_path
            try:
                plan = ves.compile_condition_query_plan(cfg)
                cached_plan = ves.compile_condition_query_plan(cfg)
                attached = ves.attach_condition_query_plan(cfg, plan)
            finally:
                ves.post_json = original_post_json
                ves.condition_query_cache_path = original_cache_path

            self.assertEqual(calls[0][1]["messages"][0]["content"].count("蝶が木の幹"), 1)
            self.assertEqual(plan["queries"], ["butterfly near a tree trunk", "small butterfly near a tree", "extra ignored query"])
            self.assertFalse(plan["cache_hit"])
            self.assertTrue(cached_plan["cache_hit"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(attached.retrieval_queries, tuple(plan["queries"]))
            self.assertEqual(attached.canonical["search"]["retrieval_query_plan"]["queries"], plan["queries"])

    def test_condition_query_plan_falls_back_to_condition_after_failures(self):
        cfg = fake_cfg(condition="private visual condition", request_retries=1)

        def fake_post_json(url, payload, timeout):
            raise OSError("server unavailable")

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_post_json = ves.post_json
            original_sleep = ves.time.sleep
            original_cache_path = ves.condition_query_cache_path
            ves.post_json = fake_post_json
            ves.time.sleep = lambda seconds: None
            ves.condition_query_cache_path = lambda cfg: pathlib.Path(tmp_dir) / "query-cache.json"
            try:
                plan = ves.compile_condition_query_plan(cfg)
            finally:
                ves.post_json = original_post_json
                ves.time.sleep = original_sleep
                ves.condition_query_cache_path = original_cache_path

        self.assertEqual(plan["queries"], ["private visual condition"])
        self.assertEqual(plan["source"], "fallback")
        self.assertEqual(plan["setup_request_count"], 2)
        self.assertIn("using original condition", " ".join(plan["warnings"]))

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
            "evidence": "visible event is present",
        }, cfg)
        self.assertTrue(accepted["is_match"])
        self.assertTrue(accepted["has_affirmative_evidence"])

        below_threshold = ves.normalize_verification_result({
            "neutral_caption": "visible event",
            "event_phase": "match",
            "confidence_score": 0.7,
            "evidence": "visible event is present",
        }, cfg)
        self.assertFalse(below_threshold["is_match"])

        missing_evidence = ves.normalize_verification_result({
            "neutral_caption": "visible event",
            "event_phase": "match",
            "confidence_score": 0.9,
            "evidence": "",
        }, cfg)
        self.assertFalse(missing_evidence["is_match"])

    def test_near_miss_is_not_lexically_overridden(self):
        cfg = fake_cfg(score_threshold=0.75)

        not_corrected = ves.normalize_verification_result({
            "neutral_caption": "A butterfly is near a tree trunk.",
            "event_phase": "near_miss",
            "confidence_score": 0.8,
            "evidence": "The butterfly is near the trunk.",
            "negative_evidence": "",
        }, cfg)
        self.assertFalse(not_corrected["is_match"])
        self.assertEqual(not_corrected["event_phase"], "near_miss")

    def test_normalization_redacts_exact_condition_and_generated_queries(self):
        cfg = fake_cfg(
            condition="private condition text",
            retrieval_queries=("generated retrieval secret",),
            score_threshold=0.75,
        )

        result = ves.normalize_verification_result({
            "neutral_caption": "private condition text is visible",
            "event_phase": "match",
            "confidence_score": 0.9,
            "evidence": "generated retrieval secret is visible",
            "negative_evidence": "private condition text is not missing",
        }, cfg)
        result_text = json.dumps(result, ensure_ascii=False)

        self.assertNotIn("private condition text", result_text)
        self.assertNotIn("generated retrieval secret", result_text)
        self.assertIn("[condition]", result_text)
        self.assertIn("[retrieval_query]", result_text)

    def test_occurrence_grouping_and_minimum_positive_samples(self):
        cfg = fake_cfg(local_scan_interval=1.0, minimum_event_duration=5.0, minimum_positive_samples=2)
        samples = [
            {"time": 10.0, "is_match": True, "verification_status": "confirmed", "confidence_score": 0.8, "caption": "a", "evidence": "x"},
            {"time": 11.0, "is_match": True, "verification_status": "confirmed", "confidence_score": 0.9, "caption": "b", "evidence": "y"},
            {"time": 16.0, "is_match": True, "verification_status": "confirmed", "confidence_score": 0.7, "caption": "c", "evidence": "z"},
        ]

        occurrences = ves.preliminary_occurrences(samples, cfg, 20.0)

        self.assertEqual(len(occurrences), 2)
        self.assertEqual(occurrences[0]["status"], "accepted")
        self.assertEqual(occurrences[1]["status"], "insufficient_positive_samples")

    def test_primary_only_match_is_not_accepted_as_occurrence(self):
        cfg = fake_cfg(local_scan_interval=1.0, minimum_positive_samples=1)
        samples = [
            {
                "time": 10.0,
                "is_match": False,
                "candidate_is_match": True,
                "verification_status": "primary_only",
                "confidence_score": 0.9,
                "caption": "candidate",
                "evidence": "candidate evidence",
            }
        ]

        occurrences = ves.preliminary_occurrences(samples, cfg, 20.0)

        self.assertEqual(occurrences, [])

    def test_two_stage_local_scan_requests_confirmation_for_related_phases(self):
        cfg = fake_cfg(max_evaluations=4)
        state = ves.new_state()
        calls = []

        def fake_evaluate_sample(cfg, cache_dir, duration, requested_time, verification_mode):
            calls.append((requested_time, verification_mode))
            is_triple = verification_mode == "triple"
            return {
                "time": requested_time,
                "neutral_caption": f"{verification_mode} caption",
                "caption": f"{verification_mode} caption",
                "event_phase": "match",
                "confidence_score": 0.9,
                "score_threshold": cfg.score_threshold,
                "passed_threshold": True,
                "is_match": is_triple,
                "has_affirmative_evidence": True,
                "evidence": "visible",
                "negative_evidence": "",
                "parse_ok": True,
                "verification_mode": verification_mode,
                "sent_image_count": 3 if is_triple else 1,
                "verification_images": [],
            }

        original = ves.evaluate_sample
        ves.evaluate_sample = fake_evaluate_sample
        try:
            ok = ves.ensure_evaluation(
                cfg,
                duration=20.0,
                cache_dir=pathlib.Path("/tmp/cache"),
                state=state,
                time_seconds=10.0,
                phase="local_scan",
                candidate_window_index=1,
                progress_writer=lambda progress: None,
            )
        finally:
            ves.evaluate_sample = original

        self.assertTrue(ok)
        self.assertEqual(calls, [(10.0, "single"), (10.0, "triple")])
        sample = state["evaluations"][ves.sample_key(10.0)]
        self.assertEqual(sample["verification_status"], "confirmed")
        self.assertTrue(sample["is_match"])
        self.assertEqual(state["vl_request_count"], 2)
        self.assertEqual(state["sent_image_count"], 4)

    def test_two_stage_local_scan_skips_confirmation_for_unrelated(self):
        cfg = fake_cfg(max_evaluations=4)
        state = ves.new_state()
        calls = []

        def fake_evaluate_sample(cfg, cache_dir, duration, requested_time, verification_mode):
            calls.append((requested_time, verification_mode))
            return {
                "time": requested_time,
                "neutral_caption": "nothing relevant",
                "caption": "nothing relevant",
                "event_phase": "unrelated",
                "confidence_score": 0.95,
                "score_threshold": cfg.score_threshold,
                "passed_threshold": True,
                "is_match": False,
                "has_affirmative_evidence": False,
                "evidence": "",
                "negative_evidence": "not relevant",
                "parse_ok": True,
                "verification_mode": verification_mode,
                "sent_image_count": 1,
                "verification_images": [],
            }

        original = ves.evaluate_sample
        ves.evaluate_sample = fake_evaluate_sample
        try:
            ok = ves.ensure_evaluation(
                cfg,
                duration=20.0,
                cache_dir=pathlib.Path("/tmp/cache"),
                state=state,
                time_seconds=10.0,
                phase="local_scan",
                candidate_window_index=1,
                progress_writer=lambda progress: None,
            )
        finally:
            ves.evaluate_sample = original

        self.assertTrue(ok)
        self.assertEqual(calls, [(10.0, "single")])
        sample = state["evaluations"][ves.sample_key(10.0)]
        self.assertEqual(sample["verification_status"], "primary_only")
        self.assertFalse(sample["is_match"])
        self.assertEqual(state["vl_request_count"], 1)
        self.assertEqual(state["sent_image_count"], 1)

    def test_max_evaluations_counts_primary_and_confirmation_requests(self):
        cfg = fake_cfg(max_evaluations=1)
        state = ves.new_state()

        def fake_evaluate_sample(cfg, cache_dir, duration, requested_time, verification_mode):
            return {
                "time": requested_time,
                "neutral_caption": "candidate",
                "caption": "candidate",
                "event_phase": "near_miss",
                "confidence_score": 0.9,
                "score_threshold": cfg.score_threshold,
                "passed_threshold": True,
                "is_match": False,
                "has_affirmative_evidence": True,
                "evidence": "candidate",
                "negative_evidence": "needs confirmation",
                "parse_ok": True,
                "verification_mode": verification_mode,
                "sent_image_count": 1,
                "verification_images": [],
            }

        original = ves.evaluate_sample
        ves.evaluate_sample = fake_evaluate_sample
        try:
            ok = ves.ensure_evaluation(
                cfg,
                duration=20.0,
                cache_dir=pathlib.Path("/tmp/cache"),
                state=state,
                time_seconds=10.0,
                phase="local_scan",
                candidate_window_index=1,
                progress_writer=lambda progress: None,
            )
        finally:
            ves.evaluate_sample = original

        self.assertFalse(ok)
        sample = state["evaluations"][ves.sample_key(10.0)]
        self.assertEqual(sample["verification_status"], "confirmation_blocked_by_max_evaluations")
        self.assertFalse(sample["is_match"])
        self.assertEqual(state["vl_request_count"], 1)

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

    def test_verification_prompt_uses_condition_only_schema(self):
        cfg = fake_cfg(condition="蝶が木の幹の近くに見えている。ただし蝶の色は問わない。")

        prompt = ves.build_verification_prompt(cfg, 10.0, 9.0, 11.0, "triple")

        self.assertIn("蝶が木の幹の近く", prompt)
        self.assertIn("condition text is the only source of truth", prompt)
        self.assertNotIn("Required visual checks", prompt)
        self.assertNotIn("Not required", prompt)
        self.assertNotIn("required_visual_evidence", prompt)
        self.assertNotIn('"is_match"', prompt)

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
            cfg = ves.attach_condition_query_plan(cfg, {
                "compiler_version": ves.QUERY_COMPILER_VERSION,
                "condition_sha256": "abc123",
                "queries": ["generated retrieval secret"],
                "source": "model",
                "cache_hit": False,
                "setup_request_count": 1,
                "fallback_reason": "",
                "warnings": [],
            })
            ves.prepare_output_dir(cfg)
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
                "verification_status": "confirmed",
                "has_affirmative_evidence": True,
                "evidence": "visible",
                "negative_evidence": "",
                "parse_ok": True,
                "frame_path": "work/frame.jpg",
                "sent_image_count": 3,
            }
            state["vl_request_count"] = 1
            state["sent_image_count"] = 3

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
            self.assertEqual(output["verification"]["vl_request_count"], 1)
            self.assertEqual(output["verification"]["total_sent_image_count"], 3)
            self.assertNotIn("condition", output)
            output_text = json.dumps(output, ensure_ascii=False)
            self.assertNotIn("private condition text", output_text)
            self.assertNotIn("generated retrieval secret", output_text)

            snapshot = json.loads((tmp_path / "config.snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["search"]["retrieval_query_plan"]["queries"], ["generated retrieval secret"])
            trace = ves.build_search_trace(cfg, 10.0, tmp_path / "cache", state)
            self.assertEqual(trace["retrieval_query_plan"]["queries"], ["generated retrieval secret"])

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
