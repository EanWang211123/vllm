# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

from vllm.v1.spec_decode.verify_adaptive_controller import (
    AdaptiveVerifyQSlotStatsCollector,
    build_q_slot_stats_report,
    resolve_q_slot_stats_output_path,
)


def test_resolve_q_slot_stats_output_path_file():
    assert (
        resolve_q_slot_stats_output_path("/tmp/stats.json")
        == "/tmp/stats.json"
    )


def test_resolve_q_slot_stats_output_path_directory(tmp_path: Path):
    out = resolve_q_slot_stats_output_path(str(tmp_path))
    assert out == str(tmp_path / "adaptive_verify_q_slot_stats.json")


def test_build_q_slot_stats_report_includes_zero_hit_slots():
    hit_counts = {32: {2: 2, 4: 3}}
    ql_levels = [2, 4, 6]
    report = build_q_slot_stats_report(hit_counts, ql_levels)

    assert report["version"] == 1
    bs = report["batch_sizes"]["32"]
    assert bs["total_decisions"] == 5
    assert bs["q_slots"] == [
        {"query_len_per_req": 2, "count": 2, "probability": 0.4},
        {"query_len_per_req": 4, "count": 3, "probability": 0.6},
        {"query_len_per_req": 6, "count": 0, "probability": 0.0},
    ]


def test_collector_flush_writes_json(tmp_path: Path):
    out_dir = tmp_path / "stats"
    collector = AdaptiveVerifyQSlotStatsCollector(str(out_dir))
    collector.record(32, [1, 3])
    collector.record(32, [3])
    collector.flush([2, 4])

    out_file = out_dir / "adaptive_verify_q_slot_stats.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["batch_sizes"]["32"]["total_decisions"] == 3
    slots = {s["query_len_per_req"]: s["count"] for s in data["batch_sizes"]["32"]["q_slots"]}
    assert slots[2] == 1
    assert slots[4] == 2
