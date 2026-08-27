"""Tests for the debug request log (Fase 2 cache-drop diagnostics)."""

import json
import os
import pytest
from unittest.mock import patch

import debug_log as dl
from config import config


@pytest.fixture(autouse=True)
def _debug_off_after():
    """Ensure the flag is OFF after each test (global raw dict is shared)."""
    yield
    config.raw.get("server", {}).pop("debug_log", None)


@pytest.fixture()
def tmp_log_file(tmp_path, monkeypatch):
    """Redirect the debug log file to a temp path with a fresh logger."""
    target = tmp_path / "requests.jsonl"
    monkeypatch.setattr(dl, "_log_path", target)
    saved = dl._requests_logger
    dl._requests_logger = None  # force lazy re-creation on the new path
    yield target
    # close handlers to release file locks on Windows
    if dl._requests_logger:
        for h in list(dl._requests_logger.handlers):
            h.close()
            dl._requests_logger.removeHandler(h)
    dl._requests_logger = saved


def _set_flag(on: bool):
    config.raw.setdefault("server", {})["debug_log"] = on


class TestDynamicFlag:
    def test_dynamic_on_off(self):
        _set_flag(True)
        assert dl.is_enabled() is True
        _set_flag(False)
        assert dl.is_enabled() is False

    def test_env_override(self, monkeypatch):
        config.raw.get("server", {}).pop("debug_log", None)
        monkeypatch.setenv("DEBUG_REQUEST_LOG", "1")
        assert dl.is_enabled() is True
        monkeypatch.setenv("DEBUG_REQUEST_LOG", "0")
        assert dl.is_enabled() is False


class TestHashing:
    def test_key_order_irrelevant(self):
        msgs_a = [{"role": "system", "content": "abc"},
                  {"role": "user", "content": "hello"}]
        msgs_b = [{"content": "abc", "role": "system"},
                  {"content": "hello", "role": "user"}]
        assert dl.canonical_messages_hash(msgs_a) == dl.canonical_messages_hash(msgs_b)

    def test_message_order_significant(self):
        msgs = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
        assert (dl.canonical_messages_hash(msgs)
                != dl.canonical_messages_hash(list(reversed(msgs))))

    def test_content_change_changes_hash(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert (dl.canonical_messages_hash(msgs)
                != dl.canonical_messages_hash([{"role": "user", "content": "hellp"}]))

    def test_fingerprint_shape(self):
        fp = dl.messages_fingerprint([{"role": "system", "content": "12345"}])
        assert fp == [{"i": 0, "role": "system", "chars": 5,
                       "h": dl.messages_fingerprint([{"role": "system", "content": "12345"}])[0]["h"]}]


class TestWriteGate:
    def test_no_write_when_off(self, tmp_log_file):
        _set_flag(False)
        dl.hook_request_in("rid-1", {"messages": []}, {})
        assert not tmp_log_file.exists()

    def test_writes_when_on(self, tmp_log_file):
        _set_flag(True)
        dl.hook_request_in("rid-2", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                           {"HTTP-Referer": "https://app"})
        assert tmp_log_file.exists()
        rec = json.loads(tmp_log_file.read_text(encoding="utf-8").strip())
        assert rec["type"] == "req_in" and rec["req_id"] == "rid-2"


class TestCorrelationChain:
    def test_in_out_res_chain_and_privacy(self, tmp_log_file):
        _set_flag(True)
        rid = "chain-xyz"
        body = {"model": "m",
                "messages": [{"role": "user", "content": "ciao"}],
                "session_id": "sess"}
        dl.hook_request_in(rid, body, {"Authorization": "Bearer TOPSECRET",
                                       "HTTP-Referer": "https://hermes.local"})
        dl.hook_request_out(rid, {**body, "provider": {"only": ["deepinfra/fp8"]}},
                            "deepinfra/fp8", 0)
        dl.hook_response(rid, "deepinfra/fp8", 200,
                         {"prompt_tokens": 100,
                          "prompt_tokens_details": {"cached_tokens": 96}})
        recs = [json.loads(l) for l in
                tmp_log_file.read_text(encoding="utf-8").strip().splitlines()]
        mine = [r for r in recs if r["req_id"] == rid]
        assert [r["type"] for r in mine] == ["req_in", "req_out", "res"]

        req_in, req_out, res = mine
        # privacy: no secret anywhere
        dumped = json.dumps(recs)
        assert "TOPSECRET" not in dumped and "Bearer" not in dumped
        assert req_in["client_headers"]["HTTP-Referer"] == "https://hermes.local"
        # contract: hash(in) == hash(out) with verbatim passthrough
        assert req_in["messages_hash"] == req_out["messages_hash"]
        assert req_out["provider_pin"] == "deepinfra/fp8"
        assert res["cached_tokens"] == 96 and res["prompt_tokens"] == 100


class TestBodyTrim:
    def test_trim_marker(self):
        out = dl._trim("x" * 3000, 100)
        assert len(out) < 3000 and "[TRUNCATED at 100 chars" in out

    def test_trim_zero_means_unlimited(self):
        assert dl._trim("y" * 500, 0) == "y" * 500
