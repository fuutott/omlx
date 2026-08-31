import sys
from pathlib import Path

from tools.qwen4_flash_next_eval import (
    api_url,
    evaluate_checks,
    load_corpus,
    parse_args,
)


def test_api_url_accepts_root_or_v1_base():
    assert api_url("http://localhost:1234", "models") == "http://localhost:1234/v1/models"
    assert api_url("http://localhost:1234/v1/", "/models") == "http://localhost:1234/v1/models"


def test_fixed_corpus_is_unique_and_has_checks():
    corpus = load_corpus(
        Path(__file__).parents[1] / "tools" / "qwen4_flash_next_eval_corpus.jsonl"
    )
    assert len(corpus) == 12
    assert len({item["id"] for item in corpus}) == len(corpus)
    assert all(item.get("checks") for item in corpus)


def test_check_evaluator_handles_json_and_forbidden_text():
    checks = [
        {"type": "json_equals", "value": {"answer": 42}},
        {"type": "not_regex", "pattern": "explanation"},
    ]
    results = evaluate_checks('{"answer": 42}', checks)
    assert all(result["passed"] for result in results)


def test_polish_blue_synonyms_are_accepted():
    check = [{"type": "regex", "pattern": r"^(?:b[łl]ękitny|blekitny)$"}]
    assert evaluate_checks("Błękitny", check)[0]["passed"]
    assert evaluate_checks("blekitny", check)[0]["passed"]


def test_disable_thinking_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen4_flash_next_eval.py",
            "run",
            "--endpoint",
            "test",
            "--base-url",
            "http://localhost:1234",
            "--model",
            "test-model",
            "--output",
            "result.jsonl",
            "--disable-thinking",
        ],
    )
    assert parse_args().enable_thinking is False
