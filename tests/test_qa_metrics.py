import json

from projected_token.metrics.qa_text import contains_match, exact_match, in_accuracy_match, token_f1
from projected_token.qa.eval import evaluate_predictions_jsonl, run_qa_eval


def test_qa_metrics_match_oscar_semantics():
    assert exact_match("Hello, World!", "hello world")
    assert token_f1("partial match text", "match") == 0.5
    assert contains_match("Everest", "Mount Everest")
    assert not in_accuracy_match("Everest", "Mount Everest")


def test_run_qa_eval_aggregates_max_over_refs():
    preds = {"1": "exact", "2": "partial match text"}
    gold = {"1": ["exact"], "2": ["match"]}

    summary, errors = run_qa_eval(preds, gold)

    assert summary["n_scored"] == 2
    assert summary["mean_em"] == 0.5
    assert summary["mean_answer_in_prediction"] == 1.0
    assert summary["mean_in_accuracy"] == 1.0
    assert len(errors) == 1


def test_evaluate_predictions_jsonl_uses_inline_answers(tmp_path):
    pred_path = tmp_path / "pred.jsonl"
    out_path = tmp_path / "metrics.json"
    errors_path = tmp_path / "errors.jsonl"
    pred_path.write_text(
        '{"qid":"a","prediction":"hello world","answers":["Hello World!"],"method":"rag","model_id":"m"}\n',
        encoding="utf-8",
    )

    payload = evaluate_predictions_jsonl(
        pred_path=pred_path,
        output_path=out_path,
        write_errors_path=errors_path,
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mean_em"] == 1.0
    assert written["mean_in_accuracy"] == 1.0
    assert written["generation_meta"]["method"] == "rag"
    assert errors_path.read_text(encoding="utf-8") == ""
