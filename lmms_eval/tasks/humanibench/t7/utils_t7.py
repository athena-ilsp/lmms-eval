from lmms_eval.tasks.humanibench.common import (
    aggregate_mean,
    apply_prompts,
    image_messages,
    image_visual,
    judge_six,
    log_group_means,
    mean_score,
)
from lmms_eval.tasks.humanibench.prompts import T7_INFERENCE


def doc_to_visual(doc):
    return image_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return apply_prompts(T7_INFERENCE.format(question=doc["Question"]), lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return image_messages(doc_to_visual(doc), doc_to_text(doc, lmms_eval_specific_kwargs))


def process_results(doc, results):
    prediction = results[0] if results else ""
    labels = {"attribute": doc.get("Attribute"), "attack_type": doc.get("attack_type"), "id": doc.get("ID")}
    scored = judge_six(doc, prediction, **labels)
    scored["retention"] = dict(scored["accuracy"])
    return scored


def aggregate_accuracy(results):
    log_group_means(results, "accuracy", key="attack_type")
    return aggregate_mean(results, "accuracy")


def aggregate_bias(results):
    return aggregate_mean(results, "bias")


def aggregate_hallucination(results):
    return aggregate_mean(results, "hallucination")


def aggregate_faithfulness(results):
    return aggregate_mean(results, "faithfulness")


def aggregate_contextual_relevance(results):
    return aggregate_mean(results, "contextual_relevance")


def aggregate_coherence(results):
    return aggregate_mean(results, "coherence")


def aggregate_retention(results):
    log_group_means(results, "retention", key="attack_type")
    clean = [row for row in results if isinstance(row, dict) and row.get("attack_type") == "clean"]
    attacked = [row for row in results if isinstance(row, dict) and row.get("attack_type") != "clean"]
    clean_mean = mean_score(clean)
    if clean_mean <= 0:
        return 0.0
    return 100.0 * mean_score(attacked) / clean_mean
