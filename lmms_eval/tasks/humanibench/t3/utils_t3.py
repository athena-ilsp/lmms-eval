from lmms_eval.tasks.humanibench.common import (
    aggregate_mean,
    apply_prompts,
    image_messages,
    image_visual,
    judge_six,
    mcq_match,
    metric_item,
)
from lmms_eval.tasks.humanibench.prompts import T3_INFERENCE


def doc_to_visual(doc):
    return image_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["Question"] + "\n" + str(doc.get("Options") or "")
    return apply_prompts(T3_INFERENCE.format(QUESTION=question), lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return image_messages(doc_to_visual(doc), doc_to_text(doc, lmms_eval_specific_kwargs))


def process_results(doc, results):
    prediction = results[0] if results else ""
    labels = {"attribute": doc.get("Attribute"), "id": doc.get("ID")}
    question = f"{doc.get('Question') or ''}\n{doc.get('Options') or ''}"
    scored = {"accuracy_statistical": metric_item(100.0 if mcq_match(prediction, doc.get("Answer")) else 0.0, **labels)}
    scored.update(judge_six(doc, prediction, question=question, gold=doc.get("Reasoning"), **labels))
    return scored


def aggregate_accuracy_statistical(results):
    return aggregate_mean(results, "accuracy_statistical")


def aggregate_accuracy(results):
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
