from lmms_eval.tasks.humanibench.common import (
    aggregate_mean,
    apply_prompts,
    drop_null_fields,
    image_messages,
    image_visual,
    judge_six,
    log_group_means,
)
from lmms_eval.tasks.humanibench.prompts import T4_OPEN_INFERENCE


def process_docs(dataset):
    return drop_null_fields(dataset, "Question", "Answer")


def doc_to_visual(doc):
    return image_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    text = T4_OPEN_INFERENCE.format(language=doc.get("Language", ""), question=doc.get("Question") or "")
    return apply_prompts(text, lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return image_messages(doc_to_visual(doc), doc_to_text(doc, lmms_eval_specific_kwargs))


def process_results(doc, results):
    prediction = results[0] if results else ""
    labels = {"attribute": doc.get("Attribute"), "language": doc.get("Language"), "id": doc.get("ID")}
    return judge_six(doc, prediction, **labels)


def aggregate_accuracy(results):
    log_group_means(results, "accuracy", key="language")
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
