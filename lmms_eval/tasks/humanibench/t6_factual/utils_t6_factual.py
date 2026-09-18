from lmms_eval.tasks.humanibench.common import (
    T6_FACTUAL_QUALITY_RUBRICS,
    aggregate_mean,
    apply_prompts,
    image_messages,
    image_visual,
    judge_six,
)
from lmms_eval.tasks.humanibench.prompts import T6_FACTUAL


def doc_to_visual(doc):
    return image_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return apply_prompts(T6_FACTUAL, lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return image_messages(doc_to_visual(doc), doc_to_text(doc, lmms_eval_specific_kwargs))


def process_results(doc, results):
    prediction = results[0] if results else ""
    labels = {"social_attribute": doc.get("social_attribute"), "id": doc.get("ID")}
    return judge_six(
        doc,
        prediction,
        question=T6_FACTUAL,
        gold=doc.get("simple_prompt"),
        rubrics=T6_FACTUAL_QUALITY_RUBRICS,
        **labels,
    )


def aggregate_accuracy(results):
    return aggregate_mean(results, "accuracy", group_key="social_attribute")


def aggregate_bias(results):
    return aggregate_mean(results, "bias", group_key="social_attribute")


def aggregate_hallucination(results):
    return aggregate_mean(results, "hallucination", group_key="social_attribute")


def aggregate_faithfulness(results):
    return aggregate_mean(results, "faithfulness", group_key="social_attribute")
