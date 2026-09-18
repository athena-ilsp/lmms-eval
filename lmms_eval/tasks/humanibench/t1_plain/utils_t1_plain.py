from lmms_eval.tasks.humanibench.common import aggregate_mean, apply_prompts, image_messages, image_visual, judge_six


def process_docs(dataset):
    return dataset.filter(lambda x: x["version_type"] == "plain_version")


def doc_to_visual(doc):
    return image_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return apply_prompts(doc["Question"], lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return image_messages(doc_to_visual(doc), doc_to_text(doc, lmms_eval_specific_kwargs))


def process_results(doc, results):
    prediction = results[0] if results else ""
    return judge_six(doc, prediction, attribute=doc.get("Attribute"), id=doc.get("ID"))


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
