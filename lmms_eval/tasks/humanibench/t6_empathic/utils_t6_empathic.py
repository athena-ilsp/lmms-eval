from lmms_eval.tasks.humanibench.common import aggregate_mean, apply_prompts, image_messages, image_visual, judge_empathy
from lmms_eval.tasks.humanibench.prompts import T6_EMPATHIC


def doc_to_visual(doc):
    return image_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return apply_prompts(T6_EMPATHIC, lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return image_messages(doc_to_visual(doc), doc_to_text(doc, lmms_eval_specific_kwargs))


def process_results(doc, results):
    prediction = results[0] if results else ""
    labels = {"social_attribute": doc.get("social_attribute"), "id": doc.get("ID")}
    return judge_empathy(prediction, image=doc.get("image"), **labels)


def aggregate_empathy(results):
    return aggregate_mean(results, "empathy", group_key="social_attribute")


def aggregate_anxiety(results):
    return aggregate_mean(results, "anxiety", group_key="social_attribute")


def aggregate_sadness(results):
    return aggregate_mean(results, "sadness", group_key="social_attribute")


def aggregate_joy(results):
    return aggregate_mean(results, "joy", group_key="social_attribute")
