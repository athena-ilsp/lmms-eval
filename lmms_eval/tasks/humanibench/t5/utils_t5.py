import re

from lmms_eval.tasks.humanibench.common import apply_prompts, image_messages, image_visual, mean_score, metric_item
from lmms_eval.tasks.humanibench.prompts import T5_INFERENCE


def doc_to_visual(doc):
    return image_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return apply_prompts(T5_INFERENCE.format(question=doc["question"]), lmms_eval_specific_kwargs)


def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return image_messages(doc_to_visual(doc), doc_to_text(doc, lmms_eval_specific_kwargs))


def parse_box(text):
    if text is None:
        return None
    if hasattr(text, "tolist"):
        text = text.tolist()
    if isinstance(text, (list, tuple)) and len(text) >= 4:
        try:
            return [float(text[0]), float(text[1]), float(text[2]), float(text[3])]
        except (TypeError, ValueError):
            return None
    nums = re.findall(r"-?\d+(?:\.\d+(?:[eE][+-]?\d+)?)?", str(text))
    if len(nums) < 4:
        return None
    return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]


def normalize_box(box, width, height):
    if not box or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return [xmin, ymin, xmax, ymax]


def box_iou(box_a, box_b):
    if not box_a or not box_b:
        return 0.0
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def image_size(doc):
    image = doc.get("image")
    if image is not None and hasattr(image, "size"):
        return image.size
    return (0, 0)


def process_results(doc, results):
    prediction = results[0] if results else ""
    width, height = image_size(doc)
    pred = normalize_box(parse_box(prediction), width, height)
    gold = normalize_box(parse_box(doc.get("bbox")), width, height)
    missing = pred is None
    iou = 0.0 if missing else box_iou(pred, gold)
    labels = {"id": doc.get("ID")}
    return {
        "mean_iou": metric_item(100.0 * iou, **labels),
        "map50": metric_item(100.0 if iou >= 0.5 else 0.0, **labels),
        "map75": metric_item(100.0 if iou >= 0.75 else 0.0, **labels),
        "missing": metric_item(100.0 if missing else 0.0, **labels),
    }


def aggregate_mean_iou(results):
    return mean_score(results)


def aggregate_map50(results):
    return mean_score(results)


def aggregate_map75(results):
    return mean_score(results)


def aggregate_missing(results):
    return mean_score(results)
