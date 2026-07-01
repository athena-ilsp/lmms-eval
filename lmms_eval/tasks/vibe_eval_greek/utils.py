import os
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from loguru import logger as eval_logger


# The Greek dataset (ilsp/vibeeval_greek) does NOT embed images -- it only carries a
# remote `media_url`. The images are identical to the English RekaAI/VibeEval set, which
# IS cached locally and DOES embed a PIL `image` column keyed by the same `example_id`.
# Compute nodes are offline (HF_HUB_OFFLINE=1), so we resolve images by an offline lookup
# into the local English VibeEval cache instead of downloading from media_url.
@lru_cache(maxsize=1)
def _vibe_eval_image_index():
    from datasets import load_dataset

    ds = load_dataset("RekaAI/VibeEval", split="test")
    index = {}
    for row in ds:
        img = row.get("image")
        if img is not None:
            index[row["example_id"]] = img
    eval_logger.info(f"vibe_eval_greek: indexed {len(index)} images from RekaAI/VibeEval")
    return index

_JUDGE_PROMPT = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, in regards to the image above.
The response and ground truth are in Greek.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""


@dataclass
class Example:
    example_id: str
    category: str
    prompt: str
    reference: str
    media_url: str
    generation: Optional[str] = None
    score: Optional[int] = None
    evaluator_explanation: Optional[str] = None


def _get_judge_client():
    """Returns an OpenAI-compatible client pointed at a local vLLM server."""
    api_key = os.getenv("VIBE_EVAL_GREEK_JUDGE_API_KEY", "EMPTY")
    base_url = os.getenv("VIBE_EVAL_GREEK_JUDGE_BASE_URL")
    if not base_url:
        eval_logger.error("VIBE_EVAL_GREEK_JUDGE_BASE_URL is not set. Point it at your vLLM OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _get_judge_model_name() -> str:
    return os.getenv("VIBE_EVAL_GREEK_JUDGE_MODEL", "default")


def vibe_el_doc_to_visual(doc):
    # Greek docs have no `image`; look it up from the local English VibeEval cache by
    # example_id (same images, offline-safe).
    img = doc.get("image")
    if img is None:
        img = _vibe_eval_image_index().get(doc["example_id"])
    if img is None:
        raise KeyError(
            f"No image for example_id={doc.get('example_id')!r} "
            f"(media_url={doc.get('media_url')!r}) in local RekaAI/VibeEval cache"
        )
    return [img.convert("RGB")]


def vibe_el_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["prompt_el"].strip()
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def _judge(example: Example) -> Example:
    client = _get_judge_client()
    if client is None:
        example.score = 0
        example.evaluator_explanation = "Judge client unavailable"
        return example

    judge_prompt = _JUDGE_PROMPT.format(
        prompt=example.prompt,
        reference=example.reference,
        generation=example.generation,
    )

    try:
        response = client.chat.completions.create(
            model=_get_judge_model_name(),
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        judge_text = response.choices[0].message.content
    except Exception as e:
        eval_logger.error(f"Judge call failed: {e}")
        example.score = 0
        example.evaluator_explanation = str(e)
        return example

    re_match = re.search(r"Rating:\s*([1-5])", judge_text)
    if re_match is None:
        example.score = 0
        example.evaluator_explanation = judge_text
        return example

    example.score = int(re_match.group(1))
    example.evaluator_explanation = judge_text
    return example


def vibe_el_process_results(doc, results):
    example = Example(
        example_id=doc["example_id"],
        category=doc["category"],
        prompt=doc["prompt_el"],
        reference=doc["reference_el"],
        media_url=doc["media_url"],
        generation=results[0],
    )

    example = _judge(example)

    data_dict = {
        "score": example.score,
        "evaluator_explanation": example.evaluator_explanation,
        "prompt": example.prompt,
        "generation": example.generation,
        "media_url": example.media_url,
        "category": example.category,
    }

    return {
        "hard": deepcopy(data_dict),
        "normal": deepcopy(data_dict),
        "all": deepcopy(data_dict),
    }


def _mean(scores: List[int]) -> float:
    """Scale from 1-5 to 0-100 and compute means."""
    # A category can be empty under --limit (smoke tests), which would div-by-zero.
    if not scores:
        return 0.0
    return sum(25 * (score - 1) for score in scores) / len(scores)


def vibe_el_aggregation_results(results, category):
    scores = [res["score"] for res in results if category in res["category"] or category == "all"]
    return _mean(scores)


def vibe_el_aggregation_results_normal(results):
    return vibe_el_aggregation_results(results, "normal")


def vibe_el_aggregation_results_hard(results):
    return vibe_el_aggregation_results(results, "hard")


def vibe_el_aggregation_results_all(results):
    return vibe_el_aggregation_results(results, "all")