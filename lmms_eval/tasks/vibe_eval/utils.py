import base64
import io
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from loguru import logger as eval_logger

REKA_API_KEY = os.getenv("REKA_API_KEY", "YOUR_API_KEY")


# ---- local judge (image-aware) ----
# The original vibe_eval grades via Reka's hosted API, which isn't available here. We grade
# instead with a local vLLM OpenAI-compatible server (Gemma-4-31B, a multimodal model), so
# the judge still SEES the image like Reka-Core did. Endpoint comes from env vars; falls
# back to the generic OPENAI_* vars the sbatch already exports for the judge.
def _get_judge_client():
    api_key = os.getenv("VIBE_EVAL_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = os.getenv("VIBE_EVAL_JUDGE_BASE_URL") or os.getenv("OPENAI_API_URL")
    if not base_url:
        eval_logger.error("VIBE_EVAL_JUDGE_BASE_URL / OPENAI_API_URL not set; point it at a vLLM OpenAI endpoint, e.g. http://host:port/v1")
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _get_judge_model_name() -> str:
    return os.getenv("VIBE_EVAL_JUDGE_MODEL") or os.getenv("MODEL_VERSION", "default")


def _pil_to_data_url(image) -> str:
    """Encode a PIL image as a base64 data URL so it can be sent offline (no GCS fetch)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

with open(Path(__file__).parent / "vibe_eval.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

EVALUATOR_NAME = config["metadata"]["evaluator"]

_PROMPT_WITH_IMAGE = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, in regards to the image above.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""

_PROMPT_WITH_NO_IMAGE = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, it's about an image shared by the user.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""


@dataclass
class Example:
    """An example loaded from vibe-eval, stored as jsonl in the repo."""

    example_id: str
    category: str
    prompt: str
    reference: str
    media_filename: str
    media_url: str

    # The fields below are not stored in the dataset, but are populated by this script.
    generation: Optional[str] = None
    score: Optional[int] = None
    evaluator_explanation: Optional[str] = None
    image: Optional[object] = None  # PIL image, passed to the image-aware judge


class Evaluator(Enum):
    # Use Reka Core (including image input).
    REKA_CORE = "reka-core"

    # Use Reka Core, only using text input.
    REKA_CORE_TEXT = "reka-core-text"


def make_evaluator_prompt(example: Example, include_image: bool) -> str:
    return (_PROMPT_WITH_IMAGE if include_image else _PROMPT_WITH_NO_IMAGE).format(
        prompt=example.prompt,
        reference=example.reference,
        generation=example.generation,
    )


def evaluate(example: Example, evaluator: Evaluator) -> Example:
    """Evaluates the generation and populates the score and explanation fields.

    Grades via a local vLLM OpenAI-compatible judge (replacing Reka's hosted API). When the
    judge is image-aware (Evaluator.REKA_CORE) and we have the PIL image, the image is sent
    inline as a base64 data URL so the judge sees it, mirroring the original Reka-Core path.
    """
    include_image = (evaluator == Evaluator.REKA_CORE) and (example.image is not None)
    evaluator_prompt = make_evaluator_prompt(example, include_image=include_image)

    client = _get_judge_client()
    if client is None:
        example.score = 0
        example.evaluator_explanation = "Judge client unavailable"
        return example

    content = [{"type": "text", "text": evaluator_prompt}]
    if include_image:
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(example.image)}})

    try:
        response = client.chat.completions.create(
            model=_get_judge_model_name(),
            messages=[{"role": "user", "content": content}],
            temperature=0.4,
            max_tokens=1024,
        )
        evaluator_response = response.choices[0].message.content
    except Exception as e:  # noqa: BLE001 - never let one judge call abort the whole run
        eval_logger.warning(f"vibe_eval judge call failed for {example.example_id}: {e}")
        example.score = 0
        example.evaluator_explanation = f"Judge error: {e}"
        return example

    re_match = re.search(r"Rating:\s*([1-5])", evaluator_response or "")
    if re_match is None:
        example.score = 0
        example.evaluator_explanation = evaluator_response
        return example
    example.score = int(re_match.group(1))
    example.evaluator_explanation = evaluator_response
    return example


def vibe_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vibe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["prompt"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def vibe_process_results(doc, results):
    example_id = doc["example_id"]
    category = doc["category"]
    prompt = doc["prompt"]
    reference = doc["reference"]
    media_filename = doc["media_url"]
    media_url = doc["media_url"]
    generation = results[0]
    image = doc.get("image")  # PIL image embedded in RekaAI/VibeEval; sent to the image-aware judge
    example = Example(example_id=example_id, category=category, prompt=prompt, reference=reference, media_filename=media_filename, media_url=media_url, generation=generation, image=image)

    evaluator = Evaluator.REKA_CORE if EVALUATOR_NAME == "reka-core" else Evaluator.REKA_CORE_TEXT

    example = evaluate(example, evaluator=evaluator)
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


def vibe_aggregation_results(results, category):
    score = []
    for res in results:
        if category in res["category"] or category == "all":
            score.append(res["score"])

    aggregate_scores = _mean(score)
    return aggregate_scores


def vibe_aggregation_results_normal(results):
    return vibe_aggregation_results(results, "normal")


def vibe_aggregation_results_hard(results):
    return vibe_aggregation_results(results, "hard")


def vibe_aggregation_results_all(results):
    return vibe_aggregation_results(results, "all")
