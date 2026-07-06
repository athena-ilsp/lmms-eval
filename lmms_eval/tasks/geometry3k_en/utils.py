import re

import sympy
from loguru import logger as eval_logger
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_BRACE_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_SQRT_BARE_RE = re.compile(r"\\sqrt\s*([0-9]+(?:\.[0-9]+)?)")

# Golden answers are short LaTeX-ish strings, e.g. "5 \sqrt { 3 }" or "4 \frac { 4 } { 11 }".
# A handful (~0.2%) are ratios/inequalities/equations (e.g. "2:3", "x < 60") that aren't a
# single numeric value; those fall back to normalized string comparison below.
def _latex_to_sympy_str(s: str) -> str:
    s = s.strip()
    s = _FRAC_RE.sub(r"((\1))/((\2))", s)
    s = _SQRT_BRACE_RE.sub(r"sqrt((\1))", s)
    s = _SQRT_BARE_RE.sub(r"sqrt((\1))", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("°", "")
    return s


def _parse_numeric(s: str):
    expr = parse_expr(_latex_to_sympy_str(s), local_dict={"sqrt": sympy.sqrt}, transformations=_TRANSFORMATIONS)
    if not expr.is_number:
        return None
    return expr


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def is_equivalent(prediction: str, answer: str, tol: float = 1e-3) -> bool:
    """Grades a free-form prediction against a golden answer that may be a plain number
    or a LaTeX-ish formula (e.g. "5 \\sqrt { 3 }"). Tries symbolic/numeric equivalence
    first; falls back to normalized string equality for the rare non-numeric answers
    (ratios, inequalities, equations) that don't reduce to a single value.
    """
    try:
        pred_val = _parse_numeric(prediction)
        gold_val = _parse_numeric(answer)
        if pred_val is not None and gold_val is not None:
            return abs(complex(pred_val.evalf()) - complex(gold_val.evalf())) <= tol
    except Exception as e:  # noqa: BLE001 - fall back to string compare on any parse error
        eval_logger.debug(f"geometry3k_en: numeric parse failed for pred={prediction!r} answer={answer!r}: {e}")

    return _normalize_text(prediction) == _normalize_text(answer)


_FINAL_ANSWER_RE = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)


def _extract_final_answer(generation: str) -> str:
    """Pulls the text after the last "Answer:" marker if present, else uses the raw
    generation's last line (robust to models that ignore the requested format).
    """
    matches = list(_FINAL_ANSWER_RE.finditer(generation))
    if matches:
        return matches[-1].group(1).strip().rstrip(".").strip()
    return generation.strip().splitlines()[-1].strip() if generation.strip() else generation.strip()


def geometry3k_en_doc_to_visual(doc):
    return [doc["images"][0].convert("RGB")]


def geometry3k_en_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # The source dataset embeds a literal "<image>" placeholder at the start of `problem`;
    # the actual image is supplied separately via doc_to_visual, so strip the placeholder.
    question = doc["problem"].replace("<image>", "").strip()
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def geometry3k_en_process_results(doc, results):
    generation = results[0]
    prediction = _extract_final_answer(generation)
    answer = doc["answer"]
    correct = is_equivalent(prediction, answer)

    return {
        "exact_match": {
            "prediction": prediction,
            "generation": generation,
            "answer": answer,
            "correct": correct,
        }
    }


def geometry3k_en_aggregate_results(results):
    if not results:
        return 0.0
    return 100.0 * sum(1 for r in results if r["correct"]) / len(results)