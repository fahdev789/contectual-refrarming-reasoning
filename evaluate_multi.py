from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama


RESULTS_PATH = Path("/content/verification-drop-llm/data/multi_model_results.json")

MODELS = ["deepseek-r1:1.5b", "deepseek-r1:8b", "qwen2.5-math:7b"]

CONDITIONS = {
    "Control": 0,
    "Low": 2,
    "Medium": 6,
    "High": 12,
}

AMBIENT_NOISE = (
    "Agricultural supply chains often experience friction due to regional transport "
    "regulations. Historical grain elevator storage data highlights that macro clerical "
    "errors are common during seasonal transitions. This has minor cascading effects on "
    "the spot price indexes of organic baking ingredients across sub-districts."
)

VERIFICATION_KEYWORD_PATTERNS = [
    r"\bwait\b",
    r"\bhold on\b",
    r"\bhowever\b",
    r"\bbut if\b",
    r"\balternatively\b",
    r"\bincorrect\b",
    r"\blet me re-evaluate\b",
    r"\bcorrection\b",
]

VERIFICATION_REGEX = re.compile("|".join(VERIFICATION_KEYWORD_PATTERNS), re.IGNORECASE)
THINK_BLOCK_REGEX = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
WORD_REGEX = re.compile(r"\b\w+\b")


@dataclass(frozen=True)
class MathProblem:
    problem_id: str
    prompt: str
    expected: str


PROBLEMS = [
    MathProblem(
        problem_id="MATH_01_ARITHMETIC",
        prompt=(
            "An assembly line produces 15 parts in the first hour. Each subsequent hour, "
            "its production increases by exactly 4 parts due to system warmup optimizations. "
            "How many total parts does the line assemble during an 8-hour shift?"
        ),
        expected="232",
    ),
    MathProblem(
        problem_id="MATH_02_FRACTIONS",
        prompt=(
            "A laboratory mixture requires 120 liters of base liquid. Assistant A pours in "
            "1/3 of the required total volume, and Assistant B pours in 2/5 of the remaining "
            "unfilled volume. How many liters of base liquid are still required to complete "
            "the mixture?"
        ),
        expected="48",
    ),
    MathProblem(
        problem_id="MATH_03_COMBINATORICS",
        prompt=(
            "A security server must assign unique codes to hardware units. Each code consists "
            "of 2 distinct letters chosen from {A, B, C, D} followed immediately by 2 distinct "
            "digits chosen from {1, 2, 3}. How many unique security codes can be generated?"
        ),
        expected="72",
    ),
]


def build_prompt(problem: MathProblem, noise_multiplier: int) -> str:
    noise_block = "\n\n".join([AMBIENT_NOISE] * noise_multiplier)
    parts = [
        "Solve the following math problem carefully.",
        "Put any private reasoning inside <think>...</think> tags.",
        "After the reasoning, provide the final answer as a concise sentence.",
    ]
    if noise_block:
        parts.extend(["Contextual background text:", noise_block])
    parts.extend(["Question:", problem.prompt])
    return "\n\n".join(parts)


def parse_response(response_text: str) -> tuple[str, str]:
    trace_parts = THINK_BLOCK_REGEX.findall(response_text)
    reasoning_trace = "\n\n".join(part.strip() for part in trace_parts if part.strip())
    final_answer = THINK_BLOCK_REGEX.sub("", response_text).strip()
    return reasoning_trace, final_answer


def count_words(text: str) -> int:
    return len(WORD_REGEX.findall(text))


def verification_density(reasoning_trace: str) -> dict[str, float | int]:
    keyword_hits = len(VERIFICATION_REGEX.findall(reasoning_trace))
    trace_word_count = count_words(reasoning_trace)
    density = keyword_hits / trace_word_count if trace_word_count else 0.0
    return {
        "verification_keyword_hits": keyword_hits,
        "reasoning_trace_word_count": trace_word_count,
        "verification_density": density,
    }


def run_problem(model: str, condition: str, noise_multiplier: int, problem: MathProblem) -> dict[str, Any]:
    prompt = build_prompt(problem, noise_multiplier)
    started_at = time.perf_counter()
    raw = ollama.generate(model=model, prompt=prompt)
    elapsed_seconds = time.perf_counter() - started_at

    response_text = str(raw.get("response", ""))
    reasoning_trace, final_answer = parse_response(response_text)
    density_metrics = verification_density(reasoning_trace)
    is_correct = problem.expected.lower() in final_answer.lower()

    return {
        "problem_id": problem.problem_id,
        "condition": condition,
        "noise_multiplier": noise_multiplier,
        "noise_token_count": count_words(" ".join([AMBIENT_NOISE] * noise_multiplier)),
        "expected_answer": problem.expected,
        "is_correct": is_correct,
        "final_answer_text": final_answer,
        "reasoning_trace": reasoning_trace,
        "elapsed_seconds": elapsed_seconds,
        **density_metrics,
    }


def evaluate() -> dict[str, Any]:
    results: dict[str, Any] = {
        "metadata": {
            "experiment": "Verification Drop Tracking",
            "ambient_noise_text": AMBIENT_NOISE,
            "verification_keyword_patterns": VERIFICATION_KEYWORD_PATTERNS,
            "models": MODELS,
            "conditions": CONDITIONS,
            "problems": [
                {
                    "problem_id": problem.problem_id,
                    "problem": problem.prompt,
                    "expected_answer": problem.expected,
                }
                for problem in PROBLEMS
            ],
        },
        "results": {},
    }

    for model in MODELS:
        model_results: dict[str, Any] = {}
        for condition, noise_multiplier in CONDITIONS.items():
            task_results = [
                run_problem(model=model, condition=condition, noise_multiplier=noise_multiplier, problem=problem)
                for problem in PROBLEMS
            ]
            densities = [task["verification_density"] for task in task_results]
            accuracy = sum(1 for task in task_results if task["is_correct"]) / len(task_results)
            model_results[condition] = {
                "noise_multiplier": noise_multiplier,
                "noise_token_count": count_words(" ".join([AMBIENT_NOISE] * noise_multiplier)),
                "mean_verification_density_score": sum(densities) / len(densities),
                "accuracy": accuracy,
                "tasks": task_results,
            }
        results["results"][model] = model_results

    return results


def main() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = evaluate()
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
