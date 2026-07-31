import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import jinja2
import numpy as np
from tqdm import tqdm

from .types import EvalResult, Message, SingleEvalResult


QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response
should be of the following format: 'Answer: $LETTER' (without quotes) where
LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    if stat == "std":
        return np.std(values)
    if stat == "min":
        return np.min(values)
    if stat == "max":
        return np.max(values)
    if stat == "n_samples":
        return len(values)
    if stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    raise ValueError(f"Unknown stat: {stat}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str, ...]] | None = None,
) -> EvalResult:
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for result in single_eval_results:
        for name, value in result.metrics.items():
            name2values[name].append(value)
        if result.score is not None:
            name2values["score"].append(result.score)
        htmls.append(result.html)
        convos.append(result.convo)
        metadata.append(result.example_level_metadata)

    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x
    if not xs:
        return []
    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    with ThreadPool(min(num_threads, len(xs))) as pool:
        return list(pbar_fn(pool.imap(f, xs), total=len(xs)))


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)

_MESSAGE_TEMPLATE = """
<div class="message {{ role }}">
  <div class="role">{{ role }}</div>
  <div class="content"><pre>{{ content }}</pre></div>
</div>
"""


def message_to_html(message: Message) -> str:
    return jinja_env.from_string(_MESSAGE_TEMPLATE).render(
        role=message["role"],
        content=message["content"],
    )


jinja_env.globals["message_to_html"] = message_to_html
