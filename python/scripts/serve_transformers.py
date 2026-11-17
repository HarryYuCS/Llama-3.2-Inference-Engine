"""
Serve the model with the hf transformers library for comparison and simpler verification.
"""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
PROMPT = "What happened in the twentieth century?"


def main() -> None:
    log.info("Loading model from %s", WEIGHTS)
    pipe = pipeline(
        "text-generation",
        model=str(WEIGHTS),
        device_map="auto",
        dtype="auto",
    )

    messages = [{"role": "user", "content": PROMPT}]
    log.info("Running single greedy inference")
    outputs = pipe(messages, max_new_tokens=2048, do_sample=False)

    generated = outputs[0]["generated_text"]
    # Chat pipeline returns the full message list; print the assistant turn.
    if isinstance(generated, list):
        reply = generated[-1].get("content", generated[-1])
    else:
        reply = generated
    log.info("Prompt: %s", PROMPT)
    log.info("Reply: %s", reply)


if __name__ == "__main__":
    main()
