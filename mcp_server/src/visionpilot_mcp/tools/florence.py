"""Florence-2 (230M, MIT license) for captions; later also phrase grounding and English OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

Detail = Literal["brief", "detailed", "very_detailed"]

DETAIL_TO_TASK: dict[str, str] = {
    "brief": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "very_detailed": "<MORE_DETAILED_CAPTION>",
}


@dataclass
class FlorenceBundle:
    model: Any
    processor: Any
    model_id: str


def load_florence(model_id: str, cpu_threads: int) -> FlorenceBundle:
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    torch.set_num_threads(cpu_threads)
    model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return FlorenceBundle(model=model, processor=processor, model_id=model_id)


def run_florence_task(
    bundle: FlorenceBundle,
    image: Image.Image,
    task: str,
    *,
    text_input: str = "",
    num_beams: int = 3,
    max_new_tokens: int = 256,
) -> Any:
    """Run one Florence-2 task prompt and return the parsed answer for that task."""
    import torch

    inputs = bundle.processor(text=task + text_input, images=image, return_tensors="pt")
    with torch.inference_mode():
        generated_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
    generated_text = bundle.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = bundle.processor.post_process_generation(generated_text, task=task, image_size=image.size)
    return parsed[task]


def describe(bundle: FlorenceBundle, image: Image.Image, detail: Detail, **generate_kwargs: Any) -> str:
    caption = run_florence_task(bundle, image, DETAIL_TO_TASK[detail], **generate_kwargs)
    return str(caption).strip()
