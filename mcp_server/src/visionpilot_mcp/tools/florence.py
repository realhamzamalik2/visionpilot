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
    try:
        # Use the copy already in the Hugging Face cache first: no network round trips, so loading stays fast
        # (and works) when the internet is slow or offline.
        model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    except OSError:  # not downloaded yet
        model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
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


def find_phrase(
    bundle: FlorenceBundle, image: Image.Image, phrase: str, **generate_kwargs: Any
) -> list[tuple[str, list[float]]]:
    """Ground a free-text phrase ('the red umbrella') to boxes."""
    parsed = run_florence_task(bundle, image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=phrase, **generate_kwargs)
    labels = parsed.get("labels", []) or []
    boxes = parsed.get("bboxes", []) or []
    return [(str(label).strip() or phrase, list(map(float, box))) for label, box in zip(labels, boxes, strict=False)]


def read_text_regions(
    bundle: FlorenceBundle, image: Image.Image, **generate_kwargs: Any
) -> list[tuple[str, list[float]]]:
    """English/Latin-script OCR with quad boxes (8 numbers: x1,y1,...,x4,y4)."""
    parsed = run_florence_task(bundle, image, "<OCR_WITH_REGION>", **generate_kwargs)
    labels = parsed.get("labels", []) or []
    quads = parsed.get("quad_boxes", []) or []
    return [
        (str(text).replace("</s>", "").strip(), list(map(float, q))) for text, q in zip(labels, quads, strict=False)
    ]
