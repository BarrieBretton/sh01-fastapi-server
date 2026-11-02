"""Bounded in-process Qwen2.5-VL inference, loaded separately from the director."""
from __future__ import annotations

import gc
import os


def describe_frames(instruction, frames):
    """Inspect up to eight real frames; release GPU weights before text planning.

    Uses Transformers' public Qwen2.5-VL/AutoProcessor APIs. No paid endpoint,
    remote code execution, video decoder helper, or full-resolution tensor stack.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    model_id = os.environ['AI_COMPOSE_VISION_HF_MODEL'].strip()
    if not torch.cuda.is_available():
        raise RuntimeError('Local vision model needs a CUDA GPU')
    if not frames:
        raise RuntimeError('No decoded frames available for visual review')
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=128*28*28, max_pixels=256*28*28)
    images, content = [], [{'type':'text','text':instruction}]
    for frame in frames[:8]:
        with Image.open(frame['path']) as image:
            images.append(image.convert('RGB'))
        content.extend([{'type':'text','text':f"Time {frame['time']:.2f}s"}, {'type':'image'}])
    model = inputs = outputs = None
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, device_map='auto',
            torch_dtype=torch.float16, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                 bnb_4bit_compute_dtype=torch.float16))
        messages = [{'role':'user','content':content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, padding=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=700, do_sample=False)
        return processor.batch_decode(outputs[:,inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]
    finally:
        del outputs, inputs, model, processor, images
        gc.collect()
        torch.cuda.empty_cache()
