"""Make one picture and exit (so all GPU memory is released). Reads a JSON job on stdin and
prints {"b64": <png>, "device": ...} as the last line of stdout.

Full precision on purpose: GTX 16-series cards produce black images in half precision. If the
card runs out of memory it retries with model offloading (only the active part on the GPU),
and as a last resort draws on the CPU, which is slow but always works.

Every finished picture goes through the Stable Diffusion safety checker on the CPU (so it costs
no GPU memory); an explicit one is discarded and reported as {"error": "explicit"}.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys

SAFETY_REPO = "CompVis/stable-diffusion-safety-checker"


def explicit(image: object, model: str) -> bool:
    """True if the safety checker flags the picture. Runs on the CPU."""

    if os.environ.get("IMAGE_SAFETY", "on") == "off":
        return False
    import numpy as np
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor

    for repo, checker_dir, processor_dir in (
        (model, "safety_checker", "feature_extractor"),
        (SAFETY_REPO, "", ""),
    ):
        try:
            checker = StableDiffusionSafetyChecker.from_pretrained(
                repo, subfolder=checker_dir, local_files_only=True
            )
            processor = CLIPImageProcessor.from_pretrained(
                repo, subfolder=processor_dir, local_files_only=True
            )
            break
        except (OSError, ValueError):
            continue
    else:
        print("no safety checker available; picture not checked", file=sys.stderr)
        return False
    pixels = processor(images=[image], return_tensors="pt").pixel_values
    array = np.asarray(image, dtype="float32")[None] / 255.0
    _, flagged = checker(images=array, clip_input=pixels)
    return bool(flagged[0])


def main() -> None:
    job = json.load(sys.stdin)
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    threads = int(os.environ.get("IMAGE_CPU_THREADS") or 0) or (os.cpu_count() or 4)
    torch.set_num_threads(threads)
    prefer = os.environ.get("IMAGE_DEVICE", "auto")

    def load() -> StableDiffusionPipeline:
        pipe = StableDiffusionPipeline.from_pretrained(
            job["model"],
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        # Some models ship another multistep scheduler's settings (DreamShaper 8: DEIS, with
        # algorithm_type "deis" / solver_type "logrho"), which DPM++ refuses; name its own.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            use_karras_sigmas=True,
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def draw(pipe: StableDiffusionPipeline) -> object:
        return pipe(
            prompt=job["prompt"],
            negative_prompt=job.get("negative_prompt") or None,
            width=job["width"],
            height=job["height"],
            num_inference_steps=job["steps"],
            guidance_scale=6.5,
        ).images[0]

    image, device = None, "cpu"
    if prefer != "cpu" and torch.cuda.is_available():
        oom = torch.cuda.OutOfMemoryError
        pipe = load()
        try:
            pipe.to("cuda")
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            image, device = draw(pipe), "cuda"
        except oom:
            pipe = None
            torch.cuda.empty_cache()
            try:
                pipe = load()
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                image, device = draw(pipe), "cuda (offloaded)"
            except oom:
                torch.cuda.empty_cache()
    if image is None:
        image, device = draw(load()), "cpu"
    if explicit(image, job["model"]):
        print(json.dumps({"error": "explicit", "device": device}))
        return
    buf = io.BytesIO()
    image.save(buf, format="PNG")  # type: ignore[attr-defined]
    print(json.dumps({"b64": base64.b64encode(buf.getvalue()).decode(), "device": device}))


if __name__ == "__main__":
    main()
