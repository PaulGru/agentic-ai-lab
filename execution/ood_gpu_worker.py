"""Standalone Linux worker for one bounded, offline Qwen calibration batch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter_ns

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 16
MAX_MESSAGE_BYTES = 16_384
TRIAL_IDS = ("stock-reference", "zone-ood", "zone-reference", "stock-ood")


def worker_sha256():
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(source).hexdigest()


def decode_message(payload):
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("GPU message exceeds the allowed size")

    def object_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate GPU message key")
        return result

    def invalid_constant(_):
        raise ValueError("Non-finite GPU message value")

    return json.loads(payload, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def validate_request(request):
    if (not isinstance(request, dict) or set(request) != {"job_id", "job_sha256", "model", "revision",
            "worker_sha256", "snapshot", "gpu_index", "trials"}):
        raise ValueError("Invalid GPU request fields")
    if request["model"] != MODEL or request["revision"] != REVISION or request["worker_sha256"] != worker_sha256():
        raise ValueError("Unapproved model revision or worker source")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", request["job_id"]):
        raise ValueError("Invalid calibration job ID")
    if not re.fullmatch(r"[a-f0-9]{64}", request["job_sha256"]):
        raise ValueError("Invalid calibration job hash")
    if type(request["gpu_index"]) is not int or not 0 <= request["gpu_index"] <= 15:
        raise ValueError("Invalid CUDA device index")
    snapshot = Path(request["snapshot"])
    if not snapshot.is_absolute() or snapshot.name != REVISION or not snapshot.is_dir():
        raise ValueError("A local snapshot directory for the approved revision is required")
    trials = request["trials"]
    if not isinstance(trials, list) or len(trials) != len(TRIAL_IDS):
        raise ValueError("Expected the fixed calibration batch")
    for expected, trial in zip(TRIAL_IDS, trials, strict=True):
        if (not isinstance(trial, dict) or set(trial) != {"id", "prompt"} or trial["id"] != expected
                or not isinstance(trial["prompt"], str) or not 1 <= len(trial["prompt"]) <= 512):
            raise ValueError("Invalid calibration trial")
    return snapshot


def execute(request):
    snapshot = validate_request(request)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    index = request["gpu_index"]
    if not torch.cuda.is_available() or index >= torch.cuda.device_count() or torch.version.cuda is None:
        raise RuntimeError("Approved CUDA device is unavailable")
    torch.cuda.set_device(index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device must support bfloat16")
    device = torch.device("cuda", index)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter_ns()
    options = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, **options)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, use_safetensors=True, torch_dtype=torch.bfloat16, **options).to(device).eval()
    responses = []
    with torch.inference_mode():
        for trial in request["trials"]:
            messages = [
                {"role": "system", "content": "Réponds uniquement par la valeur demandée, sans explication."},
                {"role": "user", "content": trial["prompt"]},
            ]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
            input_tokens = inputs["input_ids"].shape[-1]
            if not 1 <= input_tokens <= MAX_INPUT_TOKENS:
                raise ValueError("Calibration input exceeds the approved context")
            inputs = inputs.to(device)
            turn_start = perf_counter_ns()
            output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize(device)
            generated = output[0][input_tokens:]
            responses.append({
                "id": trial["id"], "text": tokenizer.decode(generated, skip_special_tokens=True),
                "input_tokens": input_tokens, "output_tokens": len(generated),
                "latency_ms": (perf_counter_ns() - turn_start) // 1_000_000,
            })
    return {
        "job_id": request["job_id"], "job_sha256": request["job_sha256"], "worker_sha256": worker_sha256(),
        "model": MODEL, "revision": REVISION, "dtype": "bfloat16", "responses": responses,
        "runtime": {
            "gpu_index": index, "gpu_name": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "elapsed_ms": (perf_counter_ns() - started) // 1_000_000,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        },
    }


def main():
    payload = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
    response = execute(decode_message(payload))
    receipt = (json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(receipt) > MAX_MESSAGE_BYTES:
        raise ValueError("GPU receipt exceeds the allowed size")
    sys.stdout.buffer.write(receipt)


if __name__ == "__main__":
    main()
