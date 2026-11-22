"""Run the paired catalog pilot on a single CUDA device with no network downloads."""
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
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 48
MAX_MESSAGE_BYTES = 131_072
PROFILES = ("clean", "transient")
POLICIES = ("no_retry", "retry_once")
LOOKUP = ('Demande une lecture du catalogue par un seul objet JSON avec les clés product et field. '
          'Recopie la référence produit et utilise stock pour une quantité ou zone pour un emplacement.')
ANSWER = ('Réponds par un seul objet JSON avec la clé answer. Utilise la valeur fournie telle quelle. '
          'Si la lecture a échoué, utilise null.')


def worker_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def decode_message(payload):
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("GPU message exceeds the allowed size")

    def unique_keys(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("Duplicate GPU message key")
        return value

    def reject_constant(_):
        raise ValueError("Non-finite JSON value")

    return json.loads(payload, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def parse_object(text):
    try:
        value = decode_message(text)
        return value if isinstance(value, dict) else {}
    except (ValueError, RecursionError):
        return {}


def tool_result(task, trial, text):
    if parse_object(text) != {"product": task["product"], "field": task["field"]}:
        return {"error": "invalid_lookup"}, ["invalid_lookup"]
    attempts = ["transient_error"] if trial["profile"] == "transient" else []
    if attempts and trial["policy"] == "no_retry":
        return {"error": "transient_error"}, attempts
    return {"value": task["value"]}, [*attempts, "ok"]


def validate_batch(parameters):
    wanted = {"model", "revision", "dtype", "max_input_tokens", "max_new_tokens", "tasks", "trials"}
    if (not isinstance(parameters, dict) or set(parameters) != wanted
            or parameters["model"] != MODEL or parameters["revision"] != REVISION or parameters["dtype"] != "bfloat16"
            or parameters["max_input_tokens"] != MAX_INPUT_TOKENS or parameters["max_new_tokens"] != MAX_NEW_TOKENS):
        raise ValueError("Invalid catalog experiment parameters")
    tasks, trials = parameters["tasks"], parameters["trials"]
    if not isinstance(tasks, dict) or len(tasks) != 8 or not isinstance(trials, list) or len(trials) != 32:
        raise ValueError("Expected 8 catalog tasks and 32 paired trials")
    for key, task in tasks.items():
        if (not re.fullmatch(r"T[0-9]{2}", key) or not isinstance(task, dict)
                or set(task) != {"prompt", "product", "field", "value"}
                or not isinstance(task["prompt"], str) or not 1 <= len(task["prompt"]) <= 300
                or not re.fullmatch(r"[A-Z]{2}-[0-9]{3}", task["product"])
                or task["product"] not in task["prompt"]
                or task["field"] not in ("stock", "zone")
                or not re.fullmatch(r"[0-9]{1,3}" if task["field"] == "stock" else r"[A-Z]-[0-9]{2}", task["value"])):
            raise ValueError("Invalid catalog task")
    if len({task["product"] for task in tasks.values()}) != 8 or sum(t["field"] == "stock" for t in tasks.values()) != 4:
        raise ValueError("Expected distinct products and balanced task fields")
    expected = {f"{task}-{profile}-{policy}" for task in tasks for profile in PROFILES for policy in POLICIES}
    for trial in trials:
        if (not isinstance(trial, dict) or set(trial) != {"trial_id", "task_id", "profile", "policy"}
                or trial["trial_id"] != f"{trial['task_id']}-{trial['profile']}-{trial['policy']}"
                or trial["trial_id"] not in expected):
            raise ValueError("Unexpected or repeated catalog trial")
        expected.remove(trial["trial_id"])


def run_trials(parameters, generate):
    validate_batch(parameters)
    responses = []
    for trial in parameters["trials"]:
        task = parameters["tasks"][trial["task_id"]]
        prompt = task["prompt"]
        lookup = generate(LOOKUP, prompt)
        result, _ = tool_result(task, trial, lookup["text"])
        answer = generate(ANSWER, f"Question : {prompt}\nRésultat de la lecture : {json.dumps(result)}")
        responses.append({"id": trial["trial_id"], "lookup": lookup, "answer": answer})
    return responses


def execute(request):
    if set(request) != {"job_id", "job_sha256", "worker_sha256", "snapshot", "gpu_index", "parameters"}:
        raise ValueError("Invalid GPU request fields")
    if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", request["job_id"])
            or not re.fullmatch(r"[a-f0-9]{64}", request["job_sha256"])
            or request["worker_sha256"] != worker_sha256()):
        raise ValueError("Invalid job or unapproved worker source")
    snapshot = Path(request["snapshot"])
    index = request["gpu_index"]
    if (not snapshot.is_absolute() or snapshot.name != REVISION or not snapshot.is_dir()
            or type(index) is not int or not 0 <= index <= 15):
        raise ValueError("Expected a local pinned snapshot and an approved CUDA device")
    validate_batch(request["parameters"])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def generate(instructions, prompt):
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[-1]
        if not 1 <= input_tokens <= MAX_INPUT_TOKENS:
            raise ValueError("Catalog input exceeds the approved context")
        inputs = inputs.to(device)
        turn_start = perf_counter_ns()
        output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize(device)
        tokens = output[0][input_tokens:]
        return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "input_tokens": input_tokens,
                "output_tokens": len(tokens), "latency_ms": (perf_counter_ns() - turn_start) // 1_000_000}

    with torch.inference_mode():
        responses = run_trials(request["parameters"], generate)
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
    receipt = json.dumps(execute(decode_message(payload)), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(receipt) > MAX_MESSAGE_BYTES:
        raise ValueError("GPU receipt exceeds the allowed size")
    sys.stdout.buffer.write(receipt)


if __name__ == "__main__":
    main()
