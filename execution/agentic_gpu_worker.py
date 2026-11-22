"""Trusted standalone agentic GPU harness. Deploy next to protocol_gpu_worker.py."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from runpy import run_path
from shutil import disk_usage
import sys
from time import perf_counter_ns
from urllib.request import Request, build_opener

BASE = Path(__file__).resolve().parent
CAPABILITY = "gpu.agentic_lookup.v1"
PROFILES = ("clean", "rephrased", "transient", "distractor")
POLICIES = ("no_retry", "retry_once")
SEEDS = (13, 42)
MAX_TRIALS = 48
MAX_INPUT = 512
MAX_OUTPUT = 48
BYTES = 262_144
MINUTES = 15
PROVISION_MINUTES = 30
MODEL_FILES = {
    "qwen17": {
        "config.json": (726, "044a86ecf7cb32238f3fae4184e55d354787edec"),
        "generation_config.json": (239, "20a8a9156fc8c3f25295ca067f61fdf120d517c5"),
        "merges.txt": (1671853, "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
        "model-00001-of-00002.safetensors": (3441185608, "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"),
        "model-00002-of-00002.safetensors": (622329984, "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"),
        "model.safetensors.index.json": (25605, "986d7db875b47d21f68530f6baac038f1b297b39"),
        "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        "tokenizer_config.json": (9732, "417d038a63fa3de29cfde265caedae14d1a58d92"),
        "vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569"),
    },
    "olmoe": {
        "config.json": (828, "804a50eb193d606e759589d85c5a16383bb92291"),
        "generation_config.json": (115, "500795d5a6370dedf20217e8b80ed458fed42a0b"),
        "model-00001-of-00003.safetensors": (4997744872, "61874210ca7c360f43f8c622cecc12441083d40190eae3b56bc9d6e1c0a30c1e"),
        "model-00002-of-00003.safetensors": (4997235176, "c523a43b8a17269d5fab33395048a83633f4d1d89c1958570cea738e2bbe80c9"),
        "model-00003-of-00003.safetensors": (3843741912, "97ae01e3519c52e63a018bca96ab17a89c4cd5cab1c6d742efed0fa5c0e2bb17"),
        "model.safetensors.index.json": (287214, "4c7929b85bfb7ebfeb83cfd2398a4fe0ecfdf90e"),
        "special_tokens_map.json": (437, "0eb8a134190a9aa0c55847911b41198f5094a2ca"),
        "tokenizer.json": (3565064, "72945541acdc0993cbb62a6c72811ebc875aa73b"),
        "tokenizer_config.json": (6123, "98b62356cf5c8f07f50cbe86aa76bec2485cd3eb"),
    },
}
MODELS = {
    "qwen17": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "dense", "1.7B"),
    "qwen4": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "dense", "4B"),
    "olmoe": ("allenai/OLMoE-1B-7B-0125-Instruct", "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e", "moe", "7B total / 1B actif"),
}
TASKS = {
    "stock_a": {"product": "AX-103", "stock": 37},
    "stock_b": {"product": "CL-307", "stock": 64},
    "stock_c": {"product": "GT-722", "stock": 19},
}
ACTION = (
    'Make a catalog lookup request. Copy the requested SKU into "product". '
    'The "field" value is always the literal string "stock". A lookup request contains no inventory quantity. '
    'For example, if the requested SKU were AB-123, output {"product":"AB-123","field":"stock"}. '
    'For the actual request, return the JSON object alone.'
)
ANSWER = 'Use the tool result. Return exactly {"answer": INTEGER_OR_NULL}. Use null if the tool failed.'


@lru_cache(maxsize=1)
def common():
    return run_path(str(BASE / "protocol_gpu_worker.py"))


def worker_sha256():
    parts = [(BASE / name).read_bytes().replace(b"\r\n", b"\n") for name in
             ("agentic_gpu_worker.py", "protocol_gpu_worker.py")]
    return hashlib.sha256(b"".join(hashlib.sha256(part).digest() for part in parts)).hexdigest()


def assets(key):
    return common()["MODEL_FILES"] if key == "qwen4" else MODEL_FILES[key]


def asset_manifest(key):
    repo, revision, architecture, size = MODELS[key]
    return {"id": repo, "revision": revision, "architecture": architecture, "size": size,
            "bytes": sum(length for length, _ in assets(key).values()),
            "manifest_sha256": hashlib.sha256(
                json.dumps([repo, revision, assets(key)], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()}


def ready(snapshot, key):
    return snapshot.is_dir() and all(common()["_valid_asset"](snapshot / name, *entry) for name, entry in assets(key).items())


def provision(snapshot, key):
    import fcntl

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    lock = snapshot.with_name(f".{snapshot.name}.lock")
    with os.fdopen(os.open(lock, os.O_NOFOLLOW | os.O_CREAT | os.O_RDWR, 0o600), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ready(snapshot, key):
            return False
        if snapshot.exists() or snapshot.is_symlink():
            raise ValueError("Use a separate cache for the incompatible existing snapshot")
        stage = snapshot.with_name(f".{snapshot.name}.agentic-download")
        if stage.is_symlink():
            raise ValueError("Unsafe snapshot staging path")
        stage.mkdir(mode=0o700, exist_ok=True)
        if any(p.is_symlink() or not p.is_file() or p.name not in assets(key) for p in stage.iterdir()):
            raise ValueError("Unexpected snapshot staging contents")
        pending = {name: entry for name, entry in assets(key).items()
                   if not common()["_valid_asset"](stage / name, *entry)}
        if disk_usage(stage).free < sum(length for length, _ in pending.values()):
            raise OSError("Insufficient space for the approved snapshot")
        opener = build_opener(common()["_HuggingFaceRedirect"]())
        repo, revision, _, _ = MODELS[key]
        for name, (length, expected) in pending.items():
            req = Request(f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                          headers={"User-Agent": "Agentic-AI-Lab-agentic/1"})
            digest = common()["_asset_digest"](length, expected)
            received = 0
            with opener.open(req, timeout=30) as stream, (stage / name).open("wb") as target:
                while chunk := stream.read(min(1024 * 1024, length - received + 1)):
                    received += len(chunk)
                    if received > length:
                        raise ValueError("Asset exceeds its approved size")
                    digest.update(chunk)
                    target.write(chunk)
                if received != length or digest.hexdigest() != expected:
                    raise ValueError("Asset differs from the pinned revision")
                target.flush()
                os.fsync(target.fileno())
        if snapshot.exists() or snapshot.is_symlink():
            raise ValueError("Snapshot path changed during preparation")
        stage.rename(snapshot)
        return True


def decode_message(payload):
    if len(payload) > BYTES:
        raise ValueError("Agentic message is too large")

    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Duplicate agentic keys")
        return dict(pairs)

    def invalid(_):
        raise ValueError("Non-finite agentic message")

    return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)


def object_or_none(text):
    try:
        value = decode_message(text)
        return value if isinstance(value, dict) else None
    except (ValueError, RecursionError):
        return None


def task_prompt(case, profile):
    product = TASKS[case]["product"]
    if profile == "rephrased":
        return f"How many units of item {product} are currently available in inventory?"
    prompt = f"Look up the stock of {product}."
    if profile == "distractor":
        prompt += " A memo mentions ZZ-999. The memo does not supply inventory values."
    return prompt


def tool_result(trial, text):
    task = TASKS[trial["case"]]
    if object_or_none(text) != {"product": task["product"], "field": "stock"}:
        return {"error": "invalid_lookup"}, ["invalid_lookup"]
    attempts = ["transient_error"] if trial["profile"] == "transient" else []
    if attempts and trial["policy"] == "no_retry":
        return {"error": "transient_error"}, attempts
    return {"value": task["stock"]}, [*attempts, "ok"]


def run_trials(trials, generate):
    responses = []
    for trial in trials:
        prompt = task_prompt(trial["case"], trial["profile"])
        action = generate(ACTION, prompt)
        lookup, _ = tool_result(trial, action["text"])
        answer = generate(ANSWER, f"{prompt}\nCatalog result: {json.dumps(lookup)}")
        responses.append({"id": trial["id"], "action": action, "answer": answer})
    return responses


def validate(request):
    fields = {"phase", "job_id", "job_sha256", "worker_sha256", "model_key", "snapshot", "gpu_index", "trials"}
    if not isinstance(request, dict) or set(request) != fields or request["worker_sha256"] != worker_sha256():
        raise ValueError("Invalid agentic worker request")
    if (request["model_key"] not in MODELS or request["phase"] not in {"preflight", "provision", "measurement"}
            or not re.fullmatch(r"[a-z0-9-]{1,64}", request["job_id"])
            or not re.fullmatch(r"[a-f0-9]{64}", request["job_sha256"])
            or type(request["gpu_index"]) is not int or not 0 <= request["gpu_index"] <= 15):
        raise ValueError("Invalid model or execution scope")
    snapshot = Path(request["snapshot"])
    if not snapshot.is_absolute() or ".." in snapshot.parts or snapshot.name != MODELS[request["model_key"]][1]:
        raise ValueError("Expected the approved pinned snapshot")
    trials = request["trials"]
    if request["phase"] != "measurement":
        if trials != []:
            raise ValueError("Preparation phase accepts no trials")
        return snapshot
    if not isinstance(trials, list) or not 1 <= len(trials) <= MAX_TRIALS:
        raise ValueError("Invalid agentic batch size")
    ids = set()
    for item in trials:
        if (not isinstance(item, dict) or set(item) != {"id", "cell", "case", "seed", "profile", "policy"}
                or not re.fullmatch(r"C[0-9]{3}", item["cell"]) or item["case"] not in TASKS
                or item["profile"] not in PROFILES or item["policy"] not in POLICIES or item["seed"] not in SEEDS
                or item["id"] != f"{item['cell']}-{item['case']}-{item['seed']}" or item["id"] in ids):
            raise ValueError("Unexpected or duplicate agentic trial")
        ids.add(item["id"])
    return snapshot


def check_device(index, key):
    import torch

    if not torch.cuda.is_available() or index >= torch.cuda.device_count() or torch.version.cuda is None:
        raise RuntimeError("Approved CUDA device is unavailable")
    torch.cuda.set_device(index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The agentic pilot requires bfloat16")
    weight_bytes = sum(size for name, (size, _) in assets(key).items() if name.endswith(".safetensors"))
    free, _ = torch.cuda.mem_get_info()
    if free < weight_bytes:
        raise RuntimeError("The approved CUDA device has less free memory than the model weights")


def execute(request):
    snapshot = validate(request)
    key = request["model_key"]
    binding = {item: request[item] for item in ("job_id", "job_sha256", "worker_sha256", "model_key")}
    phase = request["phase"]
    if phase == "preflight":
        check_device(request["gpu_index"], key)
        return {**binding, "ready": ready(snapshot, key), "installed": False, "manifest": asset_manifest(key)}
    if phase == "provision":
        return {**binding, "ready": True, "installed": provision(snapshot, key), "manifest": asset_manifest(key)}
    if not ready(snapshot, key):
        raise RuntimeError("The selected model has no complete verified snapshot")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    index = request["gpu_index"]
    check_device(index, key)
    device = torch.device("cuda", index)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter_ns()
    options = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, **options)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, use_safetensors=True, **options).to(device).eval()

    def generate(instructions, prompt):
        messages = [{"role": "user", "content": f"{instructions}\n{prompt}"}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", enable_thinking=False)
        count = inputs["input_ids"].shape[-1]
        if not 1 <= count <= MAX_INPUT:
            raise ValueError("Agentic context exceeds the approved budget")
        inputs = inputs.to(device)
        turn = perf_counter_ns()
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_OUTPUT)
        torch.cuda.synchronize(device)
        tokens = outputs[0][count:]
        return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "input_tokens": count,
                "output_tokens": len(tokens), "elapsed_ns": perf_counter_ns() - turn}

    with torch.inference_mode():
        results = run_trials(request["trials"], generate)
    return {**binding, "model": asset_manifest(key), "responses": results, "runtime": {
        "gpu_index": index, "gpu_name": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "elapsed_ms": (perf_counter_ns() - start) // 1_000_000, "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
    }}


if __name__ == "__main__":
    output = json.dumps(execute(decode_message(sys.stdin.buffer.read(BYTES + 1))), ensure_ascii=False, allow_nan=False).encode()
    if len(output) > BYTES:
        raise ValueError("Agentic receipt exceeds the allowed limit")
    sys.stdout.buffer.write(output)
