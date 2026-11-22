"""Standalone bounded GPU worker for integer prediction from approved prompts."""
import hashlib
import json
import os
from pathlib import Path
import re
from shutil import disk_usage
import sys
from time import perf_counter_ns
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_TRIALS = 32
MAX_INPUT_TOKENS = 480
MAX_NEW_TOKENS = 16
MAX_BYTES = 65_536
PROVISION_MINUTES = 30
# Pinned Hugging Face tree metadata: SHA-256 for LFS files and Git blob IDs for text assets.
MODEL_FILES = {
    "config.json": (727, "6988f134db143052042f2bd6e0c897bc6a605189"),
    "generation_config.json": (238, "432531a002c181a19de338313d2375e9d7494d7e"),
    "merges.txt": (1671839, "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"),
    "model-00001-of-00003.safetensors": (3957900840, "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6"),
    "model-00002-of-00003.safetensors": (3987450520, "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1"),
    "model-00003-of-00003.safetensors": (99630640, "7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d"),
    "model.safetensors.index.json": (32819, "4747b0297d3109f14db49886972e3369c9a00b2a"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9377, "51c1be0d9192e7f6e6596de71d0f07d58fbc32ac"),
    "vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569"),
}


def manifest_sha256():
    return hashlib.sha256(json.dumps([MODEL, REVISION, MODEL_FILES], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_bytes():
    return sum(size for size, _ in MODEL_FILES.values())


def _asset_digest(size, expected):
    if len(expected) == 40:
        return hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    return hashlib.sha256()


def _valid_asset(path, size, expected):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = _asset_digest(size, expected)
    with path.open("rb") as content:
        for chunk in iter(lambda: content.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def snapshot_ready(snapshot):
    return snapshot.is_dir() and all(_valid_asset(snapshot / name, *asset) for name, asset in MODEL_FILES.items())


class _HuggingFaceRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        allowed = {"huggingface.co", "cas-bridge.xethub.hf.co", "cdn-lfs.hf.co",
                   "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "us.aws.cdn.hf.co", "us.gcp.cdn.hf.co"}
        if (parsed.scheme != "https" or parsed.hostname not in allowed or parsed.port not in (None, 443)
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("Unexpected model download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_asset(opener, path, size, expected):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Unexpected partial model asset")
    digest = _asset_digest(size, expected)
    request = Request(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{path.name}",
                      headers={"User-Agent": "Agentic-AI-Lab-model-preparation/1"})
    received = 0
    with opener.open(request, timeout=30) as response, path.open("wb") as output:
        while chunk := response.read(min(1024 * 1024, size - received + 1)):
            received += len(chunk)
            if received > size:
                raise ValueError("Model asset exceeds the approved size")
            digest.update(chunk)
            output.write(chunk)
        if received != size or digest.hexdigest() != expected:
            raise ValueError("Downloaded model asset differs from the approved revision")
        output.flush()
        os.fsync(output.fileno())


def _install(snapshot, opener):
    existing = snapshot.exists()
    if snapshot.is_symlink() or (existing and not snapshot.is_dir()):
        raise ValueError("Existing incompatible snapshot. Select a separate model cache.")
    needed = {}
    for name, asset in MODEL_FILES.items():
        target = snapshot / name
        if target.exists() or target.is_symlink():
            if not _valid_asset(target, *asset):
                raise ValueError("Existing incompatible model asset. Select a separate model cache.")
        else:
            needed[name] = asset
    if not needed:
        return False
    staging = snapshot.with_name(f".{snapshot.name}.download")
    if staging.is_symlink():
        raise ValueError("Model staging directory must be local")
    staging.mkdir(mode=0o700, exist_ok=True)
    if any(path.is_symlink() or not path.is_file() or path.name not in MODEL_FILES for path in staging.iterdir()):
        raise ValueError("Unexpected model staging contents")
    missing = {name: asset for name, asset in needed.items() if not _valid_asset(staging / name, *asset)}
    if disk_usage(snapshot.parent).free < sum(size for size, _ in missing.values()):
        raise OSError("Not enough space to prepare the approved snapshot")
    for name, asset in missing.items():
        _download_asset(opener, staging / name, *asset)
    if existing:
        for name in needed:
            os.link(staging / name, snapshot / name)
            (staging / name).unlink()
    else:
        if snapshot.exists() or snapshot.is_symlink():
            raise ValueError("Snapshot target changed during model preparation")
        staging.rename(snapshot)
    return True


def provision_snapshot(snapshot):
    import fcntl

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    lock = snapshot.with_name(f".{snapshot.name}.lock")
    with os.fdopen(os.open(lock, os.O_NOFOLLOW | os.O_CREAT | os.O_RDWR, 0o600), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _install(snapshot, build_opener(_HuggingFaceRedirect()))


def worker_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def decode_message(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("GPU message exceeds the allowed size")

    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate GPU message key")
        return result

    def reject(_):
        raise ValueError("Non-finite GPU value")

    return json.loads(payload, object_pairs_hook=unique, parse_constant=reject)


def validate(request):
    fields = {"job_id", "job_sha256", "worker_sha256", "snapshot", "gpu_index", "prompts", "phase"}
    if not isinstance(request, dict) or set(request) != fields or request["worker_sha256"] != worker_sha256():
        raise ValueError("Invalid request or worker source")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", request["job_id"]) or not re.fullmatch(
            r"[a-f0-9]{64}", request["job_sha256"]):
        raise ValueError("Invalid job provenance")
    prompts = request["prompts"]
    if (request["phase"] not in {"preflight", "provision", "measurement"} or not isinstance(prompts, list)
            or (not 1 <= len(prompts) <= MAX_TRIALS if request["phase"] == "measurement" else bool(prompts))
            or any(not isinstance(prompt, str) or not 1 <= len(prompt) <= 300 for prompt in prompts)):
        raise ValueError("Expected a bounded list of approved prompts")
    snapshot = Path(request["snapshot"])
    index = request["gpu_index"]
    if (not snapshot.is_absolute() or snapshot.name != REVISION
            or type(index) is not int or not 0 <= index <= 15):
        raise ValueError("Expected a pinned local snapshot and an approved CUDA device")
    return snapshot


def execute(request):
    snapshot = validate(request)
    binding = {key: request[key] for key in ("job_id", "job_sha256", "worker_sha256", "snapshot")}
    if request["phase"] == "preflight":
        return {**binding, "ready": snapshot_ready(snapshot)}
    if request["phase"] == "provision":
        installed = provision_snapshot(snapshot)
        return {**binding, "ready": True, "installed": installed, "manifest_sha256": manifest_sha256()}
    if not snapshot_ready(snapshot):
        raise FileNotFoundError("Model snapshot is incomplete")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    index = request["gpu_index"]
    if not torch.cuda.is_available() or index >= torch.cuda.device_count() or torch.version.cuda is None:
        raise RuntimeError("Approved CUDA device unavailable")
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Approved CUDA device requires bfloat16")
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter_ns()
    options = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, **options)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, use_safetensors=True, **options).to(device).eval()
    responses = []
    with torch.inference_mode():
        for index_in_job, prompt in enumerate(request["prompts"]):
            messages = [{"role": "system", "content": "Réponds uniquement par un entier."},
                        {"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
            count = inputs["input_ids"].shape[-1]
            if not 1 <= count <= MAX_INPUT_TOKENS:
                raise ValueError("Prompt exceeds the approved context")
            inputs = inputs.to(device)
            turn = perf_counter_ns()
            output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize(device)
            tokens = output[0][count:]
            responses.append({"index": index_in_job, "text": tokenizer.decode(tokens, skip_special_tokens=True),
                              "input_tokens": count, "output_tokens": len(tokens),
                              "elapsed_ns": perf_counter_ns() - turn})
    return {
        **{key: request[key] for key in ("job_id", "job_sha256", "worker_sha256")},
        "model": MODEL, "revision": REVISION, "responses": responses, "runtime": {
            "gpu_index": index, "gpu_name": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "elapsed_ms": (perf_counter_ns() - started) // 1_000_000,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        },
    }


if __name__ == "__main__":
    payload = sys.stdin.buffer.read(MAX_BYTES + 1)
    receipt = json.dumps(execute(decode_message(payload)), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(receipt) > MAX_BYTES:
        raise ValueError("GPU receipt exceeds the allowed size")
    sys.stdout.buffer.write(receipt)
