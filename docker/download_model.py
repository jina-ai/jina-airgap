"""
Model download script - runs inside the Docker build downloader stage.
Downloads weights AND all model code files into HF cache structure.
At runtime, HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 ensures models load
from this local cache without any network calls.
"""
import os
import sys

os.environ["HF_HOME"] = "/model_cache"
token = os.environ.get("HF_TOKEN") or None
model_id = os.environ.get("MODEL_ID", "")
if not model_id:
    print("ERROR: MODEL_ID env var not set", file=sys.stderr)
    sys.exit(1)
model_revision = os.environ.get("MODEL_REVISION", "").strip() or None

IGNORE = ["*.ot", "*.msgpack", "flax_model*", "tf_model*", "rust_model*"]

from huggingface_hub import snapshot_download


def _repo_cache_dir(repo_id):
    return os.path.join("/model_cache", "hub", "models--" + repo_id.replace("/", "--"))


def fetch(repo_id, revision):
    """Snapshot a repo at a pinned commit, then point refs/main at that commit.

    snapshot_download() only writes refs/<branch> when it resolves a branch name.
    Pinning to a bare commit sha leaves refs/ absent, and at runtime an unpinned
    from_pretrained() resolves "main" through refs/, so an offline container would
    fail to start with "couldn't connect to huggingface.co". Writing the ref
    ourselves reproduces exactly what an unpinned download would have left behind,
    while still guaranteeing the bytes came from the commit we pinned.
    """
    print(f"Downloading {repo_id}" + (f" @ {revision}" if revision else " @ main (UNPINNED)"))
    path = snapshot_download(repo_id, revision=revision, token=token, ignore_patterns=IGNORE)
    if revision:
        refs = os.path.join(_repo_cache_dir(repo_id), "refs")
        os.makedirs(refs, exist_ok=True)
        with open(os.path.join(refs, "main"), "w") as f:
            f.write(revision)
    return path


snapshot_dir = fetch(model_id, model_revision)
print(f"Downloaded {model_id} to /model_cache")

# Fetch auxiliary code/weight repos required for offline model load.
# Driven by `extra_repos` field in models/catalog.json, passed via EXTRA_REPOS env var
# (comma-separated, each entry "repo_id@commit_sha"). For multi-encoder models
# (clip-v2, clip-v1, v4 etc.) the main HF snapshot does not cover the dynamically
# referenced implementation repos.
_extra_repos_raw = os.environ.get("EXTRA_REPOS", "")
for _entry in (r.strip() for r in _extra_repos_raw.split(",") if r.strip()):
    _repo, _, _rev = _entry.partition("@")
    fetch(_repo, _rev or None)
    print(f"  cached {_repo}")

# Patch model code in all cached locations for air-gap compatibility.
# These patches fix bugs in the HF model code that break offline loading.
#
# This has to run before the pre-load further down. The pre-load is what stages
# the .py files into transformers_modules, so it must load the patched copies:
# loading first meant loading code we already know does not import cleanly, and
# the resulting failure left the staging directory half-written.
import glob

# 1. custom_st.py: Multiple patches for air-gap compatibility
import re
for custom_st in glob.glob("/model_cache/**/custom_st.py", recursive=True):
    with open(custom_st, "r") as f:
        src = f.read()
    original = src

    # 1a. Fix default_task extraction (v3 model_args.pop / v5 model_kwargs.pop)
    src = re.sub(
        r"^\s*self\.default_task\s*=\s*model_(?:args|kwargs)\.pop\([^)]*\).*$",
        "        self.default_task = None  # patched for air-gap",
        src,
        flags=re.MULTILINE,
    )

    # 1b. Add trust_remote_code=True to AutoConfig.from_pretrained
    src = src.replace(
        'self.config = AutoConfig.from_pretrained(\n            model_name_or_path, **config_kwargs\n        )',
        'config_kwargs["trust_remote_code"] = True\n        self.config = AutoConfig.from_pretrained(\n            model_name_or_path, **config_kwargs\n        )'
    )

    # 1c. Add trust_remote_code=True to AutoModel.from_pretrained
    src = src.replace(
        'self.model = AutoModel.from_pretrained(\n            model_name_or_path, config=self.config, **model_kwargs\n        )',
        'model_kwargs["trust_remote_code"] = True\n        self.model = AutoModel.from_pretrained(\n            model_name_or_path, config=self.config, **model_kwargs\n        )'
    )

    if src != original:
        with open(custom_st, "w") as f:
            f.write(src)
        print(f"Patched {custom_st}: fixed default_task + trust_remote_code")

# 2. modeling_eurobert.py: EuroBertModel.__init__ needs **kwargs
#    because modeling_jina_embeddings_v5.py passes dtype= to from_pretrained,
#    and transformers 4.48.x forwards unknown kwargs to __init__.
for eurobert in glob.glob("/model_cache/**/modeling_eurobert.py", recursive=True):
    with open(eurobert, "r") as f:
        src = f.read()
    original = src
    old_init = '    def __init__(self, config: EuroBertConfig):\n        super().__init__(config)'
    new_init = '    def __init__(self, config: EuroBertConfig, **kwargs):\n        super().__init__(config)'
    src = src.replace(old_init, new_init)
    if src != original:
        with open(eurobert, "w") as f:
            f.write(src)
        print(f"Patched {eurobert}: added **kwargs to EuroBertModel.__init__")

# 3. modeling_jina_embeddings_v5.py: multiple patches
for modeling in glob.glob("/model_cache/**/modeling_jina_embeddings_v5.py", recursive=True):
    with open(modeling, "r") as f:
        src = f.read()
    original = src
    # 3a. PeftConfig -> LoraConfig
    src = src.replace(
        'from peft import PeftMixedModel, PeftConfig',
        'from peft import PeftMixedModel, PeftConfig, LoraConfig'
    )
    src = src.replace(
        'peft_config = PeftConfig.from_pretrained(',
        'peft_config = LoraConfig.from_pretrained('
    )
    # 3b. Fix dtype -> torch_dtype for Qwen3Model.from_pretrained
    #     transformers >=4.51.0 Qwen3Model doesn't accept 'dtype' kwarg
    src = src.replace(
        'dtype=kwargs.pop("dtype", torch.bfloat16)',
        'torch_dtype=kwargs.pop("dtype", torch.bfloat16)'
    )
    if src != original:
        with open(modeling, "w") as f:
            f.write(src)
        print(f"Patched {modeling}: PeftConfig->LoraConfig + dtype->torch_dtype")

# Remove model-repo requirements.txt files from the HF cache.
# transformers dynamic_module_utils auto-runs pip install from these at runtime,
# which overwrites our pinned versions in an air-gap environment that still has
# internet access during testing, or fails in a true air-gap environment.
# We have already installed the correct pinned versions during the Docker build.
for req_file in glob.glob("/model_cache/**/**/requirements.txt", recursive=True):
    os.remove(req_file)
    print(f"Removed {req_file} (preventing dynamic pip install at runtime)")

# Jina model code imports its siblings absolutely (`from configuration_eurobert
# import ...`, not `from .configuration_eurobert`). transformers resolves those
# through importlib.util.find_spec, which searches sys.path, so without the
# snapshot directory on it check_imports raises ImportError and the module never
# gets staged. That is what left modeling_eurobert.py out of the staging
# directory. server/app.py does the same for the staged copies at runtime.
sys.path.insert(0, snapshot_dir)

# Same two transformers shims server/app.py applies at runtime. Without them the
# pre-load dies before it finishes staging: trust_remote_code prompts for
# confirmation with no tty, and add_generation_mixin_to_remote_model assumes every
# remote model can generate, which embedding models cannot.
try:
    from transformers import dynamic_module_utils as _dmu
    _dmu.resolve_trust_remote_code = lambda *a, **k: True
except Exception:
    pass
try:
    from transformers.models.auto import auto_factory as _af
    _orig_add_gen = getattr(_af, "add_generation_mixin_to_remote_model", None)
    if _orig_add_gen:
        def _safe_add_gen(model_class):
            if not hasattr(model_class, "prepare_inputs_for_generation"):
                return model_class
            return _orig_add_gen(model_class)
        _af.add_generation_mixin_to_remote_model = _safe_add_gen
except Exception:
    pass

# Pre-load to stage the trust_remote_code .py files into
# /model_cache/modules/transformers_modules/..., the directory transformers
# imports from. There is no API for writing it; loading the model once is what
# populates it. Runs last, so the code being staged is already patched and the
# model-repo requirements.txt files are already gone.
_preloaded = False
try:
    from sentence_transformers import SentenceTransformer
    print(f"Pre-loading {model_id} via SentenceTransformer...")
    SentenceTransformer(model_id, trust_remote_code=True, device="cpu")
    _preloaded = True
    print("Dynamic modules cached via SentenceTransformer")
except Exception as e:
    print(f"SentenceTransformer pre-load failed ({type(e).__name__}: {e})")

if not _preloaded:
    # VLM / chat models (jina-vlm, ReaderLM, etc.) are not SentenceTransformer-compatible.
    # Load via AutoProcessor + AutoModelForCausalLM to materialize modeling_*.py /
    # processing_*.py / image_processing_*.py into the dynamic module cache.
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        print(f"Pre-loading {model_id} via AutoModelForCausalLM...")
        try:
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
            print("Processor cached")
        except Exception as ep:
            print(f"AutoProcessor cache skipped ({type(ep).__name__}: {ep})")
        AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, low_cpu_mem_usage=True
        )
        _preloaded = True
        print("Dynamic modules cached via AutoModelForCausalLM")
    except Exception as e:
        print(f"AutoModelForCausalLM pre-load failed ({type(e).__name__}: {e})")

if not _preloaded:
    # Not fatal on its own: colbert (pylate) and some rerankers load through neither
    # of these paths. The staging check below is the real gate.
    print("Warning: all pre-load paths failed; verifying cache state below")

with open("/model_cache/MODEL_ID", "w") as f:
    f.write(model_id)
if model_revision:
    with open("/model_cache/MODEL_REVISION", "w") as f:
        f.write(model_revision)

# Final gate: the trust_remote_code .py files must already be staged under
# /model_cache/modules/transformers_modules. If any are missing, the runtime
# stages them on first load and transformers logs "A new version of the
# following files was downloaded from <hub url>". Nothing is actually fetched
# (HF_HUB_OFFLINE=1 makes that impossible), but the runtime copy fails outright
# on a read-only container filesystem, and the log line reads as an air-gap
# violation.
#
# verify_staging.py resolves the module files directly instead of instantiating
# the model, so it works for every model type, including the colbert/reranker
# paths the pre-load above does not cover. It runs in its own process because it
# has to turn on HF offline mode before transformers is imported.
print("Verifying model code is pre-staged...")
import subprocess

# Detect by grepping the child's output for the literal string an operator would
# see. transformers sets propagate=False on its logger, so an in-process logging
# handler on the root logger never sees this record.
_verify = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_staging.py")
_env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
_res = subprocess.run(
    [sys.executable, _verify, model_id, snapshot_dir],
    env=_env,
    capture_output=True,
    text=True,
)
_output = _res.stdout + _res.stderr
if _res.returncode != 0:
    print("ERROR: could not resolve model code offline.", file=sys.stderr)
    print(_output[-4000:], file=sys.stderr)
    sys.exit(1)
if "was downloaded from" in _output:
    print(
        "ERROR: model code was not fully pre-staged during the build.\n"
        "The runtime would stage it on first load, which fails outright on a\n"
        "read-only container filesystem and logs the warning below, which reads\n"
        "as an air-gap violation. Check the pre-load step above for the\n"
        "exception that stopped it.\n",
        file=sys.stderr,
    )
    print(_output[-4000:], file=sys.stderr)
    sys.exit(1)
print(_res.stdout.strip())

print("Done.")
