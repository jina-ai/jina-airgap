#!/usr/bin/env python3
"""
jina-on-prem - On-Prem Deployment Toolkit for Jina AI Models

Two-phase workflow:

  Phase 1 - BUNDLE (requires network):
    Build a self-contained Docker image bundle with model weights baked in.
    Run this on a machine with internet access.

  Phase 2 - DEPLOY (no network required):
    Load and run the bundle on a fully disconnected air-gapped machine.
    Zero external dependencies at runtime.

Usage:
  python jina-on-prem.py                       # Layer 0: brief command list
  python jina-on-prem.py <command> --help      # Layer 1: command help + examples
  python jina-on-prem.py list
  python jina-on-prem.py bundle --model <id>
  python jina-on-prem.py deploy --image <file>
  python jina-on-prem.py serve --model <id>
"""

import os
import sys
import json
import signal
import argparse
import difflib
import gzip
import importlib.util
import subprocess
import textwrap
import tempfile
from pathlib import Path

VERSION = "0.1.0"

SCRIPT_DIR = Path(__file__).parent.resolve()
CATALOG_PATH = SCRIPT_DIR / "models" / "catalog.json"

# Exit codes
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_INTERRUPTED = 130

# Detect if we're in a TTY (pipe-safe output)
IS_TTY = sys.stderr.isatty()

# ANSI colors - only when stderr is a TTY (stderr is our diagnostic channel)
if IS_TTY:
    BOLD   = "\033[1m"
    GREEN  = "\033[32m"
    CYAN   = "\033[36m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"
else:
    BOLD = GREEN = CYAN = YELLOW = RED = DIM = RESET = ""


def err(text, color=""):
    """Write to stderr (diagnostic channel)."""
    print(f"{color}{text}{RESET}", file=sys.stderr)


def out(text=""):
    """Write to stdout (data channel)."""
    print(text)


def load_catalog():
    if not CATALOG_PATH.exists():
        err(f"Error: Catalog not found: {CATALOG_PATH}", RED)
        err(f"Fix: Make sure you are running from the jina-on-prem directory.", YELLOW)
        sys.exit(EXIT_RUNTIME_ERROR)
    with open(CATALOG_PATH) as f:
        return json.load(f)["models"]


def fuzzy_match(model_id: str, models: list) -> str | None:
    """Return the closest matching model ID, or None."""
    ids = [m["id"] for m in models]
    matches = difflib.get_close_matches(model_id, ids, n=1, cutoff=0.4)
    return matches[0] if matches else None


def find_model(model_id: str, models: list) -> dict:
    """Find model by exact ID, with helpful fuzzy-match error."""
    for m in models:
        if m["id"] == model_id:
            return m
    suggestion = fuzzy_match(model_id, models)
    if suggestion:
        err(f"Error: Model '{model_id}' not found. Did you mean '{suggestion}'?", RED)
    else:
        err(f"Error: Model '{model_id}' not found.", RED)
    err(f"Fix: Run 'python jina-on-prem.py list' to see all available models.", YELLOW)
    sys.exit(EXIT_USER_ERROR)


def print_banner():
    if IS_TTY:
        err(f"{CYAN}{BOLD}")
        err(r"     _ _             _    ___  ___                      _   ")
        err(r"    | (_)           | |   |  \/  |                     | |  ")
        err(r"    | |_ _ __   __ _| |   | .  . |  ___  __| | ___| |___   ")
        err(r"    | | | '_ \ / _` | |   | |\/| | / _ \/ _` |/ _ \ / __|  ")
        err(r"    | | | | | | (_| |_|   | |  | ||  __/ (_| |  __/ \__ \  ")
        err(r"    |_|_|_| |_|\__,_(_)   \_|  |_/ \___|\__,_|\___|_|___/  ")
        err(f"      On-Prem Deployment Toolkit for Jina AI Models  v{VERSION}{RESET}")
        err("")
    else:
        err(f"=== Jina AI On-Prem Toolkit v{VERSION} ===")


# ---------------------------------------------------------------------------
# list subcommand
# ---------------------------------------------------------------------------

def cmd_list(args):
    models = load_catalog()

    # Filtering
    if args.type:
        models = [m for m in models if m.get("type") == args.type]
        if not models:
            err(f"Error: No models with type '{args.type}'.", RED)
            err(f"Fix: Valid types are: embedding, reranker, reader, colbert, vlm", YELLOW)
            sys.exit(EXIT_USER_ERROR)
    if args.modality:
        models = [m for m in models if m.get("modality", "text") == args.modality]
        if not models:
            err(f"Error: No models with modality '{args.modality}'.", RED)
            err(f"Fix: Valid modalities are: text, multimodal, code", YELLOW)
            sys.exit(EXIT_USER_ERROR)

    if args.json:
        out(json.dumps(models, indent=2))
        sys.exit(EXIT_OK)

    # Human-readable table
    print_banner()
    filters = []
    if args.type:
        filters.append(f"type={args.type}")
    if args.modality:
        filters.append(f"modality={args.modality}")
    filter_str = f"  [{', '.join(filters)}]" if filters else ""
    err(f"{BOLD}Available Models{RESET} ({len(models)} total{filter_str})\n")

    col_id     = max(len(m["id"]) for m in models) + 2
    col_type   = 12
    col_params = 8
    col_rel    = 12

    header = (
        f"  {'MODEL':<{col_id}} {'TYPE':<{col_type}} {'PARAMS':<{col_params}} {'RELEASED':<{col_rel}}"
    )
    if args.verbose:
        header += f"  {'VRAM':>6}  {'DIMS':>6}  MODALITY    LICENSE"

    err(f"{BOLD}{header}{RESET}")
    err("  " + "-" * (len(header) - 2))

    for m in models:
        mod = m.get("modality", "text")
        if mod == "multimodal":
            mod_label = f"{CYAN}multimodal{RESET}"
        elif mod == "code":
            mod_label = f"{YELLOW}code      {RESET}"
        else:
            mod_label = f"{GREEN}text      {RESET}"

        row = (
            f"  {m['id']:<{col_id}} "
            f"{m.get('type',''):<{col_type}} "
            f"{m.get('parameters',''):<{col_params}} "
            f"{m.get('release',''):<{col_rel}}"
        )
        if args.verbose:
            vram = str(m.get("vram_gb", "?")) + "GB"
            dims = str(m.get("output_dim", "-"))
            ctx  = str(m.get("context", "-"))
            row += (
                f"  {vram:>6}  {dims:>6}  {mod_label}  "
                f"{m.get('license','')}"
            )
        err(row)

    err("")
    if not args.verbose:
        err(f"{DIM}  Tip: use -v / --verbose to see VRAM, dims, modality, license{RESET}")
        err(f"{DIM}  Tip: use --type / --modality to filter, --json for machine-readable output{RESET}")

    sys.exit(EXIT_OK)


# ---------------------------------------------------------------------------
# bundle subcommand
# ---------------------------------------------------------------------------

def get_dockerfile_path(runtime: str = "gpu") -> Path:
    """Return the Dockerfile path for the given runtime (gpu or cpu)."""
    candidates = [
        SCRIPT_DIR / "docker" / f"Dockerfile.{runtime}",
        SCRIPT_DIR / "docker" / "Dockerfile",  # legacy fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # will fail with a clear error downstream


def build_model_requirements(model: dict) -> str:
    deps = model.get("deps", {})
    lines = []
    for pkg, spec in deps.items():
        if spec is None:
            continue
        lines.append(f"{pkg}{spec}")
    return "\n".join(lines) + "\n" if lines else ""


def check_docker():
    """Check Docker is installed and running. Exit with helpful message if not."""
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err("Error: Docker is not installed or not running.", RED)
        err("Fix: Install Docker from https://docs.docker.com/get-docker/", YELLOW)
        err("     Then start Docker Desktop (or: sudo systemctl start docker)", YELLOW)
        sys.exit(EXIT_RUNTIME_ERROR)


def cmd_bundle(args):
    models = load_catalog()

    # Resolve model
    if args.model:
        model = find_model(args.model, models)
    elif args.dry_run:
        err("Error: --dry-run requires --model.", RED)
        err("Fix: python jina-on-prem.py bundle --dry-run --model jina-embeddings-v5-text-nano", YELLOW)
        sys.exit(EXIT_USER_ERROR)
    else:
        # Interactive selection only when TTY
        if not IS_TTY:
            err("Error: --model is required when not running interactively.", RED)
            err("Fix: python jina-on-prem.py bundle --model <model-id>", YELLOW)
            err("     Run 'python jina-on-prem.py list' to see all models.", YELLOW)
            sys.exit(EXIT_USER_ERROR)
        model = _select_model_interactive(models)

    # Resolve runtime
    if args.cpu_only:
        runtime = "cpu"
    else:
        runtime = "gpu"

    # Derived values
    image_tag   = f"jina/{model['id'].lower()}:{runtime}"
    output_file = args.output or f"{model['id'].lower()}-{runtime}.tar.gz"

    if args.dry_run:
        print_banner()
        err(f"{BOLD}Dry run: bundle plan{RESET}\n")
        err(f"  Model:   {model['id']}")
        err(f"  HF Repo: {model['hf_repo']}")
        err(f"  Type:    {model['type']}  |  Params: {model['parameters']}  |  VRAM: ~{model.get('vram_gb','?')}GB")
        err(f"  Runtime: {runtime}")
        err(f"  Tag:     {image_tag}")
        err(f"  Output:  {output_file}")
        deps_str = build_model_requirements(model).strip()
        if deps_str:
            err(f"\n  Model deps:\n    " + deps_str.replace("\n", "\n    "))
        err(f"\n  Dockerfile: {get_dockerfile_path(runtime)}")
        err(f"\n{GREEN}Nothing built. Remove --dry-run to proceed.{RESET}")
        sys.exit(EXIT_OK)

    # --- Real build ---
    print_banner()
    err(f"{BOLD}Phase 1: BUNDLE{RESET} {YELLOW}(requires network){RESET}")
    err("  Downloads model weights and bakes them into a Docker image.")
    err(f"  Output: self-contained .tar.gz - no network needed at runtime.\n")

    err(f"{BOLD}Step 1/4{RESET} Resolving model...")
    err(f"  Model:   {BOLD}{model['id']}{RESET}")
    err(f"  HF Repo: {model['hf_repo']}")
    err(f"  Type:    {model['type']}  |  Params: {model['parameters']}  |  VRAM: ~{model.get('vram_gb','?')}GB")
    err(f"  Runtime: {runtime}")
    err(f"  Output:  {output_file}")

    # Confirm
    if not args.yes and IS_TTY:
        try:
            confirm = input(f"\n{BOLD}Proceed? [y/N]: {RESET}").strip().lower()
        except KeyboardInterrupt:
            err("\nInterrupted.", YELLOW)
            sys.exit(EXIT_INTERRUPTED)
        if confirm not in ("y", "yes"):
            err("Cancelled.")
            sys.exit(EXIT_OK)

    check_docker()

    dockerfile = get_dockerfile_path(runtime)
    if not dockerfile.exists():
        err(f"Error: Dockerfile not found: {dockerfile}", RED)
        err("Fix: Make sure you are running from the jina-on-prem directory.", YELLOW)
        sys.exit(EXIT_RUNTIME_ERROR)

    # Write model-specific requirements into docker/ build context so COPY works
    err(f"\n{BOLD}Step 2/4{RESET} Writing deps...")
    model_reqs_path = SCRIPT_DIR / "docker" / "model-requirements.txt"
    model_reqs_content = build_model_requirements(model)
    if model_reqs_content:
        model_reqs_path.write_text(model_reqs_content)
        err(f"  Wrote {len(model_reqs_content.splitlines())} dependency lines to docker/model-requirements.txt")
    else:
        model_reqs_path.write_text("# no model-specific deps\n")
        err("  No model-specific deps")

    # Build Docker image
    err(f"\n{BOLD}Step 3/4{RESET} Building Docker image...")
    err(f"  {DIM}This downloads model weights - 3-30 min depending on model and network{RESET}")

    build_env = os.environ.copy()
    build_env["DOCKER_BUILDKIT"] = "1"

    build_args = [
        "docker", "build",
        "-f", str(dockerfile),
        "--build-arg", f"MODEL_ID={model['hf_repo']}",
        "-t", image_tag,
    ]
    build_args += ["--build-arg", f"EXTRA_REPOS={','.join(model.get('extra_repos', []))}"]
    # Pin the model to the exact HF commit recorded in the catalog so a rebuild of
    # the same tag ships identical code and weights. Absent = track main.
    if model.get("revision"):
        build_args += ["--build-arg", f"MODEL_REVISION={model['revision']}"]
    if args.hf_token:
        build_args += ["--build-arg", f"HF_TOKEN={args.hf_token}"]
    # GPU dtype override per model — some architectures (jina-bert ALiBi: v1/v2-base)
    # overflow to NaN under fp16. Defaults to float16 in the Dockerfile if absent.
    if not args.cpu_only and model.get("gpu_dtype"):
        build_args += ["--build-arg", f"DTYPE={model['gpu_dtype']}"]
    build_args.append(str(SCRIPT_DIR))

    err(f"  {DIM}Running: DOCKER_BUILDKIT=1 docker build ... -t {image_tag}{RESET}")

    result = subprocess.run(build_args, capture_output=True, text=True, env=build_env)

    # Cleanup the temporary requirements file from build context
    model_reqs_path.unlink(missing_ok=True)

    if result.returncode != 0:
        err(f"\n{BOLD}Error: Docker build failed (exit {result.returncode}){RESET}", RED)
        err("\nLast 20 lines of Docker output:", YELLOW)
        lines = (result.stdout + result.stderr).splitlines()
        for line in lines[-20:]:
            err(f"  {line}")
        err("\nCommon fixes:", YELLOW)
        err("  - HF gated model: add --hf-token YOUR_TOKEN")
        err("  - Disk space:     docker system prune")
        err("  - Network:        ensure internet access for Phase 1")
        sys.exit(EXIT_RUNTIME_ERROR)

    # Report final image size
    size_result = subprocess.run(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Size}}"],
        capture_output=True, text=True
    )
    image_size_mb = ""
    if size_result.returncode == 0:
        try:
            image_bytes = int(size_result.stdout.strip())
            image_size_mb = f" ({image_bytes / (1024**3):.1f} GB uncompressed)"
        except ValueError:
            pass
    err(f"  {GREEN}Build complete.{RESET}{image_size_mb}")

    # Save image to tar.gz
    err(f"\n{BOLD}Step 4/4{RESET} Saving bundle to: {output_file}")
    err(f"  {DIM}This may take a few minutes...{RESET}")

    save_cmd = f"docker save {image_tag} | gzip > {output_file}"
    save_result = subprocess.run(save_cmd, shell=True)
    if save_result.returncode != 0:
        err(f"Error: Failed to save bundle to {output_file}", RED)
        err("Fix: Check disk space (docker images -a to see image size)", YELLOW)
        sys.exit(EXIT_RUNTIME_ERROR)

    size_mb = Path(output_file).stat().st_size / (1024 * 1024)

    bundle_meta = {
        "model_id": model["id"],
        "hf_repo": model["hf_repo"],
        "runtime": runtime,
        "image_tag": image_tag,
        "output_file": str(Path(output_file).resolve()),
        "size_mb": round(size_mb, 1),
    }

    if args.json:
        out(json.dumps(bundle_meta, indent=2))
    else:
        err(f"\n{GREEN}{BOLD}Bundle ready!{RESET} {size_mb:.0f} MB saved to: {output_file}")
        gpu_flag = " --gpu" if runtime == "gpu" else ""
        raw_gpu = " --gpus all" if runtime == "gpu" else ""
        err(f"\n{BOLD}Deploy on air-gapped machine:{RESET}")
        err(f"  python jina-on-prem.py deploy --image {output_file}{gpu_flag}")
        err(f"\n{BOLD}Or raw Docker:{RESET}")
        err(f"  docker load < {output_file}")
        err(f"  docker run{raw_gpu} -p 8080:8080 {image_tag}")
        err(f"  curl http://localhost:8080/health")
        err(f"\n{DIM}Air-gap is enforced by HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 baked into")
        err(f"the image - no network mode flag needed. To prove it, run on a host with no")
        err(f"egress route and watch docker logs.{RESET}")

    sys.exit(EXIT_OK)


# ---------------------------------------------------------------------------
# deploy subcommand
# ---------------------------------------------------------------------------

def validate_gzip(path: str) -> bool:
    """Check if file is a valid gzip archive."""
    try:
        with gzip.open(path, "rb") as f:
            f.read(16)
        return True
    except Exception:
        return False


def find_free_port(preferred: int) -> int | None:
    """Check if preferred port is free. Return alternative if not."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", preferred))
            return preferred
        except OSError:
            # Try nearby ports
            for alt in range(preferred + 1, preferred + 20):
                try:
                    s.bind(("", alt))
                    return alt
                except OSError:
                    continue
    return None


def cmd_deploy(args):
    image_file = args.image

    # Validate file exists
    if not Path(image_file).exists():
        err(f"Error: File not found: {image_file}", RED)
        err(f"Fix: Check the path. Usage: python jina-on-prem.py deploy --image <path.tar.gz>", YELLOW)
        sys.exit(EXIT_USER_ERROR)

    # Validate it's a valid gzip
    if not validate_gzip(image_file):
        err(f"Error: '{image_file}' is not a valid gzip archive.", RED)
        err(f"Fix: Make sure the file was created by 'python jina-on-prem.py bundle'", YELLOW)
        err(f"     and was not corrupted during transfer.", YELLOW)
        sys.exit(EXIT_USER_ERROR)

    check_docker()

    port = args.port

    # Check port availability
    free_port = find_free_port(port)
    if free_port is None:
        err(f"Error: Port {port} is in use and no alternative found nearby.", RED)
        err(f"Fix: Stop the process using port {port}, or use --port <other>", YELLOW)
        sys.exit(EXIT_RUNTIME_ERROR)
    if free_port != port:
        err(f"{YELLOW}Warning: Port {port} is in use. Using port {free_port} instead.{RESET}")
        port = free_port

    print_banner()
    err(f"{BOLD}Phase 2: DEPLOY{RESET} {GREEN}(no network required){RESET}")
    err("  Loading air-gapped bundle and starting container.\n")

    err(f"  Loading: {image_file}")
    load_result = subprocess.run(
        ["docker", "load", "-i", image_file],
        capture_output=True, text=True
    )
    if load_result.returncode != 0:
        err(f"Error: Failed to load bundle: {image_file}", RED)
        last_err = load_result.stderr.strip().splitlines()
        for line in last_err[-5:]:
            err(f"  {line}")
        err(f"Fix: Ensure the file is a valid Docker image bundle (created by 'bundle' command)", YELLOW)
        sys.exit(EXIT_RUNTIME_ERROR)

    image_name = None
    for line in load_result.stdout.splitlines():
        if "Loaded image:" in line:
            image_name = line.split("Loaded image:")[-1].strip()
            break

    if not image_name:
        err(f"{YELLOW}Warning: Could not auto-detect image name.{RESET}")
        if IS_TTY:
            try:
                image_name = input(f"{BOLD}Enter image name to run: {RESET}").strip()
            except KeyboardInterrupt:
                err("\nInterrupted.", YELLOW)
                sys.exit(EXIT_INTERRUPTED)
        else:
            err("Error: Cannot determine image name. Run interactively to enter manually.", RED)
            sys.exit(EXIT_RUNTIME_ERROR)

    err(f"  {GREEN}Loaded:{RESET} {image_name}")

    # Build run command
    run_cmd = ["docker", "run", "--rm"]

    if args.detach:
        run_cmd.append("-d")

    if args.name:
        run_cmd += ["--name", args.name]

    if args.gpu:
        run_cmd += ["--gpus", "all"]

    run_cmd += ["-p", f"{port}:8080", image_name]

    err(f"\n  {DIM}Running: {' '.join(run_cmd)}{RESET}\n")

    result = subprocess.run(run_cmd, capture_output=args.detach, text=args.detach)

    if args.detach:
        if result.returncode != 0:
            err(f"Error: Failed to start container.", RED)
            err(result.stderr.strip())
            sys.exit(EXIT_RUNTIME_ERROR)
        container_id = result.stdout.strip()
        err(f"{GREEN}Container started:{RESET} {container_id[:12]}")
        err(f"\n  Test:  curl http://localhost:{port}/health")
        err(f"  Logs:  docker logs {container_id[:12]}")
        err(f"  Stop:  docker stop {container_id[:12]}")
    else:
        if result.returncode not in (0, 130):
            sys.exit(EXIT_RUNTIME_ERROR)
        err(f"\nTest the service:")
        err(f"  curl http://localhost:{port}/health")

    sys.exit(EXIT_OK)


# ---------------------------------------------------------------------------
# serve subcommand
# ---------------------------------------------------------------------------

def check_packages(model: dict):
    """Check required Python packages; show install command if missing."""
    deps = model.get("deps", {})
    missing = []
    for pkg, spec in deps.items():
        if spec is None:
            continue
        import_name = pkg.replace("-", "_").lower()
        # Handle known import name aliases
        aliases = {
            "pillow": "PIL",
            "sentence_transformers": "sentence_transformers",
            "flash_attn": "flash_attn",
        }
        import_name = aliases.get(import_name, import_name)
        if importlib.util.find_spec(import_name) is None:
            missing.append(f"{pkg}{spec}")
    if missing:
        err(f"{YELLOW}Warning: Missing packages required for this model:{RESET}")
        err(f"  {', '.join(missing)}")
        err(f"\nFix:")
        err(f"  pip install {' '.join(missing)}")
        return False
    return True


def cmd_serve(args):
    models = load_catalog()

    if args.model:
        # Try catalog first, fall back to treating as HF repo
        for m in models:
            if m["id"] == args.model:
                model = m
                break
        else:
            model = {"id": args.model, "type": "embedding", "hf_repo": args.model, "deps": {}}
    elif args.local_path:
        model = {"id": Path(args.local_path).name, "type": "embedding", "hf_repo": args.local_path, "deps": {}}
    else:
        if not IS_TTY:
            err("Error: --model or --local-path is required.", RED)
            err("Fix: python jina-on-prem.py serve --model <id>", YELLOW)
            err("     Run 'python jina-on-prem.py list' to see all models.", YELLOW)
            sys.exit(EXIT_USER_ERROR)
        model = _select_model_interactive(models)

    # Check packages
    all_present = check_packages(model)
    if not all_present:
        err("")

    port = args.port

    # Device selection
    device = args.device
    env = os.environ.copy()
    env["JINA_MODEL_ID"] = args.local_path or model["hf_repo"]
    env["PORT"] = str(port)

    if device == "cpu" or (device == "auto" and not _has_cuda()):
        env["CUDA_VISIBLE_DEVICES"] = ""
        effective_device = "cpu"
    else:
        effective_device = "cuda"

    server_script = SCRIPT_DIR / "server" / "app.py"
    if not server_script.exists():
        err(f"Error: Server script not found: {server_script}", RED)
        err(f"Fix: Make sure you are running from the jina-on-prem directory.", YELLOW)
        sys.exit(EXIT_RUNTIME_ERROR)

    err(f"  Model:  {env['JINA_MODEL_ID']}")
    err(f"  Port:   {port}")
    err(f"  Device: {effective_device}")
    err(f"\n  Test: curl http://localhost:{port}/health\n")

    subprocess.run([sys.executable, str(server_script)], env=env)
    sys.exit(EXIT_OK)


# ---------------------------------------------------------------------------
# keygen subcommand
# ---------------------------------------------------------------------------

def cmd_keygen(args):
    """Mint a time-sensitive license key. Offline, no rebuild, no Docker.

    The key is validated by the running server's license gate. Secret defaults
    to the public constant baked into the image; pass --secret to match a
    custom LICENSE_SECRET build/run override.
    """
    # Import the same signer the server uses so format never drifts.
    sys.path.insert(0, str(SCRIPT_DIR / "server"))
    try:
        import license as _license
    except Exception as e:
        err(f"Error: cannot import server/license.py: {e}", RED)
        sys.exit(EXIT_RUNTIME_ERROR)

    key = _license.issue(
        sub=args.sub,
        days=args.days,
        model=args.model or "*",
        secret=args.secret,
    )
    claims = _license.inspect(key)
    import time as _t
    exp_h = _t.strftime("%Y-%m-%d %H:%M:%SZ", _t.gmtime(claims["exp"]))

    if args.json:
        out(json.dumps({"key": key, "claims": claims, "expires": exp_h}, indent=2))
        sys.exit(EXIT_OK)

    print_banner()
    err(f"{BOLD}License key minted{RESET}\n")
    err(f"  Licensed to: {claims['sub']}")
    err(f"  Model scope: {claims['model']}")
    err(f"  Valid days:  {args.days}")
    err(f"  Expires:     {exp_h}")
    if args.secret:
        err(f"  {YELLOW}Custom secret used — server must run with matching JINA_LICENSE_SECRET.{RESET}")
    err(f"\n{BOLD}Key (stdout):{RESET}")
    out(key)
    err(f"\n{BOLD}Use it:{RESET}")
    err(f"  docker run -e JINA_LICENSE_KEY={key} -p 8080:8080 <image>")
    err(f"\n{DIM}Default server mode is 'warn' (fail-open): the key is a visible expiry")
    err(f"signal and NEVER blocks a running deployment. Use JINA_LICENSE_MODE=enforce")
    err(f"only for time-boxed trials/POCs. Renew = mint a new key + restart. No rebuild.{RESET}")
    sys.exit(EXIT_OK)


def _has_cuda() -> bool:
    try:
        result = subprocess.run(
            ["python3", "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() == "True"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Interactive helpers (TTY only)
# ---------------------------------------------------------------------------

def _select_model_interactive(models):
    err(f"\n{BOLD}Available Models:{RESET}\n")
    for i, m in enumerate(models, 1):
        mod = m.get("modality", "text")
        badge = {"multimodal": f"{CYAN}[OMNI]{RESET}", "code": f"{YELLOW}[CODE]{RESET}"}.get(mod, f"{GREEN}[TEXT]{RESET}")
        err(f"  {DIM}{str(i).rjust(2)}.{RESET} {badge} {BOLD}{m['id']}{RESET}  {DIM}{m.get('release','')}{RESET}")
        err(f"       {m['description']}")
    err("")
    while True:
        try:
            choice = input(f"{BOLD}Select model (1-{len(models)}): {RESET}").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            err(f"  {RED}Invalid. Enter 1-{len(models)}{RESET}")
        except ValueError:
            err(f"  {RED}Enter a number{RESET}")
        except (EOFError, KeyboardInterrupt):
            err("\nInterrupted.", YELLOW)
            sys.exit(EXIT_INTERRUPTED)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser(
        prog="python jina-on-prem.py",
        description="On-Prem Deployment Toolkit for Jina AI Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    parser.add_argument("--version", action="version", version=f"jina-on-prem {VERSION}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # --- list ---
    list_p = subparsers.add_parser(
        "list",
        help="List available models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="List all available Jina AI models.",
        epilog=textwrap.dedent("""\
            Examples:
              python jina-on-prem.py list
              python jina-on-prem.py list -v
              python jina-on-prem.py list --type embedding
              python jina-on-prem.py list --modality multimodal
              python jina-on-prem.py list --json
              python jina-on-prem.py list --json --type reranker | python3 -m json.tool
        """),
    )
    list_p.add_argument("-v", "--verbose", action="store_true",
                        help="Show VRAM, dims, context, license, modality")
    list_p.add_argument("--json", action="store_true",
                        help="Output as JSON to stdout (machine-readable)")
    list_p.add_argument("--type", choices=["embedding", "reranker", "reader", "colbert", "vlm"],
                        help="Filter by model type")
    list_p.add_argument("--modality", choices=["text", "multimodal", "code"],
                        help="Filter by modality")

    # --- bundle ---
    bundle_p = subparsers.add_parser(
        "bundle",
        help="[Phase 1 - network] Build and save Docker image bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Phase 1: BUNDLE (requires internet)

            Downloads model weights and bakes them into a self-contained Docker image.
            The output .tar.gz can be transferred to an air-gapped machine and deployed
            with 'deploy' - no network access needed at runtime.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python jina-on-prem.py bundle --model jina-embeddings-v5-text-nano
              python jina-on-prem.py bundle --model jina-embeddings-v5-text-small --output small.tar.gz
              python jina-on-prem.py bundle --model jina-embeddings-v5-text-nano --cpu-only
              python jina-on-prem.py bundle --model jina-reranker-v2-base-multilingual --hf-token TOKEN
              python jina-on-prem.py bundle --dry-run --model jina-embeddings-v5-text-nano
              python jina-on-prem.py bundle --model jina-embeddings-v5-text-nano --json
        """),
    )
    bundle_p.add_argument("--model", metavar="MODEL_ID",
                          help="Model ID (run 'list' to see all)")
    bundle_p.add_argument("--output", "-o", metavar="FILE",
                          help="Output .tar.gz path (default: jina-<model>-<runtime>.tar.gz)")
    bundle_p.add_argument("--hf-token", dest="hf_token", metavar="TOKEN",
                          help="HuggingFace token for gated models")
    bundle_p.add_argument("--cpu-only", action="store_true",
                          help="Build CPU-only image (no GPU required)")
    bundle_p.add_argument("-y", "--yes", action="store_true",
                          help="Skip confirmation prompt")
    bundle_p.add_argument("--dry-run", action="store_true",
                          help="Show what would be built without building")
    bundle_p.add_argument("--json", action="store_true",
                          help="Output bundle metadata as JSON to stdout when done")

    # --- deploy ---
    deploy_p = subparsers.add_parser(
        "deploy",
        help="[Phase 2 - offline] Load saved bundle and start container",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Phase 2: DEPLOY (no network required)

            Loads a .tar.gz bundle created by 'bundle' and starts the container.
            Everything is self-contained - zero internet access needed.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python jina-on-prem.py deploy --image jina-v5-nano-gpu.tar.gz --gpu
              python jina-on-prem.py deploy --image jina-v5-nano-cpu.tar.gz --port 9090
              python jina-on-prem.py deploy --image jina-v5-nano-gpu.tar.gz --gpu --detach
              python jina-on-prem.py deploy --image bundle.tar.gz --name my-embedder --detach
        """),
    )
    deploy_p.add_argument("--image", "-i", metavar="FILE", required=True,
                          help="Path to .tar.gz bundle file")
    deploy_p.add_argument("--port", "-p", type=int, default=8080,
                          help="Host port to expose the service (default: 8080)")
    deploy_p.add_argument("--gpu", action="store_true",
                          help="Enable GPU passthrough (--gpus all)")
    deploy_p.add_argument("--detach", "-d", action="store_true",
                          help="Run container in background, print container ID")
    deploy_p.add_argument("--name", metavar="NAME",
                          help="Assign a name to the container")

    # --- serve ---
    serve_p = subparsers.add_parser(
        "serve",
        help="Serve model directly (no Docker, deps must be installed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Serve a model directly without Docker.

            Requires model dependencies to be installed in the current Python environment.
            Use 'list -v' to see required dependencies for each model.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python jina-on-prem.py serve --model jina-embeddings-v5-text-nano
              python jina-on-prem.py serve --model jina-embeddings-v5-text-nano --device cuda
              python jina-on-prem.py serve --local-path /data/models/my-model
              python jina-on-prem.py serve --model jina-embeddings-v5-text-nano --port 9090
        """),
    )
    serve_p.add_argument("--model", metavar="MODEL_ID",
                         help="Model ID from catalog, or HuggingFace repo ID")
    serve_p.add_argument("--local-path", dest="local_path", metavar="PATH",
                         help="Local directory path to model files")
    serve_p.add_argument("--port", "-p", type=int, default=8080,
                         help="Port to listen on (default: 8080)")
    serve_p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                         help="Compute device: cpu, cuda, or auto-detect (default: auto)")

    # --- keygen ---
    keygen_p = subparsers.add_parser(
        "keygen",
        help="Mint a time-sensitive license key (offline, no rebuild)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Mint a time-sensitive license key.

            The key is a signed, self-contained token carrying an expiry. It is
            injected at run time (docker run -e JINA_LICENSE_KEY=...), so issuing
            or renewing never requires rebuilding the image. This is a symbolic
            compliance gate, not DRM (see server/license.py).
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python jina-on-prem.py keygen --sub acme-corp --days 30
              python jina-on-prem.py keygen --sub acme --days 90 --model jina-embeddings-v5-text-nano
              python jina-on-prem.py keygen --sub trial --days 7 --json
        """),
    )
    keygen_p.add_argument("--sub", required=True, metavar="NAME",
                          help="Who the license is issued to (customer/PM name)")
    keygen_p.add_argument("--days", type=int, default=30,
                          help="Validity window in days (default: 30)")
    keygen_p.add_argument("--model", metavar="MODEL_ID",
                          help="Restrict key to one model id (default: * = any)")
    keygen_p.add_argument("--secret", metavar="SECRET",
                          help="Signing secret override (must match server JINA_LICENSE_SECRET)")
    keygen_p.add_argument("--json", action="store_true",
                          help="Output key + claims as JSON to stdout")

    # Backward-compat hidden aliases
    for alias, real in [("pack", "bundle"), ("load", "deploy")]:
        alias_p = subparsers.add_parser(alias, help=argparse.SUPPRESS)
        if real == "bundle":
            alias_p.add_argument("--model")
            alias_p.add_argument("--output", "-o")
            alias_p.add_argument("--hf-token", dest="hf_token")
            alias_p.add_argument("--cpu-only", action="store_true")
            alias_p.add_argument("-y", "--yes", action="store_true")
            alias_p.add_argument("--dry-run", action="store_true")
            alias_p.add_argument("--json", action="store_true")
        else:
            alias_p.add_argument("--image", "-i", required=True)
            alias_p.add_argument("--port", "-p", type=int, default=8080)
            alias_p.add_argument("--gpu", action="store_true")
            alias_p.add_argument("--detach", "-d", action="store_true")
            alias_p.add_argument("--name")

    return parser


# ---------------------------------------------------------------------------
# SIGINT handler
# ---------------------------------------------------------------------------

def _sigint_handler(sig, frame):
    err("\nInterrupted.", YELLOW)
    sys.exit(EXIT_INTERRUPTED)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    signal.signal(signal.SIGINT, _sigint_handler)

    parser = make_parser()

    # Layer 0: no args -> brief usage
    if len(sys.argv) == 1:
        print_banner()
        err(f"{BOLD}Commands:{RESET}\n")
        err(f"  {BOLD}list{RESET}    List available models (28 total)")
        err(f"  {BOLD}bundle{RESET}  [Phase 1 - network]  Build + save Docker image bundle")
        err(f"  {BOLD}deploy{RESET}  [Phase 2 - offline]  Load bundle and start container")
        err(f"  {BOLD}serve{RESET}   Serve model directly (no Docker, deps required)")
        err(f"  {BOLD}keygen{RESET}  Mint a time-sensitive license key (offline, no rebuild)")
        err("")
        err(f"{BOLD}Quick start:{RESET}")
        err("  python jina-on-prem.py list")
        err("  python jina-on-prem.py bundle --model jina-embeddings-v5-text-nano")
        err("  python jina-on-prem.py deploy --image jina-embeddings-v5-text-nano-gpu.tar.gz --gpu")
        err("")
        err(f"{DIM}Run 'python jina-on-prem.py <command> --help' for command details.{RESET}")
        sys.exit(EXIT_OK)

    args = parser.parse_args()

    if args.command in ("bundle", "pack"):
        cmd_bundle(args)
    elif args.command == "keygen":
        cmd_keygen(args)
    elif args.command in ("deploy", "load"):
        cmd_deploy(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(EXIT_USER_ERROR)


if __name__ == "__main__":
    main()
