"""Provision and monitor 5 RunPod H100 pods for parallel training.

Uses the RunPod Python SDK (v1.9.0+) to:
1. Create 5 pods with H100 SXM GPUs (fallback to H100 PCIe, then A100 80GB)
2. Monitor training completion via WandB artifact checks
3. Download results when complete
4. Terminate pods automatically

Usage:
    export RUNPOD_API_KEY=$(pass show runpod/api-key)
    python scripts/provision_runpod.py [--dry-run] [--tasks gru_retrain,iter5]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import runpod
except ImportError:
    print("ERROR: runpod SDK not installed. Run: pip install runpod")
    sys.exit(1)

TASKS_CORE = ["gru_retrain", "iter5", "iter6", "distilbert_hier", "distilbert_concat"]
TASKS_ABLATIONS = [
    "ablation_shuffled", "ablation_reversed", "ablation_prefix",
    "ablation_continuation", "ablation_autoencoder",
    "ablation_mean_pool", "ablation_max_pool",
]
TASKS = TASKS_CORE + TASKS_ABLATIONS
GPU_PREFERENCES = [
    "NVIDIA L40S",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA L40",
    "NVIDIA RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
VOLUME_SIZE = 50
CONTAINER_DISK = 20
PROJECT = "rockcyber/multiturn-injection-detection-v2"
REPO_URL = "git@github.com:rocklambros/multiturn-injection-detection.git"
REPO_BRANCH = "feature/v3-clean-retrain"
MAX_WAIT_SECONDS = 7200
POLL_INTERVAL = 45


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_wandb_key():
    for path in ["~/.wandb_key", "/root/.wandb_key"]:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            with open(expanded) as f:
                key = f.read().strip()
                if key:
                    return key
    key = os.environ.get("WANDB_API_KEY", "")
    if key:
        return key
    try:
        result = subprocess.run(
            ["pass", "show", "wandb/api-key"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_deploy_key():
    import base64
    key_path = "/tmp/runpod_deploy_key"
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def build_startup_command(task):
    # Wrap in bash -c so shell operators (&, &&, ;) are interpreted.
    # nvidia_entrypoint.sh uses exec "$@" which passes args without
    # shell interpretation; bash -c forces a shell context.
    inner = (
        "/start.sh & "
        "sleep 5 && "
        "echo $WANDB_API_KEY > /root/.wandb_key && "
        "mkdir -p /root/.ssh && "
        "echo $DEPLOY_KEY_B64 | base64 -d > /root/.ssh/id_ed25519 && "
        "chmod 600 /root/.ssh/id_ed25519 && "
        "ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null && "
        f"git clone -b {REPO_BRANCH} {REPO_URL} /workspace/mt && "
        "cd /workspace/mt && "
        "pip install -q -r requirements.txt 2>&1 | tail -1 && "
        f"bash scripts/bootstrap_runpod.sh {task} 2>&1 | tee /workspace/{task}.log; "
        "sleep infinity"
    )
    return f"bash -c '{inner}'"


def create_pod(task, wandb_key, deploy_key, gpu_type, dry_run=False):
    startup_cmd = build_startup_command(task)
    pod_name = f"train-{task}"

    if dry_run:
        log(f"  [DRY RUN] Would create pod '{pod_name}' with {gpu_type}")
        return {"id": f"dry-run-{task}", "name": pod_name}

    pod = runpod.create_pod(
        name=pod_name,
        image_name=IMAGE,
        gpu_type_id=gpu_type,
        gpu_count=1,
        volume_in_gb=VOLUME_SIZE,
        container_disk_in_gb=CONTAINER_DISK,
        docker_args=startup_cmd,
        ports="22/tcp",
        env={
            "WANDB_API_KEY": wandb_key,
            "DEPLOY_KEY_B64": deploy_key,
        },
        min_download=500,
    )
    return pod


def provision_all(tasks, wandb_key, deploy_key, dry_run=False):
    pods = {}
    for task in tasks:
        last_error = None
        for gpu_type in GPU_PREFERENCES:
            try:
                log(f"  Creating pod for {task} on {gpu_type}...")
                pod = create_pod(task, wandb_key, deploy_key, gpu_type, dry_run)
                pod_id = pod.get("id", "unknown")
                pods[task] = {
                    "id": pod_id,
                    "gpu": gpu_type,
                    "status": "PROVISIONING",
                    "start_time": time.time(),
                }
                log(f"  Pod {pod_id} created for {task} ({gpu_type})")
                break
            except Exception as e:
                last_error = str(e)
                log(f"  WARN: {gpu_type} failed for {task}: {last_error}")
                continue
        else:
            log(f"  FATAL: Could not create pod for {task}. Last error: {last_error}")
            pods[task] = {"id": None, "status": "FAILED", "error": last_error}
    return pods


ARTIFACT_PREFIX = "v3_"


def check_artifact_exists(task):
    try:
        import wandb
        api = wandb.Api()
        api.artifact(f"{PROJECT}/{ARTIFACT_PREFIX}{task}_results:latest")
        return True
    except Exception:
        return False


def check_pod_status(pod_id):
    try:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime", {})
        if runtime:
            gpu_count = runtime.get("gpus", [])
            uptime = runtime.get("uptimeInSeconds", 0)
            return status, uptime
        return status, 0
    except Exception as e:
        return f"ERROR: {e}", 0


def monitor_pods(pods, dry_run=False):
    if dry_run:
        log("[DRY RUN] Skipping monitoring")
        return pods

    start = time.time()
    completed = set()
    failed = set()

    while len(completed) + len(failed) < len(pods):
        elapsed = time.time() - start

        for task, info in pods.items():
            if task in completed or task in failed:
                continue
            if info.get("id") is None:
                failed.add(task)
                continue

            pod_id = info["id"]
            task_elapsed = time.time() - info["start_time"]

            if check_artifact_exists(task):
                info["status"] = "DONE"
                info["duration"] = task_elapsed
                completed.add(task)
                log(f"  [{task}] COMPLETED in {task_elapsed:.0f}s — artifact found")
                try:
                    runpod.terminate_pod(pod_id)
                    log(f"  [{task}] Pod {pod_id} terminated")
                except Exception as e:
                    log(f"  [{task}] Failed to terminate pod: {e}")
                continue

            status, uptime = check_pod_status(pod_id)
            if "RUNNING" in str(status) and info["status"] == "PROVISIONING":
                info["status"] = "RUNNING"
                log(f"  [{task}] Pod is now RUNNING (uptime: {uptime}s)")

            if "EXITED" in str(status) or "TERMINATED" in str(status):
                if task not in completed:
                    info["status"] = "FAILED"
                    failed.add(task)
                    log(f"  [{task}] Pod terminated unexpectedly after {task_elapsed:.0f}s")
                continue

            if task_elapsed > MAX_WAIT_SECONDS:
                info["status"] = "TIMEOUT"
                failed.add(task)
                log(f"  [{task}] TIMEOUT after {task_elapsed:.0f}s")
                try:
                    runpod.terminate_pod(pod_id)
                except Exception:
                    pass
                continue

        running = sum(1 for t, i in pods.items() if t not in completed and t not in failed)
        if running > 0:
            log(f"Status: {len(completed)} done, {running} running, {len(failed)} failed — "
                f"next check in {POLL_INTERVAL}s (total elapsed: {elapsed:.0f}s)")
            time.sleep(POLL_INTERVAL)

    return pods


def download_results(pods):
    import wandb

    downloaded = []
    for task, info in pods.items():
        if info.get("status") != "DONE":
            log(f"  SKIP {task} (status: {info.get('status')})")
            continue
        try:
            api = wandb.Api()
            art = api.artifact(f"{PROJECT}/{ARTIFACT_PREFIX}{task}_results:latest")
            log(f"  Downloading {task} results (v{art.version}, {art.size / 1e6:.1f} MB)...")
            art.download(".")
            downloaded.append(task)
        except Exception as e:
            log(f"  WARN: Failed to download {task}: {e}")

    return downloaded


def terminate_all(pods):
    for task, info in pods.items():
        pod_id = info.get("id")
        if pod_id and info.get("status") not in ("DONE", "FAILED", "TIMEOUT"):
            try:
                runpod.terminate_pod(pod_id)
                log(f"  Terminated pod {pod_id} ({task})")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Provision RunPod training pods")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task list")
    parser.add_argument("--monitor-only", action="store_true", help="Skip provisioning, just monitor")
    parser.add_argument("--download-only", action="store_true", help="Skip provisioning, just download")
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else TASKS

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        try:
            result = subprocess.run(
                ["pass", "show", "runpod/api-key"],
                capture_output=True, text=True, timeout=10
            )
            api_key = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if not api_key:
        print("ERROR: No RunPod API key. Set RUNPOD_API_KEY or add to pass store.")
        sys.exit(1)

    runpod.api_key = api_key

    wandb_key = get_wandb_key()
    if not wandb_key:
        print("ERROR: No WandB API key found.")
        sys.exit(1)

    deploy_key = get_deploy_key()
    if not deploy_key:
        print("ERROR: No deploy key at /tmp/runpod_deploy_key.")
        sys.exit(1)

    log(f"=== RunPod {len(tasks)}-GPU Parallel Training ===")
    log(f"Tasks: {', '.join(tasks)}")
    log(f"GPU preference order: {', '.join(GPU_PREFERENCES)}")
    log(f"WandB key: {wandb_key[:8]}...")
    log(f"Deploy key: {deploy_key[:30]}...")
    log(f"Branch: {REPO_BRANCH}")
    log("")

    if args.download_only:
        fake_pods = {t: {"status": "DONE"} for t in tasks}
        downloaded = download_results(fake_pods)
        log(f"\nDownloaded {len(downloaded)}/{len(tasks)} results")
        return

    log("=== Step 1: Provisioning pods ===")
    pods = provision_all(tasks, wandb_key, deploy_key, dry_run=args.dry_run)

    created = sum(1 for p in pods.values() if p.get("id"))
    log(f"\n{created}/{len(tasks)} pods created")
    if created == 0:
        log("FATAL: No pods created. Exiting.")
        sys.exit(1)

    log("\n=== Step 2: Monitoring training ===")
    try:
        pods = monitor_pods(pods, dry_run=args.dry_run)
    except KeyboardInterrupt:
        log("\nInterrupted! Terminating all pods...")
        terminate_all(pods)
        sys.exit(1)

    log("\n=== Step 3: Downloading results ===")
    downloaded = download_results(pods)

    log("\n=== SUMMARY ===")
    for task, info in pods.items():
        status = info.get("status", "UNKNOWN")
        duration = info.get("duration", time.time() - info.get("start_time", time.time()))
        gpu = info.get("gpu", "?")
        symbol = "OK" if status == "DONE" else "FAIL"
        log(f"  [{symbol}] {task} — {status} in {duration:.0f}s ({gpu})")

    failed_count = sum(1 for i in pods.values() if i.get("status") != "DONE")
    if failed_count > 0:
        log(f"\nWARNING: {failed_count} tasks failed. Check WandB for partial results.")
        sys.exit(1)

    log(f"\nAll {len(tasks)} tasks completed. Results downloaded.")


if __name__ == "__main__":
    main()
