# delete_old_mlflow.py
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
import shutil
import os
import time
import argparse


def ms_from_days(days):
    return int(days * 24 * 3600 * 1000)


def delete_local_artifacts_if_exists(artifact_uri):
    # only handle file:// URIs (local paths)
    if artifact_uri is None:
        return
    p = artifact_uri
    if p.startswith("file://"):
        p = p[len("file://"):]
    # safety: require that path contains "mlruns" or another expected substring
    if "mlruns" not in p and "/artifacts" not in p:
        print("WARN: artifact path looks unexpected, skipping delete:", p)
        return
    if os.path.exists(p):
        print("Removing local artifact dir:", p)
        shutil.rmtree(p)
    else:
        print("Artifact path not found (skipping):", p)


def main(tracking_uri=None, cutoff_days=365, dry_run=True, remove_artifacts=False):
    client = MlflowClient(tracking_uri) if tracking_uri else MlflowClient()
    experiment_id = "914879025403898454"

    runs = client.search_runs(experiment_ids=[experiment_id])
    cutoff_ms = int(time.time() * 1000) - ms_from_days(cutoff_days)
    print("Cutoff (ms):", cutoff_ms)

    runs = client.search_runs([experiment_id])
    print("Total runs: ", len(runs))
    to_delete = []
    for r in runs:
        run_id = r.info.run_id
        start_time = getattr(r.info, "start_time", None)  # ms
        if start_time is None:
            continue

        if start_time < cutoff_ms:
            # try to get a human-friendly run name from tags
            run_name = "<no-name>"
            try:
                tags = getattr(r, "data", None) and getattr(
                    r.data, "tags", None)
                if tags and isinstance(tags, dict):
                    run_name = tags.get("mlflow.runName") or tags.get(
                        "run_name") or run_name
            except Exception:
                pass

            to_delete.append((run_id, run_name, r.info.artifact_uri))

    print(
        f"  Found {len(to_delete)} runs to delete in experiment {experiment_id}")

    for run_id, run_name, artifact_uri in to_delete:
        print("  Deleting run:", run_name)
        if dry_run:
            print(
                "    (dry-run) would delete run and optionally artifacts:", artifact_uri)
        else:
            client.delete_run(run_id)  # soft delete
            if remove_artifacts:
                delete_local_artifacts_if_exists(artifact_uri)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None,
                        help="mlflow tracking uri (optional)")
    parser.add_argument("--cutoff-days", type=int, default=365,
                        help="delete runs older than N days")
    parser.add_argument("--dry-run", action="store_true",
                        default=False, help="do not actually delete runs")
    parser.add_argument("--remove-artifacts", action="store_true", default=False,
                        help="try to remove local artifact dirs (only file://)")
    args = parser.parse_args()
    main(args.tracking_uri, args.cutoff_days, dry_run=args.dry_run,
         remove_artifacts=args.remove_artifacts)
