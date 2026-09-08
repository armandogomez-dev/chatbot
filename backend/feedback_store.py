import json
import os
import time
from io import BytesIO
from uuid import uuid4

from huggingface_hub import HfApi

# Cada evaluación de un psicólogo se guarda como un archivo JSON separado en un
# dataset privado de Hugging Face Hub (no en disco local: el filesystem de Railway
# es efímero y se pierde en cada redeploy/restart).
FEEDBACK_REPO = os.getenv("FEEDBACK_REPO", "armando24/chatbox-evaluaciones")

_api = HfApi()


def save_feedback(entry: dict) -> str:
    entry_id = str(uuid4())
    payload = {**entry, "id": entry_id, "timestamp": time.time()}
    _api.upload_file(
        path_or_fileobj=BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        path_in_repo=f"submissions/{entry_id}.json",
        repo_id=FEEDBACK_REPO,
        repo_type="dataset",
    )
    return entry_id


def list_feedback() -> list[dict]:
    files = [
        f for f in _api.list_repo_files(repo_id=FEEDBACK_REPO, repo_type="dataset")
        if f.startswith("submissions/") and f.endswith(".json")
    ]
    entries = []
    for f in files:
        path = _api.hf_hub_download(repo_id=FEEDBACK_REPO, repo_type="dataset", filename=f)
        with open(path, "r", encoding="utf-8") as fh:
            entries.append(json.load(fh))
    entries.sort(key=lambda e: e.get("timestamp", 0))
    return entries
