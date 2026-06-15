"""Object storage abstraction (horizontal foundation).

A local-filesystem stand-in for **S3 / GCS** with the same `put`/`get`/`list`
interface — so corpus + model artifacts move identically across clouds. In
production, swap in an `S3ObjectStore` (boto3) or `GcsObjectStore`
(google-cloud-storage); the calling code does not change.
"""

import os


class LocalObjectStore:
    """Filesystem stand-in for an S3/GCS bucket (root = the 'bucket')."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        path = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def put(self, key: str, data: bytes) -> str:
        with open(self._path(key), "wb") as f:
            f.write(data)
        return key

    def get(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.exists(os.path.join(self.root, key))

    def list(self, prefix: str = "") -> list[str]:
        keys = []
        for dirpath, _, files in os.walk(self.root):
            for name in files:
                rel = os.path.relpath(os.path.join(dirpath, name), self.root)
                if rel.startswith(prefix):
                    keys.append(rel)
        return sorted(keys)


# Production note: `S3ObjectStore(bucket)` wraps boto3 (`put_object`/`get_object`/
# `list_objects_v2`); `GcsObjectStore(bucket)` wraps google-cloud-storage Blobs.
# Both implement the same put/get/exists/list surface as LocalObjectStore.
