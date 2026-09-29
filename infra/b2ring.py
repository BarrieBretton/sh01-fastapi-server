import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from b2sdk.v2 import B2Api, InMemoryAccountInfo

from .registry import registry

logger = logging.getLogger("infra.b2")


def _env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _slot(slot: str) -> dict[str, str]:
    cfg = registry.get("b2", slot)
    required = {"key_id_env", "application_key_env", "bucket_name_env"}
    missing = required - set(cfg)
    if missing:
        raise RuntimeError(
            f"B2 slot '{slot}' is missing registry fields: "
            + ", ".join(sorted(missing))
        )
    return {
        "key_id": _env_value(cfg["key_id_env"]),
        "application_key": _env_value(cfg["application_key_env"]),
        "bucket_name": _env_value(cfg["bucket_name_env"]),
    }


def b2_runtime_config(slot: str) -> dict[str, str]:
    return _slot(slot)


def _api(slot: str) -> tuple[B2Api, Any, dict[str, str]]:
    cfg = _slot(slot)
    info = InMemoryAccountInfo()
    api = B2Api(info)
    api.authorize_account("production", cfg["key_id"], cfg["application_key"])
    bucket = api.get_bucket_by_name(cfg["bucket_name"])
    return api, bucket, cfg


def b2_health(slot: str) -> dict[str, Any]:
    try:
        _, bucket, cfg = _api(slot)
        # Touch the listing endpoint without scanning the bucket.
        first = next(iter(bucket.ls(latest_only=True, recursive=True)), None)
        return {
            "slot": slot,
            "healthy": True,
            "bucket": cfg["bucket_name"],
            "sample_file": None if first is None else first[0].file_name,
        }
    except Exception as exc:
        return {
            "slot": slot,
            "healthy": False,
            "error": str(exc),
        }


def _size(version: Any) -> int:
    for attr in ("size", "content_length", "contentLength"):
        value = getattr(version, attr, None)
        if value is not None:
            return int(value)
    return 0


def _sha1(version: Any) -> str | None:
    value = getattr(version, "content_sha1", None)
    if not value:
        return None
    value = str(value)
    # Large files may report a non-verifiable marker such as "none".
    if value.lower() in {"none", "do_not_verify"}:
        return None
    return value


def b2_inventory(slot: str) -> dict[str, dict[str, Any]]:
    _, bucket, _ = _api(slot)
    inventory: dict[str, dict[str, Any]] = {}
    for version, _folder in bucket.ls(latest_only=True, recursive=True):
        inventory[version.file_name] = {
            "size": _size(version),
            "sha1": _sha1(version),
            "file_id": getattr(version, "id_", None),
            "content_type": getattr(version, "content_type", None),
            "file_info": dict(getattr(version, "file_info", None) or {}),
        }
    return inventory


def b2_compare(source_slot: str, destination_slot: str) -> dict[str, Any]:
    if source_slot == destination_slot:
        raise ValueError("Source and destination B2 slots must differ")

    source = b2_inventory(source_slot)
    destination = b2_inventory(destination_slot)

    source_names = set(source)
    destination_names = set(destination)
    missing = sorted(source_names - destination_names)
    extra = sorted(destination_names - source_names)
    mismatches: list[dict[str, Any]] = []

    for name in sorted(source_names & destination_names):
        s = source[name]
        d = destination[name]
        same_size = s["size"] == d["size"]
        sha_check_available = bool(s.get("sha1") and d.get("sha1"))
        same_sha = (s.get("sha1") == d.get("sha1")) if sha_check_available else True
        if not same_size or not same_sha:
            mismatches.append({"file": name, "source": s, "destination": d})

    source_match = not missing and not mismatches
    match = source_match and not extra
    return {
        "match": match,
        "source_match": source_match,
        "source": source_slot,
        "destination": destination_slot,
        "source_file_count": len(source),
        "destination_file_count": len(destination),
        "missing_on_destination": missing,
        "extra_on_destination": extra,
        "file_mismatches": mismatches,
    }


def _delete_all_versions(api: B2Api, bucket: Any, file_name: str) -> int:
    deleted = 0
    for version, _folder in bucket.ls(
        folder_to_list=(file_name.rsplit("/", 1)[0] + "/") if "/" in file_name else "",
        latest_only=False,
        recursive=True,
    ):
        if version.file_name == file_name:
            api.delete_file_version(version.id_, version.file_name)
            deleted += 1
    return deleted


def b2_migrate(
    source_slot: str,
    destination_slot: str,
    prune_extra: bool = False,
) -> dict[str, Any]:
    if source_slot == destination_slot:
        raise ValueError("Source and destination B2 slots must differ")

    source_api, source_bucket, _ = _api(source_slot)
    destination_api, destination_bucket, _ = _api(destination_slot)
    source_inventory = b2_inventory(source_slot)
    destination_inventory = b2_inventory(destination_slot)

    copied: list[str] = []
    skipped: list[str] = []
    deleted_extra: list[str] = []

    with tempfile.TemporaryDirectory(prefix="b2-ring-migration-") as tmpdir:
        local_path = Path(tmpdir) / "payload.bin"

        for name in sorted(source_inventory):
            src = source_inventory[name]
            dst = destination_inventory.get(name)
            if dst:
                same_size = src["size"] == dst["size"]
                sha_available = bool(src.get("sha1") and dst.get("sha1"))
                same_sha = (src.get("sha1") == dst.get("sha1")) if sha_available else True
                if same_size and same_sha:
                    skipped.append(name)
                    continue

            logger.info("B2 mirror downloading %s from %s", name, source_slot)
            if local_path.exists():
                local_path.unlink()
            source_bucket.download_file_by_name(name).save_to(local_path)

            # Verify local bytes whenever B2 provides a verifiable SHA-1.
            expected_sha1 = src.get("sha1")
            if expected_sha1:
                sha1 = hashlib.sha1()
                with local_path.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        sha1.update(chunk)
                if sha1.hexdigest() != expected_sha1:
                    raise RuntimeError(f"B2 download checksum mismatch for {name}")

            logger.info("B2 mirror uploading %s to %s", name, destination_slot)
            destination_bucket.upload_local_file(
                local_file=str(local_path),
                file_name=name,
                content_type=src.get("content_type"),
                file_infos=src.get("file_info") or None,
            )
            copied.append(name)

        if prune_extra:
            extras = sorted(set(destination_inventory) - set(source_inventory))
            for name in extras:
                _delete_all_versions(destination_api, destination_bucket, name)
                deleted_extra.append(name)

    comparison = b2_compare(source_slot, destination_slot)
    if not comparison["source_match"]:
        raise RuntimeError(
            "B2 migration completed but inventory comparison failed: "
            f"missing={len(comparison['missing_on_destination'])} "
            f"extra={len(comparison['extra_on_destination'])} "
            f"mismatches={len(comparison['file_mismatches'])}"
        )

    if prune_extra and comparison["extra_on_destination"]:
        raise RuntimeError(
            "B2 migration completed but destination still contains extra files"
        )

    return {
        "source": source_slot,
        "destination": destination_slot,
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "deleted_extra_count": len(deleted_extra),
        "copied": copied,
        "deleted_extra": deleted_extra,
        "comparison": comparison,
    }
