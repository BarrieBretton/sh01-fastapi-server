# b2_helper.py
import os
import logging
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from b2sdk.v2 import B2Api, InMemoryAccountInfo

logger = logging.getLogger(__name__)

class B2Manager:
    """
    Backblaze B2 manager with:
    - automatic re-authorization on expired/bad auth tokens
    - thread-safe authorization
    - helpers to mint download authorization tokens on demand
    """

    def __init__(self, key_id: str = None, app_key: str = None, bucket_name: str = None):
        self.key_id = key_id or os.getenv("BACKBLAZE_KEY_ID")
        self.app_key = app_key or os.getenv("BACKBLAZE_APPLICATION_KEY")
        self.default_bucket_name = bucket_name or os.getenv("BACKBLAZE_BUCKET_NAME")

        if not self.key_id or not self.app_key:
            raise ValueError("Missing BACKBLAZE_KEY_ID / BACKBLAZE_APPLICATION_KEY")

        self.info = InMemoryAccountInfo()
        self.b2_api = B2Api(self.info)

        self._download_url: Optional[str] = None
        self._auth_lock = threading.RLock()

        # do not rely on a permanent boolean; tokens can expire
        self._authorized_once = False

    def _authorize_account(self) -> None:
        """Authorize (or re-authorize) the B2 account and cache download base URL."""
        self.b2_api.authorize_account("production", self.key_id, self.app_key)
        self._download_url = self.info.get_download_url()
        self._authorized_once = True
        logger.info("B2 account authorized. Download base URL: %s", self._download_url)

    def _ensure_authorized(self) -> None:
        """Authorize at least once."""
        with self._auth_lock:
            if not self._authorized_once or not self._download_url:
                self._authorize_account()

    @staticmethod
    def _looks_like_auth_error(exc: Exception) -> bool:
        s = str(exc).lower()
        # b2sdk exceptions vary; this catches common signals
        return ("expired_auth_token" in s) or ("bad_auth_token" in s) or ("unauthorized" in s)

    def _with_reauth_retry(self, fn, *args, **kwargs):
        """
        Run a B2 operation; if it fails due to auth token expiry, reauth and retry once.
        """
        self._ensure_authorized()

        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if self._looks_like_auth_error(e):
                logger.warning("B2 auth error detected; re-authorizing and retrying once. err=%s", e)
                with self._auth_lock:
                    self._authorize_account()
                return fn(*args, **kwargs)
            raise

    def get_bucket(self, bucket_name: Optional[str] = None):
        bucket_name = bucket_name or self.default_bucket_name
        if not bucket_name:
            raise ValueError("bucket_name required (or set BACKBLAZE_BUCKET_NAME)")
        return self._with_reauth_retry(self.b2_api.get_bucket_by_name, bucket_name)

    def upload_file(
        self,
        file_path: Path,
        bucket_name: str = None,
        file_name: str = None,
        content_type: str = None,
        expiration_days: int = 7,
    ) -> str:
        """
        Upload and return an authorized download URL (token in query string).
        NOTE: this URL will expire and must be refreshed by minting a new token.
        """
        bucket = self.get_bucket(bucket_name=bucket_name)
        bucket_name = bucket_name or self.default_bucket_name

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_name:
            file_name = file_path.name

        if not content_type:
            ext = file_path.suffix.lower()
            content_type_map = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
            }
            content_type = content_type_map.get(ext, "application/octet-stream")

        logger.info(
            "Uploading %s to B2 bucket '%s' as '%s' (%.2f MB)",
            file_path.name,
            bucket_name,
            file_name,
            file_path.stat().st_size / 1_048_576,
        )

        def _do_upload():
            bucket.upload_local_file(
                local_file=str(file_path),
                file_name=file_name,
                content_type=content_type,
                file_infos={"b2-content-disposition": "inline"},
            )

        self._with_reauth_retry(_do_upload)

        # mint token after upload
        valid_seconds = min(expiration_days * 86400, 604800)  # max 7 days
        token = self.get_download_authorization_token(
            file_name=file_name,
            bucket_name=bucket_name,
            valid_seconds=valid_seconds,
        )
        url = self.build_authorized_url(bucket_name=bucket_name, file_name=file_name, token=token)
        return url

    def get_download_authorization_token(
        self,
        file_name: str,
        bucket_name: str = None,
        valid_seconds: int = 3600,
    ) -> str:
        """
        Mint a fresh download authorization token for a specific file.
        """
        if valid_seconds < 1 or valid_seconds > 604800:
            raise ValueError("valid_seconds must be in [1, 604800]")

        bucket = self.get_bucket(bucket_name=bucket_name)
        file_prefix = file_name  # exact match prefix

        def _do_token():
            return bucket.get_download_authorization(
                file_name_prefix=file_prefix,
                valid_duration_in_seconds=valid_seconds,
            )

        return self._with_reauth_retry(_do_token)

    def build_authorized_url(self, bucket_name: str, file_name: str, token: str) -> str:
        """
        Construct /file/ URL with proper encoding (keeps slashes).
        """
        self._ensure_authorized()
        safe_file = "/".join(quote(part) for part in file_name.split("/"))
        safe_bucket = quote(bucket_name)
        return f"{self._download_url}/file/{safe_bucket}/{safe_file}?Authorization={token}"

    def delete_file(self, file_name: str, bucket_name: str = None) -> bool:
        """
        Delete all versions of a file.
        """
        bucket = self.get_bucket(bucket_name=bucket_name)
        bucket_name = bucket_name or self.default_bucket_name

        # (Your delete logic is okay; wrap calls in _with_reauth_retry where B2 calls occur.)
        def _do_delete():
            # list versions by folder and delete exact matches
            if "/" in file_name:
                folder_path = "/".join(file_name.split("/")[:-1]) + "/"
            else:
                folder_path = ""

            file_versions = []
            for file_version_info, folder_name in bucket.ls(
                folder_to_list=folder_path,
                latest_only=False,
                recursive=True,
            ):
                if file_version_info.file_name == file_name:
                    file_versions.append(file_version_info)

            if not file_versions:
                return False

            for v in file_versions:
                self.b2_api.delete_file_version(file_id=v.id_, file_name=v.file_name)
            return True

        return self._with_reauth_retry(_do_delete)


_b2_manager: Optional[B2Manager] = None

def get_b2_manager() -> B2Manager:
    global _b2_manager
    if _b2_manager is None:
        _b2_manager = B2Manager()
    return _b2_manager


def upload_to_b2(
    video_path: Path,
    yt_link: str = "",
    bucket_name: str = None,
    expiration_days: int = 7,
) -> str:
    manager = get_b2_manager()
    from datetime import datetime
    timestamp = int(datetime.utcnow().timestamp())
    file_name = f"videos/{timestamp}_{video_path.name}"
    return manager.upload_file(
        file_path=video_path,
        bucket_name=bucket_name,
        file_name=file_name,
        content_type="video/mp4",
        expiration_days=expiration_days,
    )
