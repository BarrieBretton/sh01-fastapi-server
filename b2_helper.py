"""
Backblaze B2 Helper Module
Handles file uploads to Backblaze B2 storage.
Requires:
- BACKBLAZE_KEY_ID
- BACKBLAZE_APPLICATION_KEY
- BACKBLAZE_BUCKET_NAME (optional, can be passed per call)
"""
import os
import logging
from pathlib import Path
from typing import Optional
from b2sdk.v2 import B2Api, InMemoryAccountInfo

logger = logging.getLogger(__name__)

class B2Manager:
    """Manages Backblaze B2 uploads and operations."""
    
    def __init__(self, key_id: str = None, app_key: str = None, bucket_name: str = None):
        self.key_id = key_id or os.getenv("BACKBLAZE_KEY_ID")
        self.app_key = app_key or os.getenv("BACKBLAZE_APPLICATION_KEY")
        self.default_bucket_name = bucket_name or os.getenv("BACKBLAZE_BUCKET_NAME")
        
        if not self.key_id or not self.app_key:
            raise ValueError(
                "B2 credentials missing. Set BACKBLAZE_KEY_ID and "
                "BACKBLAZE_APPLICATION_KEY environment variables."
            )
        
        self.info = InMemoryAccountInfo()
        self.b2_api = B2Api(self.info)
        self._authorized = False
        self._download_url = None  # We'll cache the proper download base URL
    
    def _ensure_authorized(self):
        """Ensure we're authorized and cache the download URL."""
        if not self._authorized:
            try:
                self.b2_api.authorize_account("production", self.key_id, self.app_key)
                self._authorized = True
                # This is the correct base for downloads (e.g. https://f005.backblazeb2.com)
                self._download_url = self.info.get_download_url()
                logger.info("B2 authorization successful. Download base URL: %s", self._download_url)
            except Exception as e:
                logger.error("B2 authorization failed: %s", e)
                raise RuntimeError(f"Failed to authorize with Backblaze B2: {e}")
    
    def upload_file(
        self,
        file_path: Path,
        bucket_name: str = None,
        file_name: str = None,
        content_type: str = None,
        expiration_days: int = 7,
    ) -> str:
        """
        Upload a file to B2 and return a temporary authorized download URL (works on private buckets).
        
        Args:
            file_path: Path to file to upload
            bucket_name: B2 bucket name (uses default if not provided)
            file_name: Remote file name (uses local filename if not provided)
            content_type: MIME type (auto-detected if not provided)
            expiration_days: How long the returned URL is valid (max 7 days)
        
        Returns:
            Temporary authorized URL that anyone can use to download the file
        """
        self._ensure_authorized()
        
        bucket_name = bucket_name or self.default_bucket_name
        if not bucket_name:
            raise ValueError("bucket_name required (or set BACKBLAZE_BUCKET_NAME env var)")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_name:
            file_name = file_path.name
        
        # Auto-detect content type
        if not content_type:
            ext = file_path.suffix.lower()
            content_type_map = {
                '.mp4': 'video/mp4',
                '.mov': 'video/quicktime',
                '.avi': 'video/x-msvideo',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
            }
            content_type = content_type_map.get(ext, 'application/octet-stream')
        
        try:
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            
            logger.info(
                "Uploading %s to B2 bucket '%s' as '%s' (%.2f MB)",
                file_path.name, bucket_name, file_name,
                file_path.stat().st_size / 1_048_576
            )
            
            # Upload (no public flag needed — we're using auth token)
            bucket.upload_local_file(
                local_file=str(file_path),
                file_name=file_name,
                content_type=content_type,
                file_infos={'b2-content-disposition': 'inline'},  # Optional: helps browsers display inline
            )
            
            # Generate temporary download authorization token for this exact file
            valid_seconds = min(expiration_days * 86400, 604800)  # Max 7 days
            auth_token = bucket.get_download_authorization(
                file_name_prefix=file_name,        # Exact file name acts as prefix → only this file
                valid_duration_in_seconds=valid_seconds,
            )
            
            # Build full authorized URL using the correct download base
            authorized_url = f"{self._download_url}/file/{bucket_name}/{file_name}?Authorization={auth_token}"
            
            logger.info("B2 upload successful. Authorized URL (valid %d days): %s", expiration_days, authorized_url)
            return authorized_url
            
        except Exception as e:
            logger.exception("B2 upload failed")
            raise RuntimeError(f"Failed to upload to B2: {e}")

    def delete_file(
        self,
        file_name: str,
        bucket_name: str = None
    ) -> bool:
        """
        Permanently delete ALL versions of a file by exact file name.
        """
        self._ensure_authorized()
    
        bucket_name = bucket_name or self.default_bucket_name
        if not bucket_name:
            raise ValueError("bucket_name required")
    
        try:
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            
            # Extract folder path from file_name to use as prefix
            # e.g., "videos/file.mp4" -> folder="videos/"
            if "/" in file_name:
                folder_path = "/".join(file_name.split("/")[:-1]) + "/"
            else:
                folder_path = ""
            
            # List files with folder prefix, then filter for exact match
            file_versions = []
            for file_version_info, folder_name in bucket.ls(
                folder_to_list=folder_path,
                latest_only=False,
                recursive=True
            ):
                # Check for exact file name match
                if file_version_info.file_name == file_name:
                    file_versions.append(file_version_info)
            
            if not file_versions:
                logger.warning("File not found in B2: %s", file_name)
                return False
            
            # Permanently delete each version
            for version_info in file_versions:
                self.b2_api.delete_file_version(
                    file_id=version_info.id_,
                    file_name=version_info.file_name
                )
                logger.info("Permanently deleted B2 file version: %s (id: %s)", 
                        file_name, version_info.id_)
            
            logger.info("Deleted %d version(s) of file: %s", len(file_versions), file_name)
            return True
            
        except Exception as e:
            logger.exception("B2 delete failed")
            raise RuntimeError(f"Failed to delete from B2: {e}")

# Global B2 manager instance
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
    expiration_days: int = 7
) -> str:
    """
    Upload a video to B2 and return a temporary authorized URL.
    """
    manager = get_b2_manager()
   
    from datetime import datetime
    timestamp = int(datetime.utcnow().timestamp())
    file_name = f"videos/{timestamp}_{video_path.name}"
   
    return manager.upload_file(
        file_path=video_path,
        bucket_name=bucket_name,
        file_name=file_name,
        content_type="video/mp4",
        expiration_days=expiration_days
    )
