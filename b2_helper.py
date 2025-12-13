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
        """
        Initialize B2 manager.
        
        Args:
            key_id: B2 application key ID (or uses BACKBLAZE_KEY_ID env var)
            app_key: B2 application key (or uses BACKBLAZE_APPLICATION_KEY env var)
            bucket_name: Default bucket name (or uses BACKBLAZE_BUCKET_NAME env var)
        """
        self.key_id = key_id or os.getenv("BACKBLAZE_KEY_ID")
        self.app_key = app_key or os.getenv("BACKBLAZE_APPLICATION_KEY")
        self.default_bucket_name = bucket_name or os.getenv("BACKBLAZE_BUCKET_NAME")
        
        if not self.key_id or not self.app_key:
            raise ValueError(
                "B2 credentials missing. Set BACKBLAZE_KEY_ID and "
                "BACKBLAZE_APPLICATION_KEY environment variables."
            )
        
        # Initialize B2 API
        self.info = InMemoryAccountInfo()
        self.b2_api = B2Api(self.info)
        self._authorized = False
        
    def _ensure_authorized(self):
        """Ensure we're authorized with B2."""
        if not self._authorized:
            try:
                self.b2_api.authorize_account("production", self.key_id, self.app_key)
                self._authorized = True
                logger.info("B2 authorization successful")
            except Exception as e:
                logger.error("B2 authorization failed: %s", e)
                raise RuntimeError(f"Failed to authorize with Backblaze B2: {e}")
    
    def upload_file(
        self,
        file_path: Path,
        bucket_name: str = None,
        file_name: str = None,
        content_type: str = None,
        public: bool = True
    ) -> str:
        """
        Upload a file to B2 and return the public URL.
        
        Args:
            file_path: Path to file to upload
            bucket_name: B2 bucket name (uses default if not provided)
            file_name: Remote file name (uses local filename if not provided)
            content_type: MIME type (auto-detected if not provided)
            public: Whether to make file publicly accessible
            
        Returns:
            Public URL of uploaded file
            
        Raises:
            RuntimeError: If upload fails
        """
        self._ensure_authorized()
        
        bucket_name = bucket_name or self.default_bucket_name
        if not bucket_name:
            raise ValueError("bucket_name required (or set BACKBLAZE_BUCKET_NAME env var)")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Use original filename if not specified
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
            # Get bucket
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            
            # Upload file
            file_info = {}
            if public:
                file_info['b2-content-disposition'] = 'inline'
            
            logger.info(
                "Uploading %s to B2 bucket '%s' as '%s' (%.2f MB)",
                file_path.name, bucket_name, file_name,
                file_path.stat().st_size / 1_048_576
            )
            
            uploaded_file = bucket.upload_local_file(
                local_file=str(file_path),
                file_name=file_name,
                content_type=content_type,
                file_infos=file_info
            )
            
            # Generate public URL
            # Format: https://f{bucket_id}.backblazeb2.com/file/{bucket_name}/{file_name}
            download_url = self.b2_api.get_download_url_for_file_name(
                bucket_name, file_name
            )
            
            logger.info("B2 upload successful: %s", download_url)
            return download_url
            
        except Exception as e:
            logger.exception("B2 upload failed")
            raise RuntimeError(f"Failed to upload to B2: {e}")
    
    def delete_file(
        self,
        file_name: str,
        bucket_name: str = None
    ) -> bool:
        """
        Delete a file from B2 by file name.
        
        Args:
            file_name: Name of file to delete
            bucket_name: B2 bucket name (uses default if not provided)
            
        Returns:
            True if deleted successfully
        """
        self._ensure_authorized()
        
        bucket_name = bucket_name or self.default_bucket_name
        if not bucket_name:
            raise ValueError("bucket_name required")
        
        try:
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            
            # List file versions (B2 keeps versions)
            file_versions = list(bucket.ls(file_name, latest_only=True))
            
            if not file_versions:
                logger.warning("File not found in B2: %s", file_name)
                return False
            
            # Delete all versions
            for file_version, _ in file_versions:
                self.b2_api.delete_file_version(
                    file_version.id_,
                    file_version.file_name
                )
                logger.info("Deleted B2 file: %s (version %s)", file_name, file_version.id_)
            
            return True
            
        except Exception as e:
            logger.exception("B2 delete failed")
            raise RuntimeError(f"Failed to delete from B2: {e}")


# Global B2 manager instance (initialized on first use)
_b2_manager: Optional[B2Manager] = None

def get_b2_manager() -> B2Manager:
    """Get or create the global B2 manager instance."""
    global _b2_manager
    if _b2_manager is None:
        _b2_manager = B2Manager()
    return _b2_manager


def upload_to_b2(
    video_path: Path,
    yt_link: str = "",
    bucket_name: str = None
) -> str:
    """
    Upload a video to Backblaze B2 and return the public URL.
    
    This is a drop-in replacement for upload_to_imagekit().
    
    Args:
        video_path: Path to video file
        yt_link: Original YouTube link (used for logging, not uploaded)
        bucket_name: B2 bucket name (optional)
        
    Returns:
        Public URL of uploaded video
    """
    manager = get_b2_manager()
    
    # Generate unique filename with timestamp
    from datetime import datetime
    timestamp = int(datetime.utcnow().timestamp())
    file_name = f"videos/{timestamp}_{video_path.name}"
    
    return manager.upload_file(
        file_path=video_path,
        bucket_name=bucket_name,
        file_name=file_name,
        content_type="video/mp4",
        public=True
    )
