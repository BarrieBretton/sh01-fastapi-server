import asyncio

# One heavy media job at a time per SH01 process. Both the existing audio-image
# renderer and the caption/clip renderer should use this lock.
MEDIA_RENDER_LOCK = asyncio.Lock()
