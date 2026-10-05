"""
Import this router into your existing SH01 FastAPI application.

Example:

    from threads_feature.router import router as threads_router
    app.include_router(threads_router)

The feature initializes Settings/DB/service lazily on the first authenticated
Threads request. This intentionally lets the master/control-plane run the same app.py
without carrying Threads runtime secrets.
"""
