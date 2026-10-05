"""
Import this router into your existing SH01 FastAPI application.

Example:

    from threads_feature.router import router as threads_router
    app.include_router(threads_router)

The feature opens its DB pool lazily. If your app already has a central lifespan
handler, you may optionally call threads_feature.router.db.connect() at startup
and db.close() at shutdown.
"""
