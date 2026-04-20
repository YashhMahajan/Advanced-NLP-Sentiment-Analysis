"""
FastAPI main application for Advanced NLP sentiment analysis service.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from .routes import router
from .schemas import ErrorResponse
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("nlp_api", level="INFO")

# Create FastAPI app
app = FastAPI(
    title="Advanced NLP Sentiment Analysis API",
    description="Production-grade sentiment analysis service with multiple ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["sentiment"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            error_message="An unexpected error occurred"
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Advanced NLP Sentiment Analysis API starting up...")
    
    # Initialize model manager
    try:
        from .routes import get_model_manager
        manager = get_model_manager()
        logger.info(f"Model manager initialized with {len(manager.models)} models")
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
    
    logger.info("API startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Advanced NLP Sentiment Analysis API shutting down...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Advanced NLP Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
