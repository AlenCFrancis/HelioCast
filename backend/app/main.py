"""
FastAPI Main Application for Solar Flare Forecasting.
Initializes CORS, pre-loads ML models on startup via lifespan handler,
and mounts API v1 routes.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.endpoints import router as api_router
from app.services.inference import inference_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("solarcaster")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for FastAPI: Preloads models into memory on startup."""
    logger.info("Initializing Solar Flare Forecasting Service...")
    try:
        inference_service.load_artifacts()
        logger.info("ML models and preprocessor artifacts loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load artifacts during startup: %s", exc)
    yield
    logger.info("Shutting down Solar Flare Forecasting Service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Operational API for predicting >= M-class solar flares in the next 24 hours.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware Configuration
origins = settings.CORS_ORIGINS
if "*" in origins:
    allow_origins = ["*"]
else:
    allow_origins = origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Endpoints
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root_status():
    """Service status and API documentation entry point."""
    return JSONResponse({
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "OPERATIONAL",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
        "cadence_hours": 1.0,
        "forecast_horizon_hours": settings.FORECAST_HOURS
    })
