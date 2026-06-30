"""Optional experiment-tracking initialisation."""
from __future__ import annotations

from loguru import logger

from app.config import settings


def init() -> None:
    """Initialise external services (wandb, sentry, ...)."""
    if settings.wandb_entity:
        try:
            import wandb

            wandb.init(
                project=settings.wandb_project,
                entity=settings.wandb_entity,
                job_type="serve",
            )
            logger.info("W&B initialised")
        except Exception:
            logger.warning("W&B init failed — continuing without tracking")
    else:
        logger.debug("W&B tracking skipped (no entity configured)")
