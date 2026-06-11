"""landtake — shared library for the land-take detection experiments.

The per-experiment workflow from the methodology pipeline (data, models,
losses, metrics, logging) lives here so all three experiments import it instead
of reaching into each other. Submodules: paths, config, losses, metrics,
logger, models, data.
"""

__version__ = "0.1.0"
