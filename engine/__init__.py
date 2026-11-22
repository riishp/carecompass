"""The team's recommendation engine (06_recommend), wrapped for the app.

recommend(service_text, location_text) -> dict with status, service_resolution, location_resolution,
results, search_notes, how_we_decided — exactly the notebook's contract.

ENGINE_WRITE_LOGS=1 turns on the engine's own logging/caching writes (query_log_service,
query_log_recommend, generated_mappings). Off by default: the app only has read access.
"""
import os

from engine import recommend_engine as _e

if os.getenv("ENGINE_WRITE_LOGS", "0") != "1":
    _e._log_call = lambda *a, **k: None
    _e._log_recommend = lambda *a, **k: None
    _e._save_generated_mapping = lambda *a, **k: None

recommend = _e.recommend
resolve_service = _e.resolve_service
resolve_location = _e.resolve_location
WEIGHTS = _e.WEIGHTS
