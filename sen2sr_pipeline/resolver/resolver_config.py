from dataclasses import dataclass
from pathlib import Path

@dataclass
class ResolverConfig:
    """
    Centralized configuration for paths for the Sen2SR mass resolving phase.
    """
    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    ingested_tif_path: Path = base_dir / "data" / "ingest"
    resolved_files_path: Path = base_dir / "data" / "resolved"

    model_path: Path = base_dir / "model" / "sen2sr-lite"
    weights_path: Path = base_dir / "model" / "custom_weights"
