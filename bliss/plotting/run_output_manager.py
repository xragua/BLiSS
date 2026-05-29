"""Create and validate output folders for BLiSS runs."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional

def create_bliss_results_folder(base_dir: str | Path='results', suffix: str='bliss') -> Path:
    """Create a timestamped BLiSS results directory.

    Parameters
    ----------
    base_dir : str or pathlib.Path, default: "results"
        Parent directory where the results folder is created.
    suffix : str, default: "bliss"
        Text appended to the timestamp in the folder name.

    Returns
    -------
    pathlib.Path
        Path to the created results folder.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(base_dir) / f'{timestamp}_{suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def ensure_output_folder(output_dir: Optional[str | Path]=None) -> Path:
    """Return a writable output directory for a BLiSS run.

    Parameters
    ----------
    output_dir : str, pathlib.Path, or None, default: None
        Existing or new output directory. When omitted, a timestamped results
        folder is created.

    Returns
    -------
    pathlib.Path
        Directory guaranteed to exist.
    """
    if output_dir is None:
        return create_bliss_results_folder()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
