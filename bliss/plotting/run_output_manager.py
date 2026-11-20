"""Create and validate output folders for BLiSS runs."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Optional

def create_bliss_results_folder(base_dir: str | Path='results', suffix: str='bliss') -> Path:
    """Create an exclusive timestamped BLiSS results directory.

    Parameters
    ----------
    base_dir : str or pathlib.Path, default: "results"
        Parent directory where the results folder is created.
    suffix : str, default: "bliss"
        Text appended to the timestamp in the folder name.

    Returns
    -------
    pathlib.Path
        Path to a new results folder. Collisions add _001, _002, etc., even
        when several runs start concurrently within the same second.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    name = f'{timestamp}_{suffix}'
    index = 0
    while True:
        output_dir = base_dir / (name if index == 0 else f'{name}_{index:03d}')
        try:
            output_dir.mkdir()
        except FileExistsError:
            index += 1
        else:
            return output_dir

def ensure_output_folder(output_dir: Optional[str | Path]=None) -> Path:
    """Return a writable output directory for a BLiSS run.

    Parameters
    ----------
    output_dir : str, pathlib.Path, or None, default: None
        Existing or new output directory. When omitted, a timestamped results
        folder is created exclusively. Existing folders are allowed; writers
        check their actual destination files before replacing any results.

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


def _output_exists_error(path):
    return FileExistsError(
        f'Output already exists: {path}. Choose another output directory or '
        'filename, or set overwrite=True to replace existing results.'
    )


def _check_output_paths(paths, *, overwrite=False):
    """Validate all destinations before computation or writing begins."""
    paths = tuple(Path(path) for path in paths)
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Output filenames must be distinct; plot_name conflicts with another output.')
    for path in paths:
        if path.is_dir():
            raise IsADirectoryError(f'Output path is a directory: {path}')
        if not overwrite and (path.exists() or path.is_symlink()):
            raise _output_exists_error(path)
    return paths


@contextmanager
def _reserve_output_paths(paths, *, overwrite=False):
    """Exclusively reserve all files before writing; release new files on error.

    Callers may write to their own reserved paths with overwrite=True. A second
    call with overwrite=False cannot acquire those paths, even concurrently.
    Existing files are never removed during cleanup.
    """
    paths = _check_output_paths(paths, overwrite=overwrite)
    created = {}
    try:
        for path in paths:
            try:
                with path.open('xb') as handle:
                    stat = os.fstat(handle.fileno())
                    created[path] = (stat.st_dev, stat.st_ino)
            except FileExistsError:
                if not overwrite:
                    raise _output_exists_error(path) from None
                if path.is_dir():
                    raise IsADirectoryError(f'Output path is a directory: {path}')
        yield
    except BaseException:
        for path, identity in created.items():
            try:
                stat = path.lstat()
                if (stat.st_dev, stat.st_ino) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise


def _plot_output_path(path):
    """Return savefig's actual destination, including its default extension."""
    path = Path(path)
    if not path.suffix:
        from matplotlib import rcParams
        path = path.with_name(path.name.rstrip('.') + '.' + rcParams['savefig.format'])
    return path
