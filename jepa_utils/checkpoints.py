import os
import glob


def find_latest_checkpoint(pattern: str, directory: str = "checkpoints") -> str:
    """Return the most recent checkpoint file matching ``pattern``.

    Parameters
    ----------
    pattern: str
        Glob pattern to match checkpoint files.
    directory: str
        Directory to search. Defaults to ``"checkpoints"``.

    Returns
    -------
    str
        Path to the newest matching checkpoint, or an empty string if none
        found.
    """
    search_path = os.path.join(directory, pattern)
    candidates = glob.glob(search_path)
    if not candidates:
        return ""
    latest = max(candidates, key=os.path.getmtime)
    return latest
