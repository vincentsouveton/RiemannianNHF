"""Where the outputs of a run live.

A run is identified by its config file, not by its model space: several configs
may target the same space (configs/r2.yaml and configs/r2-2.yaml), and each must
keep its own checkpoints, figures and metrics. Training and evaluation share this
single definition so that they can never disagree on where a run is stored.
"""

import os


def run_name(config_path, cfg=None):
    """Name of the run defined by `config_path`.

    Defaults to the config file's own stem, so that two configs for the same
    model space never overwrite each other. An explicit `run_name` entry in the
    config overrides it, which is the way to make several configs share one
    directory on purpose.
    """
    name = (cfg or {}).get("run_name")
    return name or os.path.splitext(os.path.basename(config_path))[0]


def run_dir(config_path, cfg=None, root="results", create=False):
    """Directory holding the outputs of that run."""
    path = os.path.join(root, run_name(config_path, cfg))
    if create:
        os.makedirs(path, exist_ok=True)
    return path
