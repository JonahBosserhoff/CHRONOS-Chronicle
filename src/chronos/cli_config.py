"""Options from a YAML file (--config), shared by train_os.py, validate_os.py and feature_ablation_os.py.

The file holds the options of the script, keyed by their names without '--'. Its values are turned
into command-line arguments and put before the real ones, so argparse converts and checks them all
alike, and the command line overrides the file:

    python src/chronos/train_os.py --config configs/train.yaml --target TTNT

null keeps an option's default. Strings get ~ and $VARS expanded; relative paths stay relative to
the working directory, as on the command line. Unknown keys and values of the wrong type are refused.
"""

import argparse
import os
import sys
from typing import List, Optional, Sequence


def _scalar(path: str, key: str, value) -> str:
    if isinstance(value, (bool, list, dict)) or value is None:
        raise ValueError(f"{path}: {key} takes a plain value (number or text), got {value!r}.")
    return os.path.expandvars(os.path.expanduser(value)) if isinstance(value, str) else str(value)


def config_argv(path: str, parser: argparse.ArgumentParser) -> List[str]:
    """The options of a YAML file as command-line arguments."""
    import yaml  # only needed with --config

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return []
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping of option: value, got {type(cfg).__name__}.")

    options = {a.dest: a for a in parser._actions if a.option_strings and a.dest not in ("help", "config")}
    unknown = sorted(str(k) for k in cfg if k not in options)
    if unknown:
        raise ValueError(f"{path}: unknown option(s) {unknown}. Valid: {sorted(options)}.")

    argv = []
    for key, value in cfg.items():
        if value is None:
            continue
        action = options[key]
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"{path}: {key} must be true or false, got {value!r}.")
            argv.append(flag if value else f"--no-{flag[2:]}")
        elif action.nargs is not None and action.nargs != "?":
            # options with several values, e.g. logvar_bounds: [-8.0, 8.0]
            if not isinstance(value, list):
                raise ValueError(f"{path}: {key} takes a list, e.g. [a, b], got {value!r}.")
            if isinstance(action.nargs, int) and len(value) != action.nargs:
                raise ValueError(f"{path}: {key} takes {action.nargs} values, got {len(value)}: {value!r}.")
            argv.append(flag)
            argv += [_scalar(path, key, v) for v in value]
        else:
            # --flag=value: also right for empty values and values that start with '-'
            argv.append(f"{flag}={_scalar(path, key, value)}")
    return argv


def parse_args(parser: argparse.ArgumentParser, required: Sequence[str] = (),
               argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line, with the options of --config (if given) in front of it.

    `required` options may come from either; they are checked after both are merged.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", type=str, default="")
    config = pre.parse_known_args(argv)[0].config
    if config:
        argv = config_argv(config, parser) + argv  # later arguments win: the command line overrides the file
    args = parser.parse_args(argv)
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"missing {', '.join('--' + m for m in missing)} (on the command line or in --config)")
    return args
