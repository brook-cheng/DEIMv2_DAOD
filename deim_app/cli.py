"""Unified CLI exposing ``train``, ``eval``, ``infer``, ``export`` subcommands.

This module is pure argparse + delegation. It builds a
:class:`~deim_app.api.DetectionModel` and calls its methods — it contains NO
model construction, preprocessing, postprocessing, or geometry. The
dependency-guard test forbids importing ``engine.*`` from here.

Approved flags ONLY (the whitelist is exhaustive):

.. code-block:: text

    all:    -c/--config (required)
    train:  --device --resume --output-dir
    eval:   -r/--checkpoint --device
    infer:  -r/--checkpoint -i/--input -o/--output --device
            --batch-size --score-threshold --top-k --class-filter --format
    export: -r/--checkpoint -o/--output --format --device

``--angle-rep``, ``-u key=value``, and any flag outside the above list are
rejected by argparse (exit code 2). Unknown subcommands exit non-zero.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from deim_app.api import DetectionModel
from deim_app.errors import (
    AppConfigError,
    CheckpointCompatibilityError,
    ExportError,
    InferenceBackendError,
    InputSourceError,
)

__all__ = ["main"]

#: Formats the ``infer`` subcommand recognises for output writing. ``--format``
#: validates against this tuple at parse time; formats coming from the config
#: default (when ``--format`` is omitted) are validated at dispatch time so an
#: unknown config-provided format raises :class:`ExportError`.
_KNOWN_FORMATS: tuple[str, ...] = ("json", "dota", "visualization")

#: Typed application-layer exceptions the CLI translates into a stderr message
#: + non-zero exit code. Any other exception propagates (programmer error).
_HANDLED_ERRORS: tuple[type[BaseException], ...] = (
    AppConfigError,
    CheckpointCompatibilityError,
    InputSourceError,
    InferenceBackendError,
    ExportError,
)


# ===========================================================================
# Parser construction
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with four subcommands.

    ``-c/--config`` is added to each subparser via a shared parent so it can
    appear AFTER the subcommand name (``train -c app.yml``) rather than before.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-c", "--config", required=True, metavar="PATH",
        help="path to the application YAML config",
    )

    parser = argparse.ArgumentParser(
        prog="deim_app",
        description="DEIM application CLI — train, evaluate, infer, export.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train -----------------------------------------------------------
    p_train = sub.add_parser(
        "train", parents=[parent], help="run one full training cycle",
    )
    p_train.add_argument("--device", default=None, help="override train.device")
    p_train.add_argument("--resume", default=None, metavar="CKPT",
                         help="override train.resume checkpoint path")
    p_train.add_argument("--output-dir", default=None, dest="output_dir",
                         metavar="DIR", help="override project.output_dir")

    # -- eval ------------------------------------------------------------
    p_eval = sub.add_parser(
        "eval", parents=[parent], help="run one evaluation pass",
    )
    p_eval.add_argument("-r", "--checkpoint", default=None,
                        help="checkpoint to evaluate (passed to load + evaluate)")
    p_eval.add_argument("--device", default=None,
                        help="override evaluation.device")

    # -- infer -----------------------------------------------------------
    p_infer = sub.add_parser(
        "infer", parents=[parent], help="run inference and write predictions",
    )
    p_infer.add_argument("-r", "--checkpoint", required=True,
                         help="model checkpoint to load")
    p_infer.add_argument("-i", "--input", required=True,
                         help="image file or directory to run inference on")
    p_infer.add_argument("-o", "--output", required=True, metavar="DIR",
                         help="output directory for prediction files")
    p_infer.add_argument("--device", default=None,
                         help="override inference.device")
    p_infer.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                         help="override inference.batch_size")
    p_infer.add_argument("--score-threshold", type=float, default=None,
                         dest="score_threshold",
                         help="minimum confidence score (default: config)")
    p_infer.add_argument("--top-k", type=int, default=None, dest="top_k",
                         help="keep at most K detections per image (default: config)")
    p_infer.add_argument("--class-filter", nargs="+", default=None,
                         dest="class_filter", metavar="NAME",
                         help="keep only the listed class names (default: config)")
    p_infer.add_argument("--format", nargs="+", default=None,
                         choices=_KNOWN_FORMATS, metavar="FMT",
                         help="output format(s): json, dota, visualization "
                              "(default: config inference.output_formats)")

    # -- export ----------------------------------------------------------
    p_export = sub.add_parser(
        "export", parents=[parent],
        help="export the model (not available in v1)",
    )
    p_export.add_argument("-r", "--checkpoint", required=True,
                          help="model checkpoint to export")
    p_export.add_argument("-o", "--output", required=True,
                          help="output path for the exported model")
    p_export.add_argument("--format", required=True,
                          help="export format (e.g. onnx)")
    p_export.add_argument("--device", default=None,
                          help="device for export")

    return parser


# ===========================================================================
# Subcommand handlers
# ===========================================================================


def _cmd_train(args: argparse.Namespace) -> int:
    """Build the model, apply CLI overrides, run ``model.train()``."""
    overrides: dict[str, dict[str, object]] = {}
    if args.device is not None:
        overrides["train"] = {"device": args.device}
    if args.resume is not None:
        overrides.setdefault("train", {})["resume"] = args.resume
    if args.output_dir is not None:
        overrides["project"] = {"output_dir": args.output_dir}

    model = DetectionModel.from_config(args.config, **overrides)
    model.train()
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Build the model, load checkpoint, run ``model.evaluate()``."""
    overrides: dict[str, dict[str, object]] = {}
    if args.device is not None:
        overrides["evaluation"] = {"device": args.device}

    model = DetectionModel.from_config(args.config, **overrides)
    model.load(args.checkpoint)
    model.evaluate(args.checkpoint)
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    """Build the model, load checkpoint, run filtered inference, write outputs."""
    overrides: dict[str, dict[str, object]] = {}
    if args.device is not None:
        overrides["inference"] = {"device": args.device}

    model = DetectionModel.from_config(args.config, **overrides)
    model.load(args.checkpoint)

    collection = model.predict_filtered(
        args.input,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        class_filter=args.class_filter,
        batch_size=args.batch_size,
    )

    # Resolve output formats: explicit --format wins, else config default.
    inference = model._adapter.loaded.app.inference  # noqa: SLF001
    formats: tuple[str, ...] = (
        tuple(args.format) if args.format is not None else inference.output_formats
    )

    # For save_images, use the same resolved score threshold as predict_filtered.
    resolved_score = (
        args.score_threshold
        if args.score_threshold is not None
        else inference.score_threshold
    )

    output = Path(args.output)
    for fmt in formats:
        if fmt == "json":
            collection.export_json(output / "predictions.json")
        elif fmt == "dota":
            collection.export_dota(output)
        elif fmt == "visualization":
            collection.save_images(output, score_threshold=resolved_score)
        else:
            raise ExportError(
                f"unknown inference output format: {fmt!r}; "
                f"known formats: {list(_KNOWN_FORMATS)}"
            )

    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Build the model and call adapter.export (always raises ExportError in v1)."""
    overrides: dict[str, dict[str, object]] = {}
    if args.device is not None:
        overrides["inference"] = {"device": args.device}

    model = DetectionModel.from_config(args.config, **overrides)
    model._adapter.export(args.checkpoint, args.format, args.output)  # noqa: SLF001
    return 0  # unreachable in v1 — export always raises ExportError


# ===========================================================================
# Entry point
# ===========================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Build the parser, dispatch to the subcommand handler, return exit code.

    Errors from the application layer (``AppConfigError``,
    ``CheckpointCompatibilityError``, ``InputSourceError``,
    ``InferenceBackendError``, ``ExportError``) print a concise message to
    stderr and return a non-zero exit code. The original exception is preserved
    as ``__cause__`` for Python API users — the facade itself never catches
    these exceptions, so programmatic callers get the original exception object
    with its natural cause chain intact.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch: dict[str, Callable[[argparse.Namespace], int]] = {
        "train": _cmd_train,
        "eval": _cmd_eval,
        "infer": _cmd_infer,
        "export": _cmd_export,
    }
    handler = dispatch[args.command]

    try:
        return handler(args)
    except _HANDLED_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
