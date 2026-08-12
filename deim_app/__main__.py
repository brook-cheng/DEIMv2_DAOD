"""``python -m deim_app`` entry point — delegates to :func:`deim_app.cli.main`."""

from deim_app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
