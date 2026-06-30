"""Backward-compatible wrapper for old CLI module path."""

from ml_package.cli.predict import main


if __name__ == "__main__":
    raise SystemExit(main())
