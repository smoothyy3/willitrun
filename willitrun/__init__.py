"""willitrun — Will your ML model run on your device? Find out in one command."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("willitrun")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev"
