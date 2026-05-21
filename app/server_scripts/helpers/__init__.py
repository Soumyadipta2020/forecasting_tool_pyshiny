"""helpers package initializer.

Disable writing bytecode files during import of helper modules.
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
