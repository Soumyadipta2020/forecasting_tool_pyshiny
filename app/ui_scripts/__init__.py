"""UI package initializer.

Disable writing bytecode files when importing UI modules.
"""
import sys
import os

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
