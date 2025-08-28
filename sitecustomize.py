"""Project-level sitecustomize to disable writing bytecode files (""__pycache__"" folders).

This module is imported automatically by Python's site machinery when the interpreter
is started (as long as the site module is enabled). It sets the interpreter to not
write .pyc files which prevents generation of __pycache__ directories inside the
project workspace during development runs.
"""
import sys
import os

# Disable writing .pyc files
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
