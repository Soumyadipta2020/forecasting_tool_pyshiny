"""helpers package initializer.

Disable writing bytecode files during import of helper modules.
"""
import sys
import os

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
# This file marks the helpers directory as a Python package after moving.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
