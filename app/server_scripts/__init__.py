"""server_scripts package initializer.

Set interpreter flags early to avoid writing bytecode files when modules
under this package are imported.
"""
import sys
import os

# Ensure this interpreter session does not write .pyc files
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
