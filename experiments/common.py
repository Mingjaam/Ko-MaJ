"""Shim: the experiments import `common`; the implementation lives in komaj.core."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from komaj.core import *  # noqa: F401,F403
from komaj.core import _bytes_to_unicode  # noqa: F401
