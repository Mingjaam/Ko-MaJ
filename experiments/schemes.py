"""Shim: the experiments import `schemes`; the implementation lives in komaj.schemes."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from komaj.schemes import *  # noqa: F401,F403
