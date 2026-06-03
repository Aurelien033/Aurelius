"""
Solus-7B — unit tests.
"""
import __main__
import sys, os

base = os.path.dirname(os.path.abspath(__file__))  # tests/
src  = os.path.join(base, "..", "src")
if src not in sys.path:
    sys.path.insert(0, src)

sys.path.insert(0, "/Users/christienantonio/aurelius/projects/solus/src")
import math

def run(): ...