"""Make the in-tree package (python/) and the shared test cases importable."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(os.path.dirname(HERE), "python"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
