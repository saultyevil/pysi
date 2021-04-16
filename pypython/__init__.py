import sys

import automodinit

name = "pypython"
sys.path.append("physics")

# Import all files using automodinit

__all__ = ["I will get rewritten"]
automodinit.automodinit(__name__, __file__, globals())
del automodinit
