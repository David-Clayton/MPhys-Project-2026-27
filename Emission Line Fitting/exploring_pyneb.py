# %% [markdown]
# #PyNeb short example

# %% [markdown]
# PyNeb is a very useful tool for calculating the physical properties associated with observed line emission (e.g. electron density).
# 
# Manual: https://morisset.github.io/PyNeb_Manual/html/classpyneb_1_1core_1_1pynebcore_1_1_atom.html#ab4561af1ba7e3cc0c5323cf1e4d07454
# 
# Paper: https://www.aanda.org/articles/aa/pdf/2015/01/aa23152-13.pdf
# 
# GitHub with example notebooks: https://github.com/Morisset/PyNeb_devel/tree/master
# 

# %%
# Loading packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import least_squares

# %%
! pip install pyneb

# %%
import pyneb as pn

# %%
O3_atom = pn.Atom('O',3)

# %%
O3_atom.plotGrotrian()

# %%
O3_atom.printIonic()

# %%



