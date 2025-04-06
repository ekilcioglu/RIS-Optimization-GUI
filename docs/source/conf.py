import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath('../..'))

# -- Project Information ---------------------------------------------------
project = 'Ray-Tracing Based RIS Deployment Optimization for Coverage Enhancement'
copyright = '2025, Communication Systems (CoSy) Group/ICTEAM/UCLouvain by Emre Kilcioglu and Claude Oestges'
author = 'Emre Kilcioglu and Claude Oestges'

# -- General Configuration -------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Automatically generate API documentation
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Include source code links in documentation
    'sphinx.ext.autosummary',  # Generate summary tables for modules
]

autosummary_generate = True  # Enable automatic summary generation

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML Output -----------------------------------------------
html_theme = "sphinx_rtd_theme"
#html_static_path = ['_static']  # Uncommented to allow custom CSS/images
