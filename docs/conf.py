"""Sphinx configuration for Cloud Robotics Simulation Platform."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "Cloud Robotics Simulation Platform"
copyright = "2025, Cloud Robotics Team"
author = "Cloud Robotics Team"
release = "2.0.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Templates
templates_path = ["_templates"]

# Source parsers
source_suffix = {
    ".rst": None,
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Language
language = "en"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# HTML output
html_baseurl = "https://gwinfinity.github.io/cloud-robotics-sim/"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
