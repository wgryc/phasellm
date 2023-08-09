import sys

sys.path.append('../../')

from project_metadata import NAME, VERSION, AUTHOR  # noqa: E402

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = NAME
copyright = f'2023, {AUTHOR}'
author = AUTHOR
release = VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add paths to the Python source code.
sys.path.append('../../phasellm')

# Allow markdown files to be used.
extensions = [
    'myst_parser',
    'autoapi.extension',
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

# Configure autoapi.
autoapi_dirs = ['../../phasellm']
autoapi_python_class_content = "init"

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
