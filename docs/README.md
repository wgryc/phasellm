### Docs Setup

1) Install docs dependencies
    ```
    pip install -e .[docs]
    ```

2) Run a local docs server
   ```
   sphinx-autobuild docs/source/ docs/build/html
   ```

### Docs Helper Tools

* Convert reStructuredText (.rst) to Markdown (.md)
    ```
    pip install rst-to-myst[sphinx]
    rst2myst convert docs/**/*.rst
    ```

### Useful Resources

* Document Your Scientific Project With Markdown, Sphinx, and Read the Docs | PyData Global 2021
https://www.youtube.com/watch?v=qRSb299awB0