# TODO: what is find_packages
# TODO: what is the right format for 'keywords'
# TODO: how do you add the right packages? What is 'packages' vs 'install_requires'? How to format it properly?

from setuptools import setup, find_packages

VERSION = '0.0.1'

DESCRIPTION = 'Wrappers for common large langugae models (LLMs) with support for evaluation.'

LONG_DESCRIPTION = 'PhaseLLM provides wrappers for common large language models and use cases. This makes it easy to swap models in and out as needed. We also provide support for evaluation of models so you can choose which models are better to use.'

setup(
    name="phasellm",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Wojciech Gryc",
    author_email="hello@phaseai.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='llm, nlp, evaluation, ai',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)