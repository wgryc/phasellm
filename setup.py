from setuptools import setup, find_packages

VERSION = "0.0.6"

DESCRIPTION = "Wrappers for common large langugae models (LLMs) with support for evaluation."

LONG_DESCRIPTION = "PhaseLLM provides wrappers for common large language models and use cases. This makes it easy to swap models in and out as needed. We also provide support for evaluation of models so you can choose which models are better to use."

setup(
    name="phasellm",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Wojciech Gryc",
    author_email="hello@phaseai.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.0.0",
        "requests>=2.24.0",
        "openai>=0.26.0",
        "cohere>=4.0.0",
        "transformers>=4.25.0",
        "accelerate>=0.16.0",
        "torch>=1.0.0",
        "python-dotenv",
    ],
    python_requires=">=3.7",
    keywords="llm, nlp, evaluation, ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
