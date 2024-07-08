from setuptools import setup, find_packages

from project_metadata import NAME, VERSION, AUTHOR, DESCRIPTION, LONG_DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email="hello@phaseai.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.0.0",
        "requests>=2.24.0",
        "httpx>=0.25.0",
        "openai>=1.1.0",
        "cohere>=4.0.0",
        "python-dotenv",
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "typing-extensions>=4.6.3",
        "urllib3==2.0.7",
        "sseclient-py>=1.7.2",
        "docker>=6.1.3",
        "beautifulsoup4>=4.12.2",
        "lxml>=4.9.2",
        "fake-useragent>=1.2.1",
        "playwright>=1.35.0",
        "feedparser>=6.0.10",
        "azure-identity>=1.14.0",
        "replicate==0.20.0",
        "google-cloud-aiplatform>=1.42.1",
        "anthropic>=0.30.1",
    ],
    extras_require={
        "complete": [
            "transformers>=4.25.0",
            "accelerate>=0.16.0",
            "torch>=1.0.0",
        ],
        "docs": [
            "furo",
            "sphinx>=7.1.2",
            "myst_parser>=2.0.0",
            "sphinx-autoapi>=2.1.1",
            "sphinx-autobuild>=2021.3.14",
        ],
    },
    python_requires=">=3.8.0",
    keywords="llm, nlp, evaluation, ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
