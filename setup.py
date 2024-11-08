from setuptools import find_packages, setup

setup(
    name="setlexsem",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "anthropic",
        "boto3",
        "tiktoken",
        "openai",
        "pandas",
        "pyyaml",
    ],
    extras_require={
        "dev": ["check-manifest", "flake8", "black"],
        "test": ["pytest", "coverage"],
        "nltk": ["nltk"],
    },
)
