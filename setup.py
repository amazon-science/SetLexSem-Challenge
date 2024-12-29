from setuptools import find_packages, setup

# Read long description from README file
with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name="setlexsem",
    version="0.0.2",
    long_description=long_description,
    long_description_content_type='text/markdown',  # or 'text/x-rst' if using .rst
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "boto3",
        "nltk",
        "tiktoken",
        "openai",
        "pandas",
        "pyyaml",
    ],
    extras_require={
        "dev": ["check-manifest", "flake8", "black"],
        "test": ["pytest", "coverage"],
    },
)
