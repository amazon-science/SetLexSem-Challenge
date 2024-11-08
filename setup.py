import pathlib

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_nltk_corpora():
    """Install NLTK word corpus and wordnet."""
    import nltk

    home_dir = pathlib.Path.home() / "nltk_data"
    home_dir.mkdir(parents=True, exist_ok=True)
    nltk.download("words", download_dir=home_dir)
    nltk.download("wordnet", download_dir=home_dir)


class InstallNltkWordnetDuringInstall(install):
    def run(self):
        # Run the standard install process
        install.run(self)
        install_nltk_corpora()


class InstallNltkWordnetDuringDevelop(develop):
    def run(self):
        # Run the standard install process
        develop.run(self)
        install_nltk_corpora()


setup(
    name="setlexsem",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "anthropic",
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
    cmdclass={
        "install": InstallNltkWordnetDuringInstall,
        "develop": InstallNltkWordnetDuringDevelop,
    },
)
