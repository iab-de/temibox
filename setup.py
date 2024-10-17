from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="temibox",
    version="0.1.0",
    description="Die Text-Mining Toolbox der IAB",
    packages = find_packages(),
    author="IAB",
    install_requires=requirements
)