from setuptools import find_packages, setup

setup(
    name="mlProject",
    version="0.0.1",
    author="WineMetric Team",
    description="End-to-end ML pipeline for wine quality prediction with MLflow",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)