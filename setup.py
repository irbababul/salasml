from setuptools import setup, find_packages

setup(
    name="salasml",
    version="1.0.0",
    author="Salas",
    description="Library ML shortcut untuk pipeline, tuning, dan feature importance",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "scipy",
    ],
)
