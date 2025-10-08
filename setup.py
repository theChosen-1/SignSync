from setuptools import setup, find_packages

setup(
    name="signsync",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.3',
        'matplotlib>=3.7.1',
        'pandas>=2.0.2',
        'scikit-learn>=1.2.2',
        'seaborn>=0.12.2',
    ],
)
