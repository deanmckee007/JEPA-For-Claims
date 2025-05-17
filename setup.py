from setuptools import setup, find_packages

setup(
    name='jepaforclaims',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pytorch-lightning',
        'pandas',
        'scikit-learn',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm',
    ],
)
