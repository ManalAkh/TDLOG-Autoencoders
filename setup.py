from importlib.metadata import entry_points
from setuptools import find_packages, setup

setup(
    name='Autoencoder-TDLOG',
    version ='1.0.0',
    description = 'Dimension reduction for molecular dynamics by using autoencoders',
    author='Akhannouss Boyer Vati',
    packages=find_packages(),
    install_requires=['flask', 'numpy', 'flask_sqlalchemy', 'flask_wtf', 'flask_login', 'flask_bcrypt', 'wtforms', 'requests', 'datetime', 'matplotlib', 'scipy', 'plotly', 'pytorch', 'scikit-learn', 'mdtraj']
)