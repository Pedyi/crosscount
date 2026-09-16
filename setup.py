from setuptools import setup, find_packages

setup(
    name='crosscount',
    version='1.0.0',
    description='Prediction-augmented streaming counting of higher-order triadic closure',
    author='Pedram Asadzadeh',
    author_email='p.asadzadeh2016@gmail.com',
    url='https://github.com/Pedyi/crosscount',
    packages=find_packages(),
    install_requires=['numpy>=1.24', 'scipy>=1.10', 'pandas>=2.0'],
    extras_require={'experiments': ['matplotlib>=3.7', 'gdown>=5.0']},
    python_requires='>=3.9',
    license='MIT',
)
