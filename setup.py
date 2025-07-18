from setuptools import setup, find_packages

setup(
    name = 'graphcmc',
    version = '0.1.0',
    author = 'Sihoon Choi',
    description = '',
    packages = find_packages(),
    include_package_data = True,
    package_data = {
        'graphcmc.data': ['*']
    },
    install_requires = [
        'fairchem-core==1.10.0'
    ],
    extras_require = {
        'torch-extensions': [
            'torch-scatter',
            'torch-sparse'
        ],
    }
)
