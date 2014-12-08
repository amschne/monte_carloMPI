try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'My Project',
    'author': 'Adam Schneider',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'amschne@umich.edu',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'monte_carloMPI'
}

setup(**config)