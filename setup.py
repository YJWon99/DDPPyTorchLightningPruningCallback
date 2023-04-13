from DDPPyTorchLightningPruningCallback import  __version__
from setuptools import setup

setup(
    name='DDPPyTorchLightningPruningCallback',
    version=__version__,

    url='https://oss.navercorp.com/yunjae-won/pl-pruning-callback.git',
    author='Yunjae Won',
    author_email='yunjae.won@navercorp.com',

    py_modules=['DDPPyTorchLightningPruningCallback'],
    install_requires=['optuna']
)