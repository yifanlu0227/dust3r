from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='DUST3R',
    version='0.1.0',
    # packages=find_packages(),
    packages=['dust3r', 'dust3r.utils', 'dust3r.heads', 'dust3r.datasets', 'dust3r.cloud_opt', 
              'croco', 'croco.models', 'croco.stereoflow', 'croco.utils', 'croco.models.curope'],
    ext_modules=[
        CUDAExtension(
            name='curope2d.cuRoPE2D',
            sources=['croco/models/curope/curope.cpp', 'croco/models/curope/kernels.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    author='naver',
)