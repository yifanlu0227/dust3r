from setuptools import setup, find_packages
from torch import cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()

setup(
    name='DUST3R',
    version='0.1.0',
    # packages=find_packages(),
    packages=['dust3r', 'dust3r.utils', 'dust3r.heads', 'dust3r.datasets', 'dust3r.cloud_opt', 
              'croco', 'croco.models', 'croco.stereoflow', 'croco.utils', 'croco.models.curope'],
    ext_modules=[
        CUDAExtension(
            name='croco.models.curope.curope',
            sources=['croco/models/curope/curope.cpp', 'croco/models/curope/kernels.cu'],
            extra_compile_args = dict(
                nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                cxx=['-O3']
            )
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    author='naver',
)