import os
import os.path as osp

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

# Use system Eigen instead of bundled copy
EIGEN_INCLUDE = os.environ.get('EIGEN_INCLUDE_DIR', '/usr/include/eigen3')

setup(
    name='droid_slam',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=[EIGEN_INCLUDE],
            sources=[
                'src/droid.cpp', 
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
