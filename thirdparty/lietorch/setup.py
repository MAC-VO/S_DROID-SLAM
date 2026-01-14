from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
import os.path as osp


ROOT = osp.dirname(osp.abspath(__file__))

# Use system Eigen instead of bundled copy
# Common locations: /usr/include/eigen3, /usr/local/include/eigen3
EIGEN_INCLUDE = os.environ.get('EIGEN_INCLUDE_DIR', '/usr/include/eigen3')

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    author='teedrz',
    packages=['lietorch'],
    ext_modules=[
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'lietorch/include'), 
                EIGEN_INCLUDE],
            sources=[
                'lietorch/src/lietorch.cpp', 
                'lietorch/src/lietorch_gpu.cu',
                'lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O3'], 
                'nvcc': ['-O3'],
            }),

        CUDAExtension('lietorch_extras', 
            sources=[
                'lietorch/extras/altcorr_kernel.cu',
                'lietorch/extras/corr_index_kernel.cu',
                'lietorch/extras/se3_builder.cu',
                'lietorch/extras/se3_inplace_builder.cu',
                'lietorch/extras/se3_solver.cu',
                'lietorch/extras/extras.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'], 
                'nvcc': ['-O3'],
            }),
    ],
    cmdclass={ 'build_ext': BuildExtension }
)


