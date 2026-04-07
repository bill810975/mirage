from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(this_dir, '../../../include')

setup(
    name='test_mla_kernel',
    ext_modules=[
        CUDAExtension(
            name='test_mla_kernel',
            sources=[
                os.path.join(this_dir, 'test_mla_kernel_wrapper.cu'),
            ],
            include_dirs=[
                include_dir,
            ],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': [
                    '-O2',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
