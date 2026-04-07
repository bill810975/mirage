from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(this_dir, '../../..')
include_dir = os.path.join(repo_root, 'include')
pk_tasks_dir = os.path.join(include_dir, 'mirage/persistent_kernel')
# cutlass_stub overrides cutlass/arch/barrier.h to avoid SM90 TMA cascade errors
cutlass_stub_dir = os.path.join(this_dir, 'cutlass_stub')

setup(
    name='test_mla_kernel',
    ext_modules=[
        CUDAExtension(
            name='test_mla_kernel',
            sources=[
                os.path.join(this_dir, 'test_mla_kernel_wrapper.cu'),
            ],
            include_dirs=[
                cutlass_stub_dir,  # must be first to shadow real CUTLASS
                include_dir,
                pk_tasks_dir,
            ],
            extra_compile_args={
                'cxx': ['-std=c++17', '-DMIRAGE_BACKEND_USE_CUDA'],
                'nvcc': [
                    '-O2',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                    '-DMIRAGE_BACKEND_USE_CUDA',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
