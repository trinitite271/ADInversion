from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# setup(name='staggerfd',
#       ext_modules=[cpp_extension.CppExtension('staggerfd', ['staggerfd_core.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})


setup(
    name='libtorch_staggerfd_cuda',
    ext_modules=[
        cpp_extension.CppExtension('libtorch_staggerfd_cuda', [
            'staggerfd_core_cuda2.cpp',
            # 'staggerfd_core_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


# setup(
# name='libtorch_staggerfd_cuda',
# ext_modules=[
#       cpp_extension.CppExtension('libtorch_staggerfd_cuda', [
#       'staggerfd_core_cuda2.cpp',
#       # 'staggerfd_core_cuda_kernel.cu',
#       ]),
# ],
# cmdclass={
#       'build_ext': BuildExtension
# })