import os
import setuptools

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

setuptools.setup(
    name="mvseg3d",
    version="0.0.1",
    author="darrenwang",
    author_email="wangyang9113@gmail.com",
    description="A Generic Framework for Multi-View Fusion based 3D Segmentation",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/boundles/mvsegmentation3d",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    ext_modules=[
        make_cuda_ext(
            name='devoxelization_ext',
            module='mvseg3d.ops.devoxelization',
            extra_include_path=['/usr/local/cuda/include'],
            sources=['src/devoxelize.cpp'],
            sources_cuda=['src/devoxelize_cuda.cu']),
        # make_cuda_ext(
        #     name='voxelization_ext',
        #     module='mvseg3d.ops.voxelization',
        #     extra_include_path=['/usr/local/cuda/include'],
        #     sources=['src/voxelize.cpp'],
        #     sources_cuda=['src/voxelize_cuda.cu']),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)
