import os
from os.path import join as pjoin
from setuptools import setup, find_namespace_packages
from setuptools.command.build_py import build_py as _build_py
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    conda_cuda = pjoin(os.environ['CONDA_PREFIX'], 'pkgs', 'cudatoolkit-dev')
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
        libdir = pjoin(home, 'lib64')
    elif os.path.exists(conda_cuda):
        # otherwise, use the cudatoolkit from conda
        home = conda_cuda
        nvcc = pjoin(home, 'bin', 'nvcc')
        libdir = pjoin(home, 'lib64')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
        proc = subprocess.Popen("dirname $(ldconfig -p | grep libcudart.so | awk '{print $4}' | head -n 1)", shell=True, stdout=subprocess.PIPE)
        out, err = proc.communicate()
        libdir = out.rstrip()

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib': libdir}
    return cudaconfig

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


_radonusfft = Extension(
    'ptychotomo._radonusfft',
    swig_opts=['-c++', '-Isrc/include'],
    sources=['src/ptychotomo/radonusfft.i', 'src/cuda/radonusfft.cu'],
    library_dirs=[CUDA['lib']],
    libraries=['cudart','cufft','cublas'],
    # this syntax is specific to this build system
    # we're only going to use certain compiler args with nvcc and not with gcc
    # the implementation of this trick is in customize_compiler() below
    extra_compile_args={'gcc': [],
                        'nvcc': ['--compiler-options', "'-fPIC' '-O3' "]},
    extra_link_args=['-lgomp'],
    include_dirs = [numpy_include, CUDA['include'], 'src/include'],)

_ptychofft = Extension(
    'ptychotomo._ptychofft',
    swig_opts=['-c++', '-Isrc/include'],
    sources=['src/ptychotomo/ptychofft.i', 'src/cuda/ptychofft.cu'],
    library_dirs=[CUDA['lib']],
    libraries=['cudart','cufft','cublas'],
    extra_compile_args={'gcc': [],
                        'nvcc': ['--compiler-options', "'-fPIC' '-O3' "]},
    extra_link_args=['-lgomp'],
    include_dirs = [numpy_include, CUDA['include'], 'src/include'],)


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# custom build_py does build_ext first so SWIG generated python module is
# copied after it is created
class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

setup(
    name='ptychotomo',
    author='Viktor Nikitin',
    version='0.3.0',
    # this is necessary so that the swigged python file gets picked up
    package_dir={"": "src"},
    packages=find_namespace_packages(where='src', include=['ptychotomo*']),
    ext_modules = [_radonusfft, _ptychofft],
    cmdclass={
        'build_py' : build_py,
        'build_ext': custom_build_ext,
    },
    zip_safe=False
)
