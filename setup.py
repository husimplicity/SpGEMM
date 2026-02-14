from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "scipy_sparse_opt._spgemm_cpp",
        ["src/scipy_sparse_opt/_spgemm_cpp.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
