from setuptools import setup, find_packages, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

def load_requirements(path_dir=here, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [line.strip() for line in file.readlines()]
    requirements = []
    for line in lines:
        # filer all comments
        if comment_char in line:
            line = line[: line.index(comment_char)]
        if line:  # if requirement is not empty
            requirements.append(line)
    return requirements
    
class get_pybind_include:
    """Helper class to determine the pybind11 include path. The purpose of this class is
    to postpone importing pybind11 until it is actually installed, so that the
    ``get_include()`` method can be invoked.
    """
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        [
            "src/utils/NURBS/BSPLINE2DBasisDers.cpp",
            "src/utils/NURBS/hRefineNURBS.cpp",
            "src/utils/NURBS/KnotInsertion.cpp",
            "src/utils/NURBS/NURBS.cpp",
            "src/utils/NURBS/NURBSbasisDers.cpp",
            "src/utils/NURBS/NURBSinterpolation.cpp",
            "src/utils/NURBS/NURBSOneBasis.cpp",
        ],
        language="C++",
        include_dirs=[
            os.environ.get("EIGEN_INCLUDE_DIR", "/usr/include/eigen3/"),
            get_pybind_include(),
            get_pybind_include(user=True)
        ]
    )
]

setup(
          name="geotaichi",
          version="0.1.0",
          author="Shi-YiHao",
          author_email="syh-1999@outlook.com",
          description="A Taichi-powered high-performance numerical simulator for multiscale geophysical problems",
          long_description=long_description,
          long_description_content_type="text/markdown",
          url="https://github.com/Yihao-Shi/GeoTaichi",
          packages=find_packages(include=['geotaichi', 'src', 'third_party', 'example', 'test', 'taichi_demo', 'asserts', 'docs']),
          install_requires=load_requirements(),
          classifiers=[
                          'Programming Language :: Python :: 3.8',
                          'Programming Language :: Python :: 3.9',
                          'Programming Language :: Python :: 3.10',
                          'License :: OSI Approved :: GNU License',
                          'Topic :: Software Development :: multiscale geophysical simulator',
                          'Development Status :: 3 - Alpha'
                      ]
    )
