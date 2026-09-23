# pybind11 module over the already-built libORB_SLAM3.
#
# Separate from upstream's CMakeLists on purpose: upstream stays byte-identical to the
# pinned commit, and this links against the library it produced rather than rebuilding
# any of it.
cmake_minimum_required(VERSION 3.12)
project(orbslam3_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ORB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../upstream")
set(COMPAT   "${CMAKE_CURRENT_SOURCE_DIR}/../compat_include")

find_package(pybind11 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

pybind11_add_module(orbslam3_py orbslam3_py.cpp)

# Thirdparty/DBoW2 includes <boost/serialization/...> in its headers without ever
# declaring the dependency (see bug_log.txt [5]) -- works on Linux where boost is in
# /usr/include, fails here. DEPS_INCLUDE_DIR is the conda prefix that actually has it.
set(DEPS_INCLUDE_DIR "" CACHE PATH "Prefix providing boost headers")

target_include_directories(orbslam3_py PRIVATE
  ${COMPAT}                       # <stdint-gcc.h>, <tr1/*> shims; see build_macos.sh
  ${DEPS_INCLUDE_DIR}
  ${ORB_ROOT}
  ${ORB_ROOT}/include
  ${ORB_ROOT}/include/CameraModels
  ${ORB_ROOT}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(orbslam3_py PRIVATE
  ${ORB_ROOT}/lib/libORB_SLAM3.dylib
  ${ORB_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.dylib
  ${ORB_ROOT}/Thirdparty/g2o/lib/libg2o.dylib
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)

# The ORB-SLAM3 dylibs are not installed to a system prefix, so the extension has to
# carry their location or it fails at import with a dyld error that names the library
# rather than the cause.
set_target_properties(orbslam3_py PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${ORB_ROOT}/lib;${ORB_ROOT}/Thirdparty/DBoW2/lib;${ORB_ROOT}/Thirdparty/g2o/lib;${CMAKE_CURRENT_SOURCE_DIR}/../pangolin-install/lib"
)
