cmake_minimum_required(VERSION 3.13)

project(CRE)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DARMA_DONT_USE_WRAPPER)
add_definitions(-DARMA_USE_ARPACK)
add_definitions(-DARMA_64BIT_WORD)
add_definitions(-DBOOST_LOG_DYN_LINK)

include_directories(/opt/projects/mosek/8/tools/platform/linux64x86/h /usr/lib/jvm/java-8-openjdk-amd64/include /usr/lib/jvm/java-8-openjdk-amd64/include/linux /opt/projects/CLI11/include)
link_directories(/opt/projects/mosek/8/tools/platform/linux64x86/bin)
link_libraries(mosek64 openblas pthread boost_system boost_chrono boost_filesystem boost_thread boost_log boost_log_setup)

add_library(cre_rt SHARED Utils.cpp Kernel.RBF.cpp Stopwatch.cpp QuadProg.cpp Data.cpp ClassRatioEstimator.cpp LibSVMFormat.cpp Matrix.Sparse.cpp Matrix.Dense.cpp ClassRatioEstimatorRuntime.cpp ClassRatioEstimatorJNI.cpp)

add_executable(cre main.cpp)
target_link_libraries(cre cre_rt)

install(TARGETS cre RUNTIME DESTINATION bin)
install(TARGETS cre_rt LIBRARY DESTINATION bin)


