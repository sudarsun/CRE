cmake_minimum_required(VERSION 2.8)

project(cre)

include_directories(/projects/tools/mosek/7/tools/platform/linux64x86/h)
link_directories(/projects/tools/mosek/7/tools/platform/linux64x86/bin)
link_libraries(mosek64 boost_thread boost_system boost_chrono armadillo)

add_executable(cre  Utils.cpp Tester.cpp Kernel.RBF.Armadillo.cpp Stopwatch.cpp Kernel.RBF.Threaded.cpp QuadProg.cpp Data.cpp ClassRatioEstimator.cpp LibSVMFormat.cpp Matrix.Sparse.cpp Matrix.Dense.cpp Matrix.Symmetric.cpp Kernel.RBF.cpp main.cpp)

install(TARGETS cre RUNTIME DESTINATION bin)
