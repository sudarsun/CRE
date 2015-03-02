cmake_minimum_required(VERSION 2.8)

project(cre.arma)

add_definitions(-DARMA_DONT_USE_WRAPPER)
add_definitions(-DARMA_USE_ARPACK)
add_definitions(-DARMA_64BIT_WORD)

include_directories(/projects/tools/mosek/7/tools/platform/linux64x86/h /projects/tools/armadillo/include)
link_directories(/projects/tools/mosek/7/tools/platform/linux64x86/bin)
link_libraries(mosek64 boost_thread boost_system boost_chrono openblas arpack boost_filesystem )

add_executable(cre Tester.cpp Utils.cpp Kernel.RBF.cpp Stopwatch.cpp QuadProg.cpp Data.cpp ClassRatioEstimator.cpp LibSVMFormat.cpp Matrix.Sparse.cpp Matrix.Dense.cpp main.cpp)

add_executable(pca pca.cpp PCA.cpp PCA.hpp Matrix.Dense.cpp Matrix.Sparse.cpp Matrix.hpp LibSVMFormat.cpp LibSVMFormat.hpp Data.cpp Data.hpp )

add_executable(trans transform.cpp PCA.cpp PCA.hpp Matrix.Dense.cpp Matrix.Sparse.cpp Matrix.hpp LibSVMFormat.cpp LibSVMFormat.hpp Data.cpp Data.hpp )

install(TARGETS cre RUNTIME DESTINATION bin)
install(TARGETS pca RUNTIME DESTINATION bin)
install(TARGETS trans RUNTIME DESTINATION bin)
