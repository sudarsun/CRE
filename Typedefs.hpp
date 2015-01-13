#ifndef __CRE_TYPEDEFS_HPP__
#define __CRE_TYPEDEFS_HPP__

//#include "Matrix.hpp"
#include <vector>

typedef std::vector<float> real_array;
typedef std::vector<double> double_array;
typedef std::vector<int> int_array;

typedef std::pair<int, float> weight_t;
typedef std::vector<weight_t> weights_t;

class Matrix;
typedef std::vector<Matrix> matrix_array;

class DenseMatrix;
typedef std::vector<DenseMatrix> dense_matrix_array;

#endif // __CRE_TYPEDEFS_HPP__
