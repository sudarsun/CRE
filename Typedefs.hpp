#ifndef __CRE_TYPEDEFS_HPP__
#define __CRE_TYPEDEFS_HPP__

//#include "Matrix.hpp"
#include <vector>

typedef std::vector<float> real_array;
typedef std::vector<double> double_array;
typedef std::vector<int> int_array;

struct PropsAtPivot {
	real_array left;
	real_array right;
};

#ifdef ARMA_64BIT_WORD
typedef std::vector<unsigned long long> uint_array;
#else
typedef std::vector<unsigned> uint_array;
#endif

typedef std::pair<int, float> weight_t;
typedef std::vector<weight_t> weights_t;

class Matrix;
typedef std::vector<Matrix> matrix_array;

class DenseMatrix;
typedef std::vector<DenseMatrix> dense_matrix_array;

/*
real_array operator +(const real_array &a, const real_array &b) {
	if (a.size() < b.size()) {
		real_array out(b.size());
		for (int i = 0; i < a.size(); ++i )
			out[i] = a[i] + b[i];
		for (int i = a.size(); i < b.size(); ++i)
			out[i] = b[i];

		return out;
	}

	real_array out(a.size());
	for (int i = 0; i < b.size(); ++i )
		out[i] = a[i] + b[i];
	for (int i = b.size(); i < a.size(); ++i)
		out[i] = a[i];

	return out;
}

real_array & operator +=(real_array &a, const real_array &b) {
	if (a.size() < b.size()) {
		real_array out(b.size());
		for (int i = 0; i < a.size(); ++i )
			out[i] = a[i] + b[i];
		for (int i = a.size(); i < b.size(); ++i)
			out[i] = b[i];

		a = out;
	}
	else {
		real_array out(a.size());
		for (int i = 0; i < b.size(); ++i )
			out[i] = a[i] + b[i];
		for (int i = b.size(); i < a.size(); ++i)
			out[i] = a[i];

		a = out;
	}

	return a;
}*/

#endif // __CRE_TYPEDEFS_HPP__
