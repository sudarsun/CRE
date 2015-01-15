/*
 * Boost Software License - Version 1.0 - August 17th, 2003
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <armadillo>
#include "Typedefs.hpp"

/// Implementation of a Symmetric Matrix
class Matrix
{

public:

	Matrix( void ) : mRows(0), mCols(0)
	{}

	virtual ~Matrix()
	{}

	bool operator==(const Matrix& other);

	virtual Matrix & operator=( const Matrix & )
	{
		throw std::logic_error("Matrix & operator=(const Matrix &) not implemented" );
	}

	/// allocate space for the matrix.
	virtual bool Resize( int rows, int cols ) = 0;

	/// const accessor method
	virtual const float & operator()( int r, int c ) const = 0;

	/// non-const accessor/mutator method
	virtual float & operator()( int r, int c ) = 0;

	/// read a text stream to load the matrix.
	virtual void operator << ( std::istream & ) = 0;

	/// write the matrix to an output stream
	virtual void operator >> ( std::ostream & ) = 0;

	/// Append the matrix with the new matrix with columns size agreement.
	virtual void Append( const Matrix & ) = 0;

	virtual void Clear( void ) = 0;

	virtual bool Exists(int r, int c) const
	{
		if ( r >= 0 and r < mRows and c >= 0 and c < mCols )
			return true;

		return false;
	}

	int	Rows( void ) const
	{
		return mRows;
	}

	int Columns( void ) const
	{
		return mCols;
	}

	int mRows, mCols;				///< the dimension of the matrix.

};

class SymmetricMatrix : public Matrix
{

public:

	/// equals comparator
	bool operator==(const SymmetricMatrix& other);

	/// resize the matrix to (rows x cols), clears the matrix.
	bool Resize( int rows, int cols );

	/// const accessor
	const float & operator()( int r, int c ) const;

	/// non-const accessor/mutator method
	float & operator()( int r, int c );

	/// read a text stream to load the matrix.
	void operator << ( std::istream & );

	/// write the matrix to an output stream
	void operator >> ( std::ostream & );

	void Append( const Matrix & );

	Matrix & operator=( const Matrix & );

	void Clear( void );

private:

	std::vector<float> 	mData;		///< the underlying symmetric floating point matrix container.

};

class DenseMatrix : public Matrix
{

public:

	DenseMatrix(void )
	{}

	/// constructor for pre-allocation of space.
	DenseMatrix( int rows, int cols, float value = 0 );

	virtual ~DenseMatrix()
	{
	}

	/// equals comparator
	bool operator==(const DenseMatrix& other);

	DenseMatrix & operator+=( const DenseMatrix & );
	DenseMatrix & operator*=( const DenseMatrix & );
	DenseMatrix & operator*=( float scalar );
	DenseMatrix & operator+=( float scalar );
	DenseMatrix & operator-=( float scalar );

	/// clear the allocation.
	void Clear( void );

	/// resize the matrix to (rows x cols)
	bool Resize( int rows, int cols );

	/// const accessor
	const float & operator()( int r, int c ) const;

	/// non-const accessor/mutator method
	float & operator()( int r, int c );

	/// read a text stream to load the matrix.
	void operator << ( std::istream & );

	/// write the matrix to an output stream
	void operator >> ( std::ostream & );

	void Append( const Matrix & );

	Matrix & operator= ( const Matrix & );

	DenseMatrix & operator=( const DenseMatrix & );

	DenseMatrix Select( const int_array &idx1, const int_array &idx2 ) const;

	DenseMatrix Transpose( void ) const;

	enum Order_t
	{
		eRowWise,
		eColWise,
		eWholesome
	};

	DenseMatrix Mean( Order_t inOrder = eColWise ) const;
	DenseMatrix Sum( Order_t inOrder = eColWise ) const;
	DenseMatrix Min( Order_t inOrder = eColWise ) const;
	DenseMatrix Max( Order_t inOrder = eColWise ) const;

private:

	boost::numeric::ublas::matrix<float> mMatrix;

};
/*
class DenseMatrixArmadillo : public Matrix
{

public:

	DenseMatrixArmadillo(void )
	{}

	DenseMatrixArmadillo( const arma::Row<float> &row ) : mMatrix(row)
	{}

	DenseMatrixArmadillo( const arma::Col<float> &col ) : mMatrix(col)
	{}

	/// constructor for pre-allocation of space.
	DenseMatrixArmadillo( int rows, int cols, float value = 0 );

	~DenseMatrixArmadillo()
	{
	}

	/// equals comparator
	bool operator==(const DenseMatrixArmadillo& other);

	DenseMatrix & operator+=( const DenseMatrix & );
	DenseMatrix & operator*=( const DenseMatrix & );
	DenseMatrix & operator*=( float scalar );
	DenseMatrix & operator+=( float scalar );
	DenseMatrix & operator-=( float scalar );

	/// clear the allocation.
	void Clear( void )
	{
		mRows = mCols = 0;
		mMatrix.resize(0,0);
	}

	/// resize the matrix to (rows x cols)
	bool Resize( int rows, int cols )
	{
		mRows = rows, mCols = cols;
		mMatrix.resize(rows, cols);
	}

	/// const accessor
	const float & operator()( int r, int c ) const
	{
		return mMatrix(r,c);
	}

	/// non-const accessor/mutator method
	float & operator()( int r, int c )
	{
		return mMatrix(r,c);
	}

	/// read a text stream to load the matrix.
	void operator << ( std::istream & );

	/// write the matrix to an output stream
	void operator >> ( std::ostream & );

	void Append( const Matrix & );

	Matrix & operator= ( const Matrix & );

	DenseMatrix & operator=( const DenseMatrixArmadillo & );

	DenseMatrix Select( const int_array &idx1, const int_array &idx2 ) const;

	DenseMatrix Transpose( void ) const;

	arma::Mat<float> & Data( void )
	{
		return mMatrix;
	}

	const arma::Mat<float> & Data( void ) const
	{
		return mMatrix;
	}

	enum Order_t
	{
		eRowWise,
		eColWise,
		eWholesome
	};

	DenseMatrix Mean( Order_t inOrder = eColWise ) const;
	DenseMatrix Sum( Order_t inOrder = eColWise ) const;
	DenseMatrix Min( Order_t inOrder = eColWise ) const;
	DenseMatrix Max( Order_t inOrder = eColWise ) const;

private:

	arma::Mat<float> mMatrix;

};*/

class IdentityMatrix : public DenseMatrix
{

public:

	IdentityMatrix(int rows, int cols) : DenseMatrix(rows, cols)
	{
		int k = std::min(rows, cols);
		for ( int i = 0; i < k; ++i )
			(*this)(i, i) = 1;
	}
};

class SparseMatrix : public Matrix
{

public:

	~SparseMatrix()
	{}

	/// equals comparator
	bool operator==(const SparseMatrix& other);

	/// clear the allocation.
	void Clear( void );

	/// resize the matrix to (rows x cols)
	bool Resize( int rows, int cols );

	/// const accessor
	const float& operator()( int r, int c ) const;

	/// non-const accessor/mutator method
	float & operator()( int r, int c );

	/// read a text stream to load the matrix.
	void operator << ( std::istream & );

	/// write the matrix to an output stream
	void operator >> ( std::ostream & );

	/// existence check of a cell.
	bool Exists( int row, int col ) const;

	void Append( const Matrix & );

	Matrix & operator= ( const Matrix & );

private:

	boost::numeric::ublas::compressed_matrix<float> mMatrix;

};

#endif // MATRIX_H
