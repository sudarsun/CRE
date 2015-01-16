#include "Matrix.hpp"
#include <stdexcept>
#include <exception>
#include <cstdio>
#include <boost/algorithm/string.hpp>
#include "Typedefs.hpp"
#include <cfloat>
#include <iomanip>

using namespace arma;

DenseMatrix::DenseMatrix( int rows, int cols, float value )
{
	mMatrix.resize( rows, cols );
	mRows = rows, mCols = cols;

	if ( value )
	{
		for ( int r = 0; r < mRows; ++r )
			for ( int c = 0; c < mCols; ++c )
				mMatrix(r,c) = value;
	}
}

DenseMatrix DenseMatrix::Select( const int_array &inArray1, const int_array &inArray2 ) const
{
	int size1_ = inArray1.size(), size2_ = inArray2.size();
	DenseMatrix result;
	if ( size1_ == 0 and size2_ == 0 )
	{
		result.mMatrix = mMatrix;
		result.mRows = mMatrix.n_rows;
		result.mCols = mMatrix.n_cols;
		return result;
	}

	uvec row_indices;
	if ( size1_ )
	{
		for ( int i = 0; i < size1_; ++i )
			row_indices << inArray1[i];
	}
	else
	{
		for ( int i = 0; i < mMatrix.n_rows; ++i )
			row_indices << i;
	}

	uvec col_indices;
	if ( size2_ )
	{
		for ( int i = 0; i < size2_; ++i )
			col_indices << inArray2[i];
	}
	else
	{
		for ( int i = 0; i < mMatrix.n_cols; ++i )
			col_indices << i;
	}

	result.mMatrix = mMatrix.submat( row_indices, col_indices );
	result.mRows = result.mMatrix.n_rows;
	result.mCols = result.mMatrix.n_cols;

	return result;
}

DenseMatrix DenseMatrix::Mean( Order_t inOrder ) const
{
	if ( inOrder == eColWise )
	{
		const Row<float> &row = mean( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<float> &col = mean( mMatrix, 1 );
		return DenseMatrix( col );
	}

	DenseMatrix result(1,1);
	result(0,0) = mean( mean( mMatrix ) );
	return result;

}

DenseMatrix DenseMatrix::Min( Order_t inOrder ) const
{
	if ( inOrder == eColWise )
	{
		const Row<float> &row = min( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<float> &col = min( mMatrix, 1 );
		return DenseMatrix( col );
	}

	DenseMatrix result(1,1);
	result(0,0) = min( min( mMatrix ) );
	return result;
}

DenseMatrix DenseMatrix::Max( Order_t inOrder ) const
{
	if ( inOrder == eColWise )
	{
		const Row<float> &row = max( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<float> &col = max( mMatrix, 1 );
		return DenseMatrix( col );
	}

	DenseMatrix result(1,1);
	result(0,0) = max( max( mMatrix ) );
	return result;

}

DenseMatrix DenseMatrix::Sum( Order_t inOrder ) const
{
	if ( inOrder == eColWise )
	{
		const Row<float> &row = sum( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<float> &col = sum( mMatrix, 1 );
		return DenseMatrix( col );
	}

	DenseMatrix result(1,1);
	result(0,0) = sum( sum( mMatrix ) );
	return result;
}

void DenseMatrix::operator<<(std::istream& is )
{
	typedef std::vector<float> row_t;

	int rows = 0, cols = 0;
	while ( is.peek() != EOF )
	{
		std::string line;
		std::getline( is, line );
		boost::trim( line );

		if ( line.empty() )
			continue;

		row_t temp;

		cols = 0;
		std::stringstream ss(line);
		while ( ss.peek() != EOF )
		{
			float value;
			ss >> value;
			++cols;
			temp.push_back(value);
		}

		if ( mMatrix.n_rows <= rows )
		{
			mMatrix.resize( mMatrix.n_rows+100, cols );

			if ( mMatrix.n_cols and mMatrix.n_cols != cols )
				throw std::runtime_error( "matrix malformed" );
		}

		for ( int i = 0; i < cols; ++i )
			mMatrix( rows, i ) = temp[i];

		++rows;
	}

	mMatrix.resize(rows, cols);

	mRows = mMatrix.n_rows;
	mCols = mMatrix.n_cols;
}

void DenseMatrix::operator>>(std::ostream& os )
{
	for ( int r = 0; r < mRows; ++r )
	{
		for ( int c = 0; c < mCols; ++c )
		{
			os << std::setprecision(5) << mMatrix(r, c) << " ";
		}

		os << std::endl;
	}
}

void DenseMatrix::Append( const Matrix &inMatrix )
{
	// resize the matrix allocation to accomodate new rows.
	mMatrix.resize( mMatrix.n_rows + inMatrix.Rows(), mMatrix.n_cols );

	for ( int r = mRows, r1 = 0; r < mRows+inMatrix.Rows(); ++r, ++r1 )
	{
		for ( int c = 0; c < mCols; ++c )
			mMatrix( r, c ) = inMatrix(r1, c);
	}

	mRows = mMatrix.n_rows;
}


Matrix & DenseMatrix::operator=( const Matrix &inMatrix )
{
	try
	{
		const DenseMatrix &smat = dynamic_cast<const DenseMatrix&>(inMatrix);
		mMatrix = smat.mMatrix;
		Matrix::mRows = inMatrix.Rows();
		Matrix::mCols = inMatrix.Columns();
	}
	catch( std::bad_cast & )
	{
		throw std::runtime_error( "assignment operator only defined for same matrix types" );
	}

	return *this;
}

DenseMatrix & DenseMatrix::operator=( const DenseMatrix &inMatrix )
{
	//mMatrix.resize( inMatrix.Rows(), inMatrix.Columns());
	mMatrix = inMatrix.mMatrix;
	DenseMatrix::mRows = inMatrix.Rows();
	DenseMatrix::mCols = inMatrix.Columns();

	return *this;
}

DenseMatrix & DenseMatrix::operator+=( const DenseMatrix &other )
{
	if ( Matrix::mRows != other.mRows or Matrix::mCols != other.mCols )
		throw std::runtime_error( "matrix dimensions disgree during addition" );

	for ( int r = 0; r < mRows; ++r )
		for ( int c = 0; c < mCols; ++c )
			(*this)(r,c) += other(r,c);

	return *this;
}

DenseMatrix & DenseMatrix::operator*=( const DenseMatrix &other )
{
	if ( Matrix::mRows != other.mRows or Matrix::mCols != other.mCols )
		throw std::runtime_error( "matrix dimensions disgree during element wise multiplication" );

	for ( int r = 0; r < mRows; ++r )
		for ( int c = 0; c < mCols; ++c )
			(*this)(r,c) *= other(r,c);

	return *this;

}

DenseMatrix & DenseMatrix::operator*=( float scalar )
{
	for ( int r = 0; r < mRows; ++r )
		for ( int c = 0; c < mCols; ++c )
			(*this)(r,c) *= scalar;

	return *this;

}

DenseMatrix & DenseMatrix::operator-=( float scalar )
{
	for ( int r = 0; r < mRows; ++r )
		for ( int c = 0; c < mCols; ++c )
			(*this)(r,c) -= scalar;

	return *this;

}

DenseMatrix & DenseMatrix::operator+=( float scalar )
{
	for ( int r = 0; r < mRows; ++r )
		for ( int c = 0; c < mCols; ++c )
			(*this)(r,c) += scalar;

	return *this;

}

DenseMatrix DenseMatrix::Transpose( void ) const
{
	DenseMatrix temp;
	temp.mMatrix = this->mMatrix.t();

	temp.mRows = mCols;
	temp.mCols = mRows;

	return temp;
}
