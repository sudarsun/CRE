#include "Matrix.hpp"
#include <stdexcept>
#include <exception>
#include <cstdio>
#include <boost/algorithm/string.hpp>
#include "Typedefs.hpp"
#include <cfloat>
#include <iomanip>

DenseMatrix::DenseMatrix( int rows, int cols, float value )
{
	mMatrix.resize( rows, cols, false );
	mRows = rows, mCols = cols;

	if ( value )
	{
		for ( int r = 0; r < mRows; ++r )
			for ( int c = 0; c < mCols; ++c )
				mMatrix(r,c) = value;
	}
}

void
DenseMatrix::Clear( void )
{
	mRows = mCols = 0;
	mMatrix.resize(0,0);
}

bool DenseMatrix::Resize( int rows, int cols )
{
	mMatrix.resize( rows, cols );
	mRows = rows, mCols = cols;
	return true;
}

float & DenseMatrix::operator()( int r, int c )
{
	return mMatrix.at_element(r, c);

}

const float & DenseMatrix::operator()( int r, int c ) const
{
	return (*const_cast<DenseMatrix*>(this))(r, c);

}

DenseMatrix DenseMatrix::Select( const int_array &inArray1, const int_array &inArray2 ) const
{
	int size1_ = inArray1.size(), size2_ = inArray2.size();
	int size1 = mRows, size2 = mCols;

	// if array1 is empty, select all the rows
	if ( size1_ == 0 and size2_ > 0 )
	{
		size2 = size2_;
	}
	// if array2 is empty, select all the cols
	else if ( size1_ > 0 and size2_ == 0 )
	{
		size1 = size1_;
	}
	// if both arrays are valid, use only those indices
	else if ( size1_ > 0 and size2_ > 0 )
	{
		size1 = size1_;
		size2 = size2_;
	}
	// default is use all the rows and cols.

	DenseMatrix ktemp( size1, size2 );
	for ( int i = 0; i < size1; ++i )
		for ( int j = 0; j < size2; ++j )
		{
			int p = i;
			if ( size1_ ) p = inArray1[i];

			int q = j;
			if ( size2_ ) q = inArray2[j];

			if ( !Exists(p, q) )
				throw std::runtime_error( "index out of bounds accessing dense matrix" );

			ktemp(i,j) = (*this)( p, q );
		}

	return ktemp;
}

DenseMatrix DenseMatrix::Mean( Order_t inOrder ) const
{
	DenseMatrix result;
	if ( inOrder == eColWise )
	{
		result.Resize(1, mCols);
		for ( int c = 0; c < mCols; ++c )
		{
			float sum = 0;
			for ( int r = 0; r < mRows; ++r )
				sum += (*this)(r,c);

			sum /= mRows;
			result(0, c) =  sum;
		}
	}
	else if ( inOrder == eRowWise )
	{
		result.Resize(mRows, 1);
		for ( int r = 0; r < mRows; ++r )
		{
			float sum = 0;
			for ( int c = 0; c < mCols; ++c )
				sum += (*this)(r,c);

			sum /= mCols;
			result(r,0) = sum;
		}
	}
	else if ( inOrder == eWholesome )
	{
		result.Resize(1,1);
		float sum = 0;
		for ( int c = 0; c < mCols; ++c )
			for ( int r = 0; r < mRows; ++r )
				sum += (*this)(r,c);

		sum /= (float)(mRows*mCols);
		result(0,0) = sum;
	}

	return result;
}

DenseMatrix DenseMatrix::Min( Order_t inOrder ) const
{
	DenseMatrix result;
	if ( inOrder == eColWise )
	{
		result.Resize(1, mCols);
		for ( int c = 0; c < mCols; ++c )
		{
			float min = FLT_MAX;
			for ( int r = 0; r < mRows; ++r )
				min = std::min( min, (*this)(r,c) );

			result(0, c) =  min;
		}
	}
	else if ( inOrder == eRowWise )
	{
		result.Resize(mRows, 1);
		for ( int r = 0; r < mRows; ++r )
		{
			float min = FLT_MAX;
			for ( int c = 0; c < mCols; ++c )
				min = std::min( min, (*this)(r,c) );

			result(r,0) = min;
		}
	}
	else if ( inOrder == eWholesome )
	{
		result.Resize(1,1);
		float min = FLT_MAX;
		for ( int c = 0; c < mCols; ++c )
			for ( int r = 0; r < mRows; ++r )
				min = std::min( min, (*this)(r,c) );

		result(0,0) = min;
	}

	return result;
}

DenseMatrix DenseMatrix::Max( Order_t inOrder ) const
{
	DenseMatrix result;
	if ( inOrder == eColWise )
	{
		result.Resize(1, mCols);
		for ( int c = 0; c < mCols; ++c )
		{
			float max = FLT_MIN;
			for ( int r = 0; r < mRows; ++r )
				max = std::max( max, (*this)(r,c) );

			result(0, c) =  max;
		}
	}
	else if ( inOrder == eRowWise )
	{
		result.Resize(mRows, 1);
		for ( int r = 0; r < mRows; ++r )
		{
			float max = FLT_MIN;
			for ( int c = 0; c < mCols; ++c )
				max = std::max( max, (*this)(r,c) );

			result(r,0) = max;
		}
	}
	else if ( inOrder == eWholesome )
	{
		result.Resize(1,1);
		float max = FLT_MIN;
		for ( int c = 0; c < mCols; ++c )
			for ( int r = 0; r < mRows; ++r )
				max = std::max( max, (*this)(r,c) );

		result(0,0) = max;
	}

	return result;
}

DenseMatrix DenseMatrix::Sum( Order_t inOrder ) const
{
	DenseMatrix result;
	if ( inOrder == eColWise )
	{
		result.Resize(1, mCols);
		for ( int c = 0; c < mCols; ++c )
		{
			float sum = 0;
			for ( int r = 0; r < mRows; ++r )
				sum += (*this)(r,c);

			result(0, c) =  sum;
		}
	}
	else if ( inOrder == eRowWise )
	{
		result.Resize(mRows, 1);
		for ( int r = 0; r < mRows; ++r )
		{
			float sum = 0;
			for ( int c = 0; c < mCols; ++c )
				sum += (*this)(r,c);

			result(r,0) = sum;
		}
	}
	else if ( inOrder == eWholesome )
	{
		result.Resize(1,1);
		float sum = 0;
		for ( int c = 0; c < mCols; ++c )
			for ( int r = 0; r < mRows; ++r )
				sum += (*this)(r,c);

		result(0,0) = sum;
	}

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

		if ( mMatrix.size1() <= rows )
		{
			mMatrix.resize( mMatrix.size1()+100, cols );

			if ( mMatrix.size2() and mMatrix.size2() != cols )
				throw std::runtime_error( "matrix malformed" );
		}

		for ( int i = 0; i < cols; ++i )
			mMatrix.insert_element(rows, i, temp[i]);

		++rows;
	}

	mMatrix.resize(rows, cols);

	mRows = mMatrix.size1();
	mCols = mMatrix.size2();
}

void DenseMatrix::operator>>(std::ostream& os )
{
	for ( int r = 0; r < mRows; ++r )
	{
		for ( int c = 0; c < mCols; ++c )
		{
			os << std::setprecision(5) << mMatrix.at_element(r, c) << " ";
		}

		os << std::endl;
	}
}

void DenseMatrix::Append( const Matrix &inMatrix )
{
	// resize the matrix allocation to accomodate new rows.
	mMatrix.resize( mMatrix.size1() + inMatrix.Rows(), mMatrix.size2() );

	for ( int r = mRows, r1 = 0; r < mRows+inMatrix.Rows(); ++r, ++r1 )
	{
		for ( int c = 0; c < mCols; ++c )
			mMatrix.insert_element( r, c, inMatrix(r1, c) );
	}

	mRows = mMatrix.size1();
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
	mMatrix.resize( inMatrix.Rows(), inMatrix.Columns());
	mMatrix = inMatrix.mMatrix;
	Matrix::mRows = inMatrix.Rows();
	Matrix::mCols = inMatrix.Columns();

	return *this;
}

DenseMatrix & DenseMatrix::operator+=( const DenseMatrix &other )
{
	if ( mRows != other.mRows or mCols != other.mCols )
		throw std::runtime_error( "matrix dimensions disgree during addition" );

	for ( int r = 0; r < mRows; ++r )
		for ( int c = 0; c < mCols; ++c )
			(*this)(r,c) += other(r,c);

	return *this;
}

DenseMatrix & DenseMatrix::operator*=( const DenseMatrix &other )
{
	if ( mRows != other.mRows or mCols != other.mCols )
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
	DenseMatrix temp(*this);

	boost::numeric::ublas::trans( temp.mMatrix );

	temp.mRows = mCols;
	temp.mCols = mRows;

	return temp;
}
