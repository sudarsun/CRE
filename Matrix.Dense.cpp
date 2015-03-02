#include "Matrix.hpp"
#include <stdexcept>
#include <exception>
#include <cstdio>
#include <boost/algorithm/string.hpp>
#include "Typedefs.hpp"
#include <iomanip>
#include <boost/filesystem.hpp>

using namespace arma;

DenseMatrix::DenseMatrix( int rows, int cols, double value )
{
	mMatrix.resize( rows, cols );
	mRows = rows, mCols = cols;

	// fill the matrix with the default value.
	mMatrix.fill(value);
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

	uvec row_indices(inArray1.size());
	if ( size1_ )
	{
		for ( int i = 0; i < size1_; ++i )
			row_indices[i] = inArray1[i];
	}
	else
	{
		/// @TODO: change this to span(0,n_rows-1)
		row_indices.set_size(mMatrix.n_rows);
		for ( int i = 0; i < mMatrix.n_rows; ++i )
			row_indices[i] = i;
	}

	uvec col_indices(inArray2.size());
	if ( size2_ )
	{
		for ( int i = 0; i < size2_; ++i )
			col_indices[i] = inArray2[i];
	}
	else
	{
		/// @TODO: change this to span(0,n_cols-1)
		col_indices.set_size( mMatrix.n_cols );
		for ( int i = 0; i < mMatrix.n_cols; ++i )
			col_indices[i] = i;
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
		const Row<double> &row = mean( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<double> &col = mean( mMatrix, 1 );
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
		const Row<double> &row = min( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<double> &col = min( mMatrix, 1 );
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
		const Row<double> &row = max( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<double> &col = max( mMatrix, 1 );
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
		const Row<double> &row = sum( mMatrix, 0 );
		return DenseMatrix( row );
	}
	else if ( inOrder == eRowWise )
	{
		const Col<double> &col = sum( mMatrix, 1 );
		return DenseMatrix( col );
	}

	DenseMatrix result(1,1);
	result(0,0) = sum( sum( mMatrix ) );
	return result;
}

void DenseMatrix::operator<<(std::istream& is )
{
	typedef std::vector<double> row_t;

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
			double value;
			ss >> value;
			++cols;
			temp.push_back(value);
		}

		if ( mMatrix.n_rows <= rows )
		{
			mMatrix.resize( mMatrix.n_rows+1000, cols );

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

void DenseMatrix::operator>>(std::ostream& os ) const
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
	mMatrix = inMatrix.mMatrix;
	DenseMatrix::mRows = inMatrix.Rows();
	DenseMatrix::mCols = inMatrix.Columns();

	return *this;
}

DenseMatrix & DenseMatrix::operator+=( const DenseMatrix &other )
{
	if ( Matrix::mRows != other.mRows or Matrix::mCols != other.mCols )
		throw std::runtime_error( "matrix dimensions disgree during addition" );

	mMatrix += other.mMatrix;
	return *this;
}

DenseMatrix & DenseMatrix::operator%=( const DenseMatrix &other )
{
	if ( Matrix::mRows != other.mRows or Matrix::mCols != other.mCols )
		throw std::runtime_error( "matrix dimensions disgree during element wise multiplication" );

	// perform element wise multiplication
	mMatrix %= other.mMatrix;
	return *this;
}

DenseMatrix & DenseMatrix::operator*=( double scalar )
{
	mMatrix *= scalar;
	return *this;
}

DenseMatrix & DenseMatrix::operator-=( double scalar )
{
	mMatrix -= scalar;
	return *this;
}

DenseMatrix & DenseMatrix::operator+=( double scalar )
{
	mMatrix += scalar;
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

bool DenseMatrix::Load( const std::string &inName )
{
	if ( ! boost::filesystem::exists( inName.c_str() ) )
		return false;

	mMatrix.quiet_load(inName.c_str());
	mRows = mMatrix.n_rows;
	mCols = mMatrix.n_cols;

	return true;
}

void DenseMatrix::Save( const std::string &inName ) const
{
	mMatrix.quiet_save(inName.c_str());
}
