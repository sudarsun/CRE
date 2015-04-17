#include "Matrix.hpp"
#include "Stopwatch.hpp"
#include <stdexcept>
#include <exception>
#include <cstdio>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

static double kZERO = 0;

void
SparseMatrix::Clear( void )
{
	mRows = mCols = 0;
	mMatrix.reset();
}

/// const accessor
double SparseMatrix::operator()( int r, int c ) const
{
	return mMatrix(r,c);
}

/// non-const accessor/mutator method
double & SparseMatrix::operator()( int r, int c )
{
	throw std::invalid_argument( "double & SparseMatrix::operator()(rows, cols) not implemented" );
}


bool SparseMatrix::Resize( int rows, int cols )
{
	mMatrix.reshape( rows, cols );
	mRows = rows, mCols = cols;
	return true;
}

void SparseMatrix::operator<<(std::istream& is )
{
	umat posMatrix;
	Col<double> values;
	int nnzeros = 0;

	int maxcols = 0;
	int rows = 0, cols = 0;
	while ( is.peek() != EOF )
	{
		std::string line;
		std::getline( is, line );
		boost::trim( line );

		if ( line.empty() )
			continue;

		std::stringstream ss(line);
		while ( ss.peek() != EOF )
		{
			std::string tuple;
			ss >> tuple;

			size_t pos = tuple.find(':');
			if ( pos == std::string::npos )
				throw std::runtime_error( "libsvm format expected for sparse matrices" );

			std::string colid = tuple.substr( 0, pos );
			int col = boost::lexical_cast<int>(colid);
			std::string value = tuple.substr( pos+1 );
			double val = boost::lexical_cast<double>(value);

			maxcols = std::max( maxcols, col );

			int size = values.size();
			if ( size <= nnzeros )
			{
				posMatrix.reshape( 2, size + 50000 );
				values.resize( size + 50000 );
			}

			posMatrix(0, nnzeros) = rows;
			posMatrix(1, nnzeros) = col-1;
			values(nnzeros) = val;

			++nnzeros;
		}

		++rows;
	}

	// eliminate the padding values
	posMatrix.resize( 2, nnzeros );
	values.resize(nnzeros);

	// compressed matrix cannot be populated directly!
	mMatrix = SpMat<double>( posMatrix, values );

	mRows = mMatrix.n_rows;
	mCols = mMatrix.n_cols;
}

void SparseMatrix::operator>>(std::ostream& os ) const
{
	for ( int r = 0; r < mRows; ++r )
	{
		for ( int c = 0; c < mCols; ++c )
		{
			double val = mMatrix(r, c);
			if ( val )
				os << val << " ";
			else
				os << "- ";
		}

		os << std::endl;
	}
}

void SparseMatrix::Append( const Matrix &inMatrix )
{
	umat posMatrix;
	Col<double> values;
	int nnzeros = 0;

	// copy the current matrix
	for ( int r = 0; r < mRows; ++r )
	{
		for ( int c = 0; c < mCols; ++c )
		{
			double val = mMatrix(r,c);
			if ( val )
			{
				int size = values.size();
				if ( size <= nnzeros )
				{
					posMatrix.reshape( 2, size + 50000 );
					values.resize( size + 50000 );
				}

				values(nnzeros) = val;
				posMatrix( 0, nnzeros ) = r;
				posMatrix( 1, nnzeros ) = c;

				++nnzeros;
			}
		}
	}

	// copy the incoming matrix
	int inCols = inMatrix.Columns();
	int new_rows = mRows + inMatrix.Rows();
	for ( int r = mRows, r1 = 0; r < new_rows; ++r, ++r1 )
	{
		for ( int c = 0; c < inCols; ++c )
		{
			double val = inMatrix( r1, c );
			if ( val )
			{
				int size = values.size();
				if ( size <= nnzeros )
				{
					posMatrix.reshape( 2, size + 50000 );
					values.resize( size + 50000 );
				}

				values(nnzeros) = val;
				posMatrix( 0, nnzeros ) = r;
				posMatrix( 1, nnzeros ) = c;

				++nnzeros;
			}
		}
	}

	// prune the padded zeros.
	posMatrix.reshape( 2, nnzeros );
	values.resize( nnzeros );

	mMatrix = SpMat<double>( posMatrix, values );
	mRows = mMatrix.n_rows;
	mCols = mMatrix.n_cols;
}

bool SparseMatrix::Exists( int row, int col ) const
{
	return mMatrix(row, col) != 0;
}

Matrix & SparseMatrix::operator=( const Matrix &inMatrix )
{
	try
	{
		const SparseMatrix &smat = dynamic_cast<const SparseMatrix&>(inMatrix);
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

bool SparseMatrix::Load( const std::string &inName )
{
	if ( ! boost::filesystem::exists( inName.c_str() ) )
		return false;

	mMatrix.quiet_load(inName.c_str());
	mRows = mMatrix.n_rows;
	mCols = mMatrix.n_cols;

	return true;
}

void SparseMatrix::Save( const std::string &inName ) const
{
	mMatrix.quiet_save(inName.c_str());
}
