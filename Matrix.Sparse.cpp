#include "Matrix.hpp"
#include <stdexcept>
#include <exception>
#include <cstdio>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

static const float kZERO = 0;

void
SparseMatrix::Clear( void )
{
	mRows = mCols = 0;
	mMatrix.resize(0,0);
}

bool SparseMatrix::Resize( int rows, int cols )
{
	mMatrix.resize( rows, cols );
	mRows = rows, mCols = cols;
	return true;
}

float & SparseMatrix::operator()( int r, int c )
{
	return mMatrix(r, c).ref();
}

const float & SparseMatrix::operator()( int r, int c ) const
{
	if ( Exists(r,c) )
		return (*const_cast<SparseMatrix*>(this))(r, c);

	return kZERO;
}

void SparseMatrix::operator<<(std::istream& is )
{
	typedef std::pair<int,float> tuple_t;
	typedef std::vector<tuple_t> tuples_t;
	typedef std::vector<tuples_t> matrix_t;

	matrix_t matrix;
	int maxcols = 0;

	int rows = 0, cols = 0;
	while ( is.peek() != EOF )
	{
		std::string line;
		std::getline( is, line );
		boost::trim( line );

		if ( line.empty() )
			continue;

		tuples_t tuples;

		std::stringstream ss(line);
		while ( ss.peek() != EOF )
		{
			std::string tuple;
			ss >> tuple;

			int pos = tuple.find(':');
			if ( pos == std::string::npos )
				throw std::runtime_error( "libsvm format expected for sparse matrices" );

			std::string colid = tuple.substr( 0, pos );
			int col = boost::lexical_cast<int>(colid);
			std::string value = tuple.substr( pos+1 );
			float val = boost::lexical_cast<float>(value);

			maxcols = std::max( maxcols, col );

			tuples.push_back( tuple_t( col, val ) );
		}

		matrix.push_back( tuples );
		++rows;
	}

	// compressed matrix cannot be populated directly!
	boost::numeric::ublas::compressed_matrix<float> temp( rows, maxcols, rows*maxcols );
	for ( int i = 0; i < rows; ++i )
	{
		tuples_t &tuples = matrix[i];
		for (int j = 0; j < tuples.size(); ++j )
		{
			temp(i, tuples[j].first - 1) = tuples[j].second;
		}
	}

	mMatrix = temp;

	mRows = mMatrix.size1();
	mCols = mMatrix.size2();
}

void SparseMatrix::operator>>(std::ostream& os ) const
{
	for ( int r = 0; r < mRows; ++r )
	{
		for ( int c = 0; c < mCols; ++c )
		{
			float *f = (const_cast<boost::numeric::ublas::compressed_matrix<float> &> (mMatrix)).find_element(r, c);
			if ( f )
				os << *f << " ";
			else
				os << "- ";
		}

		os << std::endl;
	}
}

void SparseMatrix::Append( const Matrix &inMatrix )
{
	int new_rows = mRows + inMatrix.Rows();
	int new_cols = std::max(mCols, inMatrix.Columns());
	boost::numeric::ublas::compressed_matrix<float> temp( new_rows, new_cols, new_rows*new_cols );

	// copy the current matrix
	for ( int r = 0; r < mRows; ++r )
	{
		for ( int c = 0; c < mCols; ++c )
		{
			if ( mMatrix.find_element(r, c) != NULL )
				temp(r, c) = mMatrix(r,c);
		}
	}

	// copy the incoming matrix
	int inCols = inMatrix.Columns();
	for ( int r = mRows, r1 = 0; r < new_rows; ++r, ++r1 )
	{
		for ( int c = 0; c < inCols; ++c )
		{
			if ( inMatrix.Exists(r1, c) )
				temp(r, c) = inMatrix(r1,c);
		}
	}

	mMatrix = temp;
	mRows = mMatrix.size1();
	mCols = mMatrix.size2();
}

bool SparseMatrix::Exists( int row, int col ) const
{
	return mMatrix.find_element(row, col) != NULL;
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
