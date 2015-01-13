#include "Utils.hpp"

float StandardDeviation( const real_array &inArray )
{
	int N = inArray.size();
/*
	float sum_x = 0, sum_x2 = 0;

	for ( int i = 0; i < N; ++i )
	{
		sum_x += inArray[i];
		sum_x2 += inArray[i]*inArray[i];
	}

	float e_x = sum_x/N;
	float e_x2 = sum_x2/N;

	return sqrt( e_x2 - e_x*e_x );*/

	// std (x) = sqrt ( 1/(N-1) SUM_i (x(i) - mean(x))^2 )

	float sum_x = 0;
	for ( int i = 0; i <N; ++i )
		sum_x += inArray[i];

	float mean_x = sum_x / N;

	sum_x = 0;
	for ( int i = 0; i < N; ++i )
	{
		float val = inArray[i] - mean_x;
		sum_x += val * val;
	}

	return sqrt( sum_x / (N-1) );
}

real_array StandardDeviation( const Matrix &inMatrix )
{
	int N = inMatrix.Rows();
	int F = inMatrix.Columns();

	real_array mean(F);
	for ( int c = 0; c < F; ++c )
	{
		float &m = mean[c];
		for ( int i = 0; i < N; ++i )
			m += inMatrix(i,c);

		m /= N;
	}

	real_array sum(F);
	for ( int c = 0; c < F; ++c )
	{
		float &s = sum[c];
		for ( int i = 0; i < N; ++i )
		{
			float val = inMatrix(i,c) - mean[c];
			s += val*val;
		}

		s = sqrt( s / (N-1) );
	}

	return sum;

/*
	int cols = inMatrix.Columns();
	real_array stddev(cols);

	int rows = inMatrix.Rows();
	real_array sum_x(cols), sum_x2(cols);

	for ( int r = 0; r < rows; ++r )
	{
		for ( int c = 0; c < cols; ++c )
		{
			float val = inMatrix(r,c);
			sum_x[c] += val;
			sum_x2[c] += val*val;
		}
	}

	for ( int c = 0; c < cols; ++c )
	{
		float e_x = sum_x[c] / rows;
		float e_x2 = sum_x2[c] / rows;

		stddev[c] = sqrt( e_x2 - e_x*e_x );
	}

	return stddev;*/
}

real_array StandardDeviation( const Matrix &inMatrix, const int_array &inCols )
{
	int N = inMatrix.Rows();
	int F = inMatrix.Columns();
	real_array stddev(F);
	int csize = inCols.size();

	real_array mean(csize);
	for ( int c = 0; c < csize; ++c )
	{
		float &m = mean[inCols[c]];
		for ( int i = 0; i < N; ++i )
			m += inMatrix(i,inCols[c]);

		m /= N;
	}

	real_array sum(csize);
	for ( int c = 0; c < csize; ++c )
	{
		float &s = sum[inCols[c]];
		for ( int i = 0; i < N; ++i )
		{
			float val = inMatrix(i,inCols[c]) - mean[c];
			s += val*val;
		}

		s = sqrt( s / (N-1) );
	}

	return sum;
/*
	int cols = inMatrix.Columns();
	real_array stddev(cols);

	int rows = inMatrix.Rows();

	int csize = inCols.size();
	real_array sum_x(csize), sum_x2(csize);

	for ( int r = 0; r < rows; ++r )
	{
		for ( int c = 0; c < csize; ++c )
		{
			float val = inMatrix(r,inCols[c]);
			sum_x[c] += val;
			sum_x2[c] += val*val;
		}
	}

	for ( int c = 0; c < csize; ++c )
	{
		float e_x = sum_x[c] / rows;
		float e_x2 = sum_x2[c] / rows;

		stddev[c] = sqrt( e_x2 - e_x*e_x );
	}

	return stddev;*/
}

float Mean( const real_array &array )
{
	int size = array.size();

	float sum = 0;
	for ( int i = 0; i < size; ++i )
		sum += array[i];

	return sum/size;
}

int_array Indices( const int_array &inArray, int inValue )
{
	int_array indices;
	int size = inArray.size();
	for ( int i = 0; i < size; ++i )
		if ( int(inArray[i]) == inValue )
			indices.push_back( i );

	return indices;
}
