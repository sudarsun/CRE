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

int_array Indices( const Matrix &inColMat, int inValue )
{
	int_array indices;
	int size = inColMat.Rows();
	for ( int i = 0; i < size; ++i )
		if ( int(inColMat(i,0)) == inValue )
			indices.push_back( i );

	return indices;
}

std::ostream & operator<<( std::ostream &ioStream, const real_array &array )
{
	for ( int i = 0; i < array.size(); ++i )
		ioStream << i << ": " << array[i] << std::endl;

	return ioStream;
}

std::ostream & operator<<( std::ostream &ioStream, const double_array &array )
{
	for ( int i = 0; i < array.size(); ++i )
		ioStream << i << ": " << array[i] << std::endl;

	return ioStream;
}

real_array ClassProportions( const Matrix &inLabels )
{
	real_array props;

	int R = inLabels.Rows();
	for ( int r = 0; r < R; ++r )
	{
		int label = inLabels(r,0);
		if ( label == 0 )
			throw std::runtime_error("class labels should be positive");

		if ( label > props.size() )
			props.resize(label);

		++props[label-1];
	}

	for ( int i = 0; i < props.size(); ++i )
		props[i] /= (float)R;

	return props;
}

float LpNorm( const real_array &ref, const real_array &test, int p )
{
	if ( ref.size() != test.size() )
		throw std::runtime_error("lpnorm: class proportions length disagreement");

	float error = 0.0;
	for ( int i = 0; i < ref.size(); ++i )
		error += pow(fabs(ref[i] - test[i]), p);

	return pow(error, (1.0/p));
}

float Correlation(const real_array& ref, const real_array& test)
{
	if ( ref.size() != test.size() )
		throw std::runtime_error("correlation: class proportions length disagreement");

	int N = ref.size();

	float mean_ref = Mean(ref), mean_test = Mean(test);
	float std_ref = StandardDeviation(ref), std_test = StandardDeviation(test);

	float sum = 0;
	for ( int i = 0; i < N; ++i )
		sum += (ref[i] - mean_ref) * (test[i] - mean_test);

	float corr = sum / ((N-1)*std_ref*std_test);
	return corr;
}

float Cosine(const real_array& ref, const real_array& test)
{
	if ( ref.size() != test.size() )
		throw std::runtime_error("cosine: class proportions length disagreement");

	int N = ref.size();

	float sum_ref = 0, sum_test = 0, prod = 0;
	for ( int i = 0; i < N; ++i )
	{
		sum_ref += ref[i]*ref[i];
		sum_test += test[i]*test[i];
		prod += ref[i] * test[i];
	}

	return prod/sqrt(sum_ref * sum_test);
}

real_array & operator += ( real_array &ioArray, const real_array &inArray )
{
	int size = ioArray.size();
	if ( !size )
		ioArray = inArray;

	if ( size && size == inArray.size() )
	{
		for ( int i = 0; i < size; ++i )
			ioArray[i] += inArray[i];
	}

	return ioArray;
}

real_array & operator /= ( real_array &ioArray, float scale )
{
	int size = ioArray.size();
	for ( int i = 0; i < size; ++i )
		ioArray /= scale;

	return ioArray;
}
