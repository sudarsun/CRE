#include "Utils.hpp"
#include <cmath>

void Scorer::Reset(void )
{
	mScores.clear();
}

void Scorer::Add(const real_array& inScore)
{
	mScores.push_back( inScore );
}

real_array Scorer::Finale(void )
{
	int n = mScores.size();

	// if blank, no work to do.
	if ( n == 0 )
		return real_array();

	// if there is only one item, just return that.
	if ( n == 1 )
		return mScores[0];

	// if there are just two items, return the mean of the items.
	if ( n == 2 )
	{
		real_array acc = mScores[0];
		acc += mScores[1];
		acc /= 2.0;
		return acc;
	}

	int_array corepts;
	int_array nbd(n);
	float eps = 0.9;
	int minpts = 1+n/2;

	for ( int i = 0; i < n; ++i )
	{
		for ( int j = 0; j < n; ++j )
		{
			if ( i==j )
				continue;

			float distance = LpNorm( mScores[i], mScores[j], 1 );
			if ( distance < eps )
				continue;

			if ( ++nbd[i] >= minpts )
				corepts.push_back(i);
		}
	}


	// perform clustering.
	Mat<double> dmat( n, n );
	for ( int i = 0; i < n-1; ++i )
		for ( int j = i+1; j < n; ++j )
			dmat(i,j) = LpNorm( mScores[i], mScores[j], 1 );

	Col<double> colsum = sum( dmat, 1 );

	return real_array();
}


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
		error += powf(fabs(ref[i] - test[i]), p);

	return powf(error, (1.0/p));
}

float L1Score(const real_array& ref, const real_array& test)
{
	const float p = 1;
	float Lp = LpNorm(ref, test, p);

	// L1 norm has a range of 0.0 to 2.0^(1/p); Where 0.0 means very close and 2.0^(1/p) is completely off.
	// So, scaling it by multiplying with 2^(1/p).
	float Lpnorm = Lp / powf(2, (1.0/p));

	// similarity score is complement of L1norm.
	return 1.0 - Lpnorm;
}

float BinaryL1Score(const real_array& ref, const real_array& test)
{
	// it is analytically found that L1(ref,test) = |ref0-test0| = |ref1-test1|
	return 1 - fabs( ref[0] - test[0] );
}

float ModifiedBinaryL1Score( const real_array &ref, const real_array &test )
{
	// it is analytically found that L1(ref,test) = |ref0-test0| = |ref1-test1|
	// the modified L1 is given by max( |ref0-test0|/ref0, |ref1-test1|/ref1 )
	float L1 = fabs(ref[0]-test[0]);
	return 1.0 - L1/std::max( ref[0], 1-ref[0] );

	float error0 = L1/ref[0], error1 = L1/ref[1];
	float modifiedL1error = std::max(error0, error1);

	// when ref = {0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95},
	// the maximum modified L1 error is {19,9,4,2.3333333333,1.5,1,1.5,2.3333333333,4,9,19} respectively
	// fitting a Linear Regression model with basis expansion set to [1 x x.^2 x.^4 x.^6 x.^8], yielded
	// the betas as [1 6.351 -21.07 831.5 -7830 2.993e+04];
	float x = abs(ref[0]-0.5), x2 = x*x, x4 = x2*x2;
	float expectedMaximumModifiedL1error = 1 + x*6.351 + x2*-21.07 + x4*831.5 + x4*x2*-7830 + x4*x4*29930;

	float scaledL1error = modifiedL1error/expectedMaximumModifiedL1error;
	float modifiedL1score = 1 - scaledL1error;

	return modifiedL1score;
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
		ioArray[i] /= scale;

	return ioArray;
}

real_array & operator *= ( real_array &ioArray, float scale )
{
	int size = ioArray.size();
	for ( int i = 0; i < size; ++i )
		ioArray[i] *= scale;

	return ioArray;
}
