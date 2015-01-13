#ifndef __quadprog_hpp__
#define __quadprog_hpp__

#include <mosek.h>
#include "Matrix.hpp"
#include <cstring>
#include <armadillo>

struct Qdata
{
	Qdata( void ) : qsubi(NULL), qsubj(NULL), qval(NULL), nonzeros(0)
	{}

	Qdata( const arma::Mat<double> &QuadObjective )
	{
		int I = QuadObjective.n_rows;
		int J = QuadObjective.n_cols;

		nonzeros = 0;
		int_array iarray, jarray;
		double_array varray;
		for ( int i = 0; i < I; ++i )
		{
			for ( int j = 0; j <= i; ++j )
			{
				float value = QuadObjective(i,j);
				if ( value )
				{
					iarray.push_back(i);
					jarray.push_back(j);
					varray.push_back((double)value);
					++nonzeros;
				}
			}
		}

		qsubi = new MSKint32t[ iarray.size() ];
		qsubj = new MSKint32t[ jarray.size() ];
		qval  = new double[ varray.size() ];

		memcpy( qsubi, &iarray[0], iarray.size() * sizeof(MSKint32t) );
		memcpy( qsubj, &jarray[0], jarray.size() * sizeof(MSKint32t) );
		memcpy( qval, &varray[0], varray.size() * sizeof(double) );
	}

	Qdata( const DenseMatrix &QuadObjective )
	{
		int I = QuadObjective.Rows();
		int J = QuadObjective.Columns();

		nonzeros = 0;
		int_array iarray, jarray;
		double_array varray;
		for ( int i = 0; i < I; ++i )
		{
			for ( int j = 0; j <= i; ++j )
			{
				float value = QuadObjective(i,j);
				if ( value )
				{
					iarray.push_back(i);
					jarray.push_back(j);
					varray.push_back((double)value);
					++nonzeros;
				}
			}
		}

		qsubi = new MSKint32t[ iarray.size() ];
		qsubj = new MSKint32t[ jarray.size() ];
		qval  = new double[ varray.size() ];

		memcpy( qsubi, &iarray[0], iarray.size() * sizeof(MSKint32t) );
		memcpy( qsubj, &jarray[0], jarray.size() * sizeof(MSKint32t) );
		memcpy( qval, &varray[0], varray.size() * sizeof(double) );
	}

	~Qdata()
	{
		if ( qsubi )
			delete [] qsubi;
		if ( qsubj )
			delete [] qsubj;
		if ( qval )
			delete [] qval;
	}

	MSKint32t *qsubi;
	MSKint32t *qsubj;
	double *qval;
	int nonzeros;
};

extern int QuadProg( const Qdata &q, double *c, int classes, real_array &outValues );


#endif // __quadprog_hpp__
