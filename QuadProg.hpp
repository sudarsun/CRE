#ifndef __quadprog_hpp__
#define __quadprog_hpp__

#include <mosek.h>
#include "Matrix.hpp"
#include <cstring>
#include <armadillo>

struct Qdata
{
	Qdata() : qsubi(nullptr), qsubj(nullptr), qval(nullptr), nonzeros(0)
	{}

	[[nodiscard]] bool isOK() const
	{
		return qsubi != nullptr && qsubj != nullptr && qval != nullptr && nonzeros != 0;
	}

	Qdata & operator=(Qdata &q) noexcept
	{
		// clear the underlying memory, if we already have one.
		this->~Qdata();

		// change of ownership.
		qsubi = q.qsubi;
		qsubj = q.qsubj;
		qval = q.qval;
		nonzeros = q.nonzeros;

		// reset the original owner.
		q.qsubi = nullptr;
		q.qsubj = nullptr;
		q.qval = nullptr;
		q.nonzeros = 0;
	}

	explicit Qdata( const arma::Mat<double> &QuadObjective )
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
				double value = QuadObjective(i,j);
				if ( abs(value) > std::numeric_limits<double>::epsilon() )
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
/*
	explicit Qdata( const DenseMatrix &QuadObjective )
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
				double value = QuadObjective(i,j);
				if ( abs(value) > std::numeric_limits<double>::epsilon() )
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
	}*/

	~Qdata()
	{
		delete [] qsubi;
		delete [] qsubj;
		delete [] qval;
	}

	MSKint32t *qsubi;
	MSKint32t *qsubj;
	double *qval;
	int nonzeros;
};

extern int QuadProg( const Qdata &q, double *c, int classes, real_array &outValues );


#endif // __quadprog_hpp__
