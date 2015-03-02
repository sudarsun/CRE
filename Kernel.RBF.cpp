/*
 * Boost Software License - Version 1.0 - August 17th, 2003
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */

#include "Kernel.hpp"
#include "Stopwatch.hpp"
#include <armadillo>


/*
 * 	function [H]=rbf_dot(patterns1,patterns2, sigma);

	size1=size(patterns1);
	size2=size(patterns2);

	G = sum((patterns1.*patterns1),2);  // compute X^2
	H = sum((patterns2.*patterns2),2);  // compute Y^2

	// compute (X-Y)^2 = X^2 + Y^2 - 2XY
	H = repmat(G,1,size2(1)) + repmat(H',size1(1),1) - 2*patterns1*patterns2';

	H=exp(-H/2/sigma^2);
*/

Mat<double> O1, O2;

void
RBFKernel::Compute( const Matrix &inA, const Matrix &inB, DenseMatrix &outKernel )
{
	using namespace arma;

	int c1 = inA.Columns(), c2 = inB.Columns();
	if ( c1 != c2 )
		throw std::runtime_error("column size disagreement");

	int n1 = inA.Rows(), n2 = inB.Rows();

	const SparseMatrix *smatrixA = dynamic_cast<const SparseMatrix *>(&inA);
	const SparseMatrix *smatrixB = dynamic_cast<const SparseMatrix *>(&inB);
	if ( smatrixA and smatrixB )
	{
		Mat<double> dmatA( smatrixA->Data() );
		Mat<double> dmatB( smatrixB->Data() );

		Mat<double> dmatA2 = dmatA % dmatA;
		Mat<double> dmatB2 = dmatB % dmatB;

		// this is slower by 10% in comparison to going via ones()
		// Mat<double> H = repmat( sum(dmatA2,1), 1, n2 ) + repmat( sum(dmatB2,1).t(), n1, 1 ) - 2*dmatA*dmatB.t();
		//Mat<double> O1(c1, n2);
		//O1.ones();
		//Mat<double> O2(n1, c1);
		//O2.ones();

		//Mat<double> H = dmatA2*O1 + O2*dmatB2.t() - 2*dmatA*dmatB.t();
		//Mat<double> K = exp( H*mScale );

		Mat<double> &K = outKernel.Data();
		K = exp((dmatA2*ones(c1,n2) + ones(n1,c1)*dmatB2.t() - 2*dmatA*dmatB.t())*mScale);
		//K = exp((dmatA2*O1 + O2*dmatB2.t() - 2*dmatA*dmatB.t())*mScale);

		// copy the data back to return variable.
		//outKernel.Data() = K;
		outKernel.mRows = K.n_rows;
		outKernel.mCols = K.n_cols;

		return;
	}

	const DenseMatrix *dmatrixA = dynamic_cast<const DenseMatrix *>(&inA);
	const DenseMatrix *dmatrixB = dynamic_cast<const DenseMatrix *>(&inB);
	if ( dmatrixA and dmatrixB )
	{
		const Mat<double> &dmatA = dmatrixA->Data();
		const Mat<double> &dmatB = dmatrixB->Data();
		Mat<double> dmatA2 = dmatA % dmatA;
		Mat<double> dmatB2 = dmatB % dmatB;

		// this is slower by 10% in comparison to going via ones()
		// Mat<double> H = repmat( dmatA2, 1, n2 ) + repmat( dmatB2.t(), n1, 1 ) - 2*dmatA*dmatB.t();
		//Mat<double> O1(c1, n2);
		//O1.ones();
		//Mat<double> O2(n1, c1);
		//O2.ones();

		//Mat<double> H = dmatA2*O1 + O2*dmatB2.t() - 2*dmatA*dmatB.t();
		//Mat<double> K = exp( H*mScale );

		Mat<double> &K = outKernel.Data();
		K = exp((dmatA2*ones(c1,n2) + ones(n1,c1)*dmatB2.t() - 2*dmatA*dmatB.t())*mScale);
		//K = exp((dmatA2*O1 + O2*dmatB2.t() - 2*dmatA*dmatB.t())*mScale);

		// copy the data back to return variable.
		//outKernel.Data() = K;
		outKernel.mRows = K.n_rows;
		outKernel.mCols = K.n_cols;

		return;
	}

	throw std::invalid_argument("RBFKernel::Compute() expects both the matrices to be of same type with Sparse or Dense" );
}

void
RBFKernel::Compute( const Matrix &inA, const Matrix &inB, DenseMatrix &outKernel, const int_array &inCols )
{
	int c1 = inA.Columns(), c2 = inB.Columns();
	if ( c1 != c2 )
		throw std::runtime_error("column size disagreement");
/*
	uvec cols;
	for ( int i = 0; i < inCols.size(); ++i )
		cols << inCols[i];

	DenseMatrix A;
	int n1 = inA.Rows();

	const SparseMatrix *smatrixA = dynamic_cast<const SparseMatrix *>(&inA);
	if ( smatrixA )
	{
		const SpMat<double> &smatA = smatrixA->Data();
		A.Data() = smatA.submat( span(0, n1-1), cols );
	}
	else
	{
		const DenseMatrix *dmatrixA = dynamic_cast<const DenseMatrix *>(&inA);
		if ( dmatrixA )
		{
			const Mat<double> &dmatA = dmatrixA->Data();
			A.Data() = dmatA.submat( span(0, n1-1), cols );
		}
		else
		{
			throw std::invalid_argument("RBFKernel::Compute() expects a Sparse or Dense matrix argument");
		}
	}

	A.mRows = n1;
	A.mCols = inCols.size();

	DenseMatrix B;
	int n2 = inB.Rows();

	const SparseMatrix *smatrixB = dynamic_cast<const SparseMatrix *>(&inB);
	if ( smatrixB )
	{
		const SpMat<double> &smatB = smatrixB->Data();
		B.Data() = smatB.submat( span(0, n2-1), inCols );
	}
	else
	{
		const DenseMatrix *dmatrixB = dynamic_cast<const DenseMatrix *>(&inB);
		if ( dmatrixB )
		{
			const Mat<double> &dmatB = dmatrixB->Data();
			B.Data() = dmatB.submat( span(0, n2-1), inCols );
		}
		else
		{
			throw std::invalid_argument("RBFKernel::Compute() expects a Sparse or Dense matrix argument");
		}
	}

	B.mRows = n2;
	B.mCols = inCols.size();

	Compute( A, B, outKernel );

	*/


	int n1 = inA.Rows(), n2 = inB.Rows();
	int cx = inCols.size();

	DenseMatrix a( n1, cx ), b( n2, cx );
	for ( int c = 0; c < cx; ++c )
	{
		int col = inCols[c];
		for ( int i = 0; i < n1; ++i )
			a(i,c) = inA(i,col);
		for ( int i = 0; i < n2; ++i )
			b(i,c) = inB(i,col);
	}

	Compute( a, b, outKernel );
}
