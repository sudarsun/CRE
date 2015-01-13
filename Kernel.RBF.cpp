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
void
RBFKernel::Compute( const Matrix &inA, const Matrix &inB, DenseMatrix &outKernel )
{
	int c1 = inA.Columns(), c2 = inB.Columns();
	if ( c1 != c2 )
		throw std::runtime_error("column size disagreement");

	int n1 = inA.Rows(), n2 = inB.Rows();
	outKernel.Resize( n1, n2 );

	for ( int i = 0; i < n1; ++i )
	{
		for ( int j = 0; j < n2; ++j )
		{
			float &distance = outKernel(i,j);

			distance = 0;
			for ( int k = 0; k < c1; ++k )
			{
				float diff = inA(i,k) - inB(j,k); // perform \sum{(X_i - Y_i)^2}
				distance += diff * diff;
			}

			// perform \exp{-\over{(X-Y)^2}{2*sigma^2}}
			distance = exp(distance*mScale);
		}
	}

}

void
RBFKernel::Compute( const Matrix &inA, const Matrix &inB, DenseMatrix &outKernel, const int_array &inCols )
{
	int c1 = inA.Columns(), c2 = inB.Columns();
	if ( c1 != c2 )
		throw std::runtime_error("column size disagreement");

	int n1 = inA.Rows(), n2 = inB.Rows();
	outKernel.Resize( n1, n2 );

	int csize = inCols.size();
	for ( int i = 0; i < n1; ++i )
	{
		for ( int j = 0; j < n2; ++j )
		{
			float &distance = outKernel(i,j);

			distance = 0;
			for ( int kdash = 0; kdash < csize; ++kdash )
			{
				int k = inCols[kdash];
				float diff = inA(i,k) - inB(j,k);
				distance += diff * diff;

			}

			// perform \exp{-\over{(X-Y)^2}{2*sigma^2}}
			distance = exp(distance*mScale);
		}
	}
}

