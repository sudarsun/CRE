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

#include "LibSVMFormat.hpp"
#include "Matrix.hpp"
#include "Stopwatch.hpp"
#include <stdexcept>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>

bool LibSVMFormat::Read(const std::string& inFileName, SparseMatrix& outFeatures, DenseMatrix& outLabels)
{
	std::ifstream file( inFileName.c_str() );
	if ( file.good() == false )
		return false;

	return Read( file, outFeatures, outLabels );
}

bool LibSVMFormat::Read(std::istream& inStream, SparseMatrix& outFeatures, DenseMatrix& outLabels)
{
	std::stringstream features;
	std::stringstream labels;

	while ( inStream.peek() != EOF )
	{
		std::string line;
		std::getline( inStream, line );

		boost::trim( line );
		if (line.empty())
			continue;

		std::stringstream ss( line );

		int label = 0;
		ss >> label;
		labels << label << std::endl;

		std::string remaining;
		std::getline( ss, remaining );

		features << remaining << std::endl;
	}

	try
	{
		outFeatures << features;
		outLabels << labels;
	}
	catch( std::runtime_error &e )
	{
		return false;
	}

	return true;

}

bool LibSVMFormat::Write( std::ostream &os, const Matrix& inFeatures, const Matrix& inLabels)
{
	int rows = inLabels.Rows();
	int cols = inFeatures.Columns();

	for ( int r = 0; r < rows; ++r )
	{
		os << inLabels(r, 0);
		for ( int c = 0; c < cols; ++c )
		{
			if ( inFeatures.Exists(r,c) )
			{
				os << " " << (c+1) << ":" << inFeatures(r,c);
			}
		}

		os << std::endl;
	}

	return true;
}

bool LibSVMFormat::Write( std::ostream &os, const Matrix& inFeatures)
{
	int rows = inFeatures.Rows();
	int cols = inFeatures.Columns();

	for ( int r = 0; r < rows; ++r )
	{
		for ( int c = 0; c < cols; ++c )
		{
			if ( inFeatures.Exists(r,c) )
			{
				os << (c+1) << ":" << std::setprecision(10) << inFeatures(r,c) << " ";
			}
		}

		os << std::endl;
	}

	return true;
}

bool LibSVMFormat::CheckFormat(std::istream& inStream)
{
	int lines = 0;
	std::string line;
	while ( inStream.peek() != EOF )
	{
		std::getline( inStream, line );
		boost::trim(line);

		if ( line.empty() or line[0] == '#' )
			continue;

		std::stringstream ss(line);
		if ( ss.peek() == EOF )
			return false;

		std::string label;
		ss >> label;

		int pindex = 0;
		while ( ss.peek() != EOF )
		{
			std::string token;
			ss >> token;

			size_t p = 0;
			if ( (p=token.find(':')) == std::string::npos )
				return false;

			std::string index_ = token.substr(0, p);
			int index = boost::lexical_cast<int>( index_ );

			if ( index < pindex )
				return false;

			pindex = index;
		}

		// checking only 5 lines maximum.
		if ( ++lines > 5 )
			return true;
	}

	return true;

}

bool LibSVMFormat::CheckFormat(const std::string& inFileName)
{
	std::ifstream file( inFileName.c_str() );
	if ( file.good() == false )
		return false;

	return CheckFormat(file);
}
