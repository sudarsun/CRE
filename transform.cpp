#include "PCA.hpp"
#include "Data.hpp"
#include "Stopwatch.hpp"
#include "LibSVMFormat.hpp"

int main( int argc, char **argv )
{
	if ( argc < 4 )
	{
		std::cerr << "usage: " << argv[0] << " <in-matrix-name> <coeff-mat-name> <out-matrix-name>\n" << std::endl;
		return -1;
	}

	try
	{
		Stopwatch timer;

		Data input;
		input.Load( argv[1] );

		std::cout << "loaded data in: " << timer.Restart() << "mS" << std::endl;

		const Matrix &features = input.Features();
		std::cout << "input matrix: " << features.Rows() << "x" << features.Columns() << std::endl;

		DenseMatrix coeff;
		std::ifstream cfile( argv[2] );
		coeff << cfile;
		std::cout << "coefficient matrix: " << coeff.Rows() << "x" << coeff.Columns() << std::endl;

		DenseMatrix trans;
		PCA::Transform(features, coeff, trans);

		std::cout << "Transformed matrix: " << trans.Rows() << "x" << trans.Columns() << std::endl;

		std::ofstream ofile( argv[3] );
		LibSVMFormat::Write( ofile, trans, input.Labels());
	}
	catch ( std::exception &e )
	{
		std::cerr << "error: " << e.what() << std::endl;
		return -1;
	}


	return 0;
}
