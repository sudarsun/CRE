#include "PCA.hpp"
#include "Data.hpp"
#include "Stopwatch.hpp"
#include "LibSVMFormat.hpp"

int main( int argc, char **argv )
{
	if ( argc < 4 )
	{
		std::cerr << "usage: " << argv[0] << " <in-matrix-name> <auc cutoff> <out-coeff-matrix-name>\n" << std::endl;
		return -1;
	}

	try
	{
		Stopwatch timer;

		Data input;
		input.Load( argv[1] );

		std::cout << "loaded data in: " << timer.Restart() << "mS" << std::endl;

		double auc = atof(argv[2]);

		const Matrix &features = input.Features();

		std::cout << "input matrix: " << features.Rows() << "x" << features.Columns() << std::endl;

		DenseMatrix coeffs;
		const SparseMatrix *smat = dynamic_cast<const SparseMatrix *>( &features );
		if ( smat )
		{
			timer.Restart();
			PCA::Compute( *smat, coeffs, auc );
			std::cout << "PCA Coeff matrix: " << coeffs.Rows() << "x" << coeffs.Columns() << std::endl;
			std::cout << "PCA computed in: " << timer.Restart() << "mS" << std::endl;
		}
		else
		{
			const DenseMatrix *dmat = dynamic_cast<const DenseMatrix *>( &features );
			if ( !dmat )
				throw std::invalid_argument("invalid input matrix type, expecting sparse or dense formats");

			timer.Restart();
			PCA::Compute( *dmat, coeffs, auc );
			std::cout << "PCA Coeff matrix: " << coeffs.Rows() << "x" << coeffs.Columns() << std::endl;
			std::cout << "PCA computed in: " << timer.Restart() << "mS" << std::endl;
		}

		std::ofstream ofile( argv[3] );
		coeffs >> ofile;
	}
	catch ( std::exception &e )
	{
		std::cerr << "error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}
