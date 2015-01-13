#include <iostream>
#include "Matrix.hpp"
#include "LibSVMFormat.hpp"
#include <fstream>
#include <boost/chrono/duration.hpp>
#include <boost/chrono/system_clocks.hpp>
#include <boost/thread.hpp>
#include "Kernel.hpp"
#include "Data.hpp"
#include "ClassRatioEstimator.hpp"
#include "Stopwatch.hpp"

/* @TODO
 * 1. Save kernels
 * 2. Create different apps for different steps
 * 3. Cross Validation based kernel selection
 * 	DONE 4. Multi thread
 * 5. Memory leaks and crashes
 */

// usage: <tr-data> <val-data>
int main(int argc, char **argv)
{
	if ( argc < 3 )
	{
		std::cerr << argv[0] << " <tr-data> <val-data> [type=0:naive,1:threaded_naive,2:armadillo]\n" << std::endl;
		return -1;
	}

	Data trdata;
	std::string filename(argv[1]);
	trdata.Load( filename );

	Data tedata;
	std::string te_fname( argv[2] );
	tedata.Load( te_fname );

	int minLabel = trdata.MinLabel();
	int maxLabel = trdata.MaxLabel();

	const Matrix &tr_features = trdata.Features();
	const Matrix &tr_labels = trdata.Labels();
	const Matrix &te_features = tedata.Features();

	ClassRatioEstimator cre;
	float bw = cre.BandwidthSelect( tr_features );

	KernelImplType type = eNAIVE;
	if ( argc > 3  )
	{
		int t = atoi(argv[3]);
		if ( t >= 0 and t <= 2 )
			type = (KernelImplType)t;

		if ( type == eNAIVE_THREADED )
			std::cout << "using " << boost::thread::hardware_concurrency() << " threads .." << std::endl;
	}

	weights_t wts;
	cre.BestKernel(trdata, tedata, wts);

	real_array props = cre.ClassProportions(tedata.Labels());
	for ( int i = 0; i < props.size(); ++i )
		std::cout << "theta[" << i << "] = " << props[i] << std::endl;

	return -1;
/*
	RBFKernel_T kernel1(bw);
	DenseMatrix K1;

	boost::chrono::steady_clock::time_point start = boost::chrono::steady_clock::now();
	kernel1.Compute( tr_features, tr_features, K1 );
	boost::chrono::steady_clock::time_point end = boost::chrono::steady_clock::now();

	boost::chrono::steady_clock::duration diff = end - start;
	boost::chrono::milliseconds diff_ms = boost::chrono::duration_cast<boost::chrono::milliseconds>(diff);
	std::cout << "Multi-Threaded Kernel: " << diff_ms.count() << " mS" << std::endl;

	std::ofstream file( "k1" );
	K1 >> file;
	file << std::endl;
	file.close();

	RBFKernel kernel2(bw);
	DenseMatrix K2;

	boost::chrono::steady_clock::time_point start2 = boost::chrono::steady_clock::now();
	kernel2.Compute( te_features, tr_features, K2 );
	boost::chrono::steady_clock::time_point end2 = boost::chrono::steady_clock::now();

	boost::chrono::steady_clock::duration diff2 = end2 - start2;
	boost::chrono::milliseconds diff2_ms = boost::chrono::duration_cast<boost::chrono::milliseconds>(diff2);

	std::cout << "\nSingle-Threaded Kernel: " << diff2_ms.count() << " mS" << std::endl;

	std::ofstream file( "k2" );
	K2 >> file;
	file << std::endl;
	file.close();

	RBFKernel_Armadillo kernel3(bw);
	DenseMatrix K3;

	Stopwatch sw3;
	kernel3.Compute( te_features, tr_features, K3 );
	sw3.Stop();

	std::cout << "\nSingle-Threaded Armadillo Kernel: " << sw3.Elapsed() << " mS" << std::endl;

	std::ofstream file3( "k3" );
	K3 >> file3;
	file3 << std::endl;
	file3.close();

	return -1;
*/
	dense_matrix_array Krr, Kre;

	Stopwatch sw;
	cre.GetKernels( te_features, tr_features, Kre, bw, false, type );
	float re_time = sw.Elapsed();

	sw.Restart();
	cre.GetKernels( tr_features, tr_features, Krr, bw, false, type );
	float rr_time = sw.Elapsed();

	int noClasses = maxLabel;
	DenseMatrix labels = dynamic_cast<const DenseMatrix &>(tr_labels);
	if ( minLabel == 0 )
	{
		labels += 1;
		noClasses = maxLabel + 1;  // including the 0 as a label.
	}

	sw.Restart();
	real_array prop;
	cre.MMD( labels, noClasses, Krr[2], Kre[2], prop );
	float mmd_time = sw.Elapsed();

	for ( int i = 0; i < prop.size(); ++i )
		std::cout << "theta("<<i<<") = " << prop[i] << std::endl;

	std::cerr << "Kers: " << re_time << " mS" << std::endl
			  << "Krrs: " << rr_time << " mS" << std::endl
			  << "MMD:  " << mmd_time << " mS" << std::endl;

    return 0;
}
