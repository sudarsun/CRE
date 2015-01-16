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
#include "Utils.hpp"

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

	int folds = 4;
	cvdata_t cvdata;
	trdata.GetCrossValidationDataSet(folds, cvdata);

	real_array weights;
	std::ofstream file("weights");
	for ( int f = 0; f < folds; ++f )
	{
		file << "CV Iteration: " << f+1 << std::endl;

		const Data &train = cvdata[f].test;
		const Data &eval = cvdata[f].train;

		real_array wts;
		cre.BestKernel(train, eval, wts);
		weights += wts;

		std::cout << "Eval True Theta:\n" << ClassProportions(eval.Labels()) << std::endl;
		std::cout << "Train True Theta:\n" << ClassProportions(train.Labels()) << std::endl;

		file << "Eval True Theta:\n" << ClassProportions(eval.Labels()) << std::endl;
		file << "Train True Theta:\n" << ClassProportions(train.Labels()) << std::endl;
		file << "Weights:\n" << wts << std::endl;
	}

	std::cout << "Combined Weights:\n" << weights << std::endl;
	file << "Combined Weights:\n" << weights << std::endl;

	DenseMatrix Krr, Ker;

	Stopwatch sw;
	cre.GetKernels( te_features, tr_features, Ker, bw, weights );
	float re_time = sw.Elapsed();

	sw.Restart();
	cre.GetKernels( tr_features, tr_features, Krr, bw, weights );
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
	cre.MMD( labels, noClasses, Krr, Ker, prop );
	float mmd_time = sw.Elapsed();

	std::cerr << "Kers: " << re_time << " mS" << std::endl
			  << "Krrs: " << rr_time << " mS" << std::endl
			  << "MMD:  " << mmd_time << " mS" << std::endl;

	real_array true_prop = ClassProportions(tedata.Labels());
	std::cout << "True Prop:\n" << true_prop << std::endl;
	std::cout << "Estimated Prop:\n" << prop << std::endl;
	std::cout << "Correlation: " << Correlation(true_prop, prop) << std::endl;
	std::cout << "L2 Norm: " << LpNorm(true_prop, prop) << std::endl;

    return 0;
}
