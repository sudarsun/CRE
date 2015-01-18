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
 * DONE 3. Cross Validation based kernel selection
 * 	DONE 4. Multi thread
 *  DONE 5. Memory leaks and crashes
 *DONE 6. PSD issues (eig), (random perturb)
 * 7. equal proportions train set while CV.
 */

// usage: <tr-data> <val-data>
int main(int argc, char **argv)
{
	if ( argc < 4 )
	{
		std::cerr << argv[0] << " <tr-data> <val-data> <cvfolds> [type=0:naive,1:threaded_naive,2:armadillo]\n" << std::endl;
		return -1;
	}

	try {

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
	if ( argc > 4  )
	{
		int t = atoi(argv[4]);
		if ( t >= 0 and t <= 2 )
			type = (KernelImplType)t;

		if ( type == eNAIVE_THREADED )
			std::cout << "using " << boost::thread::hardware_concurrency() << " threads .." << std::endl;
	}

	int folds = atoi(argv[3]);

//	cvdata_t cvdata(folds);

	cvdata_t cvdata;
	trdata.GetCrossValidationDataSet(folds, cvdata);

	real_array weights;
	std::ofstream file("weights");
	for ( int f = 0; f < folds; ++f )
	{
		file << "CV Iteration: " << f+1 << std::endl;
/*
		char name[50];
		sprintf( name, "cvdata-train-%d", f );
		cvdata[f].test.Load(name);
		sprintf( name, "cvdata-test-%d", f );
		cvdata[f].train.Load( name );
*/
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

	real_array true_prop = ClassProportions(tedata.Labels());
	std::cout << "True Prop:\n" << true_prop << std::endl;

	sw.Restart();
	real_array prop;
	if ( cre.MMD( labels, noClasses, Krr, Ker, prop ) )
	{
		std::cout << "Estimated Prop:\n" << prop << std::endl;
		std::cout << "Correlation: " << Correlation(true_prop, prop) << std::endl;
		std::cout << "Cosine Sim : " << Cosine(true_prop, prop) << std::endl;
		std::cout << "L1 Norm: " << LpNorm(true_prop, prop, 1) << std::endl;
		std::cout << "L1 Norm Corr: " << (1 - LpNorm(true_prop, prop, 1)/noClasses) << std::endl;
	}
	else
	{
		std::cout << "MMD Failed\n" << std::endl;
		for ( int f = 0; f < folds; ++f )
		{
			char name[50];
			sprintf( name, "cvdata-train-%d", f );
			cvdata[f].test.Save( name );
			sprintf( name, "cvdata-test-%d", f );
			cvdata[f].train.Save( name );
		}
	}

	float mmd_time = sw.Elapsed();

	std::cerr << "Kers: " << re_time << " mS" << std::endl
			  << "Krrs: " << rr_time << " mS" << std::endl
			  << "MMD:  " << mmd_time << " mS" << std::endl;

	} catch ( std::exception &e ) {
		std::cerr << e.what() << std::endl;
	}

    return 0;
}
