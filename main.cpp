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
#include <armadillo>
#include "Tester.hpp"

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
	/*
	std::ifstream file( argv[1] );

	Data d1;
	d1.Load(argv[1]);

	Data d2;
	d2.Load(argv[2]);

	const Matrix &f1 = d1.Features();
	const Matrix &f2 = d2.Features();

	const SparseMatrix &s1 = dynamic_cast<const SparseMatrix &>(f1);
	const Matrix &l1 = d1.Labels();

	const SparseMatrix &s2 = dynamic_cast<const SparseMatrix &>(f2);
	const Matrix &l2 = d2.Labels();

	Stopwatch timer;

	int folds = 3;
	cvdata_t cv;
	d1.GetCrossValidationDataSet( folds, cv );

	d1.SaveCrossValidationDataSet( cv, "cvtest2" );

	/*
	for ( int i = 0; i < 13; ++i )
	{
		float r = arma::randn();
		RBFKernel_Armadillo kernel(r);
		DenseMatrix K;

		Stopwatch tm;
		kernel.Compute( s1, s2, K );
		std::cerr << tm.Elapsed() << " ";
	}

	std::cerr << "\nMulti: " << timer.Elapsed() << " mS" << std::endl;
	timer.Restart();

	int fcount = f1.Columns();
	for ( int i = 0; i < fcount; ++i )
	{
		int_array cols;
		cols.push_back(i);

		float r = arma::randn();
		RBFKernel_Armadillo kernel(r);
		DenseMatrix K;

		Stopwatch tm;
		kernel.Compute( s1, s2, K, cols );
		std::cerr << tm.Elapsed() << " ";
	}

	float tt = timer.Elapsed();

	std::cerr << "Kernel Time: " << tt << " mS" << std::endl;
//	K >> std::cout;

	return -1;
*/
	if ( argc < 4 )
	{
		std::cerr << argv[0] << " <tr-data> <val-data> <cvfolds> [cvdata]\n" << std::endl;
		return -1;
	}

	try {

		Stopwatch sw1;

	Data trdata;
	std::string filename(argv[1]);
	trdata.Load( filename );

	Data tedata;
	std::string te_fname( argv[2] );
	tedata.Load( te_fname );

	int minLabel = trdata.MinLabel();
	int maxLabel = trdata.MaxLabel();

	if ( minLabel <= 0 )
	{
		std::cerr << "minLabel=" << minLabel << ": labels are not positive, consider reordering the class labels!\n" << std::endl;
		return -1;
	}

	const Matrix &tr_features = trdata.Features();
	const Matrix &tr_labels = trdata.Labels();
	const Matrix &te_features = tedata.Features();

	ClassRatioEstimator cre;
	float bw = cre.BandwidthSelect( tr_features );

	int folds = atoi(argv[3]);

	cvdata_t cvdata;

	if ( argc > 4 )
	{
		std::cout << "loading CV data from: " << argv[4] << std::endl;
		Data::LoadCrossValidationDataSet( argv[4], cvdata );
	}
	else
		trdata.GetCrossValidationDataSet(folds, cvdata);

	sw1.Stop();
	std::cout << "loaded in: " << sw1.Elapsed() << " mS" << std::endl;

	std::vector<real_array> estimated_props(folds);
	real_array sim_scores(folds);

	for ( int f = 0; f < folds; ++f )
	{
		const Data &train = cvdata[f].test;
		const Data &eval = cvdata[f].train;

		real_array weights;
		cre.BestKernel(train, eval, weights);

		std::cout << "\nEval True Theta:\n" << ClassProportions(eval.Labels()) << std::endl;
		std::cout << "\nTrain True Theta:\n" << ClassProportions(train.Labels()) << std::endl;

		DenseMatrix Krr, Ker;

		std::cout << "\nU-L Kernels.." << std::endl;

		Stopwatch sw;
		cre.GetKernels( te_features, tr_features, Ker, bw, weights );
		float re_time = sw.Elapsed();

		std::cout << "\nL-L Kernels.." << std::endl;

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
		std::cout << "\nTrue Prop:\n" << true_prop << std::endl;

		sw.Restart();
		if ( cre.MMD( labels, noClasses, Krr, Ker, estimated_props[f] ) )
		{
			const real_array &prop = estimated_props[f];
			std::cout << "Estimated Prop:\n" << prop << std::endl;
			std::cout << "L1 Norm: " << LpNorm(true_prop, prop, 1) << std::endl;
			std::cout << "L1 Norm Corr: " << L1Score(true_prop, prop) << std::endl;
			std::cout << "Cosine Sim : " << Cosine(true_prop, prop) << std::endl;
			std::cout << "Correlation: " << Correlation(true_prop, prop) << "\n" << std::endl;

			sim_scores[f] = L1Score(true_prop, prop);
		}
		else
		{
			std::cout << "MMD Failed\n" << std::endl;
		}

		float mmd_time = sw.Elapsed();

		std::cerr << "Kers: " << re_time << " mS" << std::endl
				<< "Krrs: " << rr_time << " mS" << std::endl
				<< "MMD:  " << mmd_time << " mS" << std::endl;

	}

	float denom = 0, conf_denom = 0;
	real_array props, conf_props;
	for ( int f = 0; f < folds; ++f )
	{
		real_array prop = estimated_props[f];
		prop *= sim_scores[f];

		if ( sim_scores[f] >= 0.95 )
		{
			conf_denom += sim_scores[f];
			conf_props += prop;
		}

		props += prop;
		denom += sim_scores[f];
	}

	if ( conf_denom )
	{
		conf_props /= conf_denom;
		std::cout << "Estimated Final Prop (Confident): \n" << conf_props << std::endl;
		real_array true_prop = ClassProportions(tedata.Labels());
		std::cout << "L1-score: " << L1Score(conf_props, true_prop) << std::endl;
	}

	props /= denom;
	std::cout << "Estimated Final Prop:\n" << props << std::endl;

	} catch ( std::exception &e ) {
		std::cerr << e.what() << std::endl;
	}

    return 0;
}
