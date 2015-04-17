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
// --train <traindata> --predict <test|eval data> --cv <3|5> --cutoff .95 --correlation-threshold 0.9 --use-only-mvkernels --use-only-uvkernels --mvkernel-range -6:6 --log <logfile> --verbosity 1 --load-cv-dataset <cvfile-prefix> --save-cv-dataset <cvfile-prefix>

int main(int argc, char **argv)
{
	if ( argc < 4 )
	{
		std::cerr << "Class Ratio Estimator v2.0 April 2015\n";
		std::cerr << argv[0] << " <tr-data> <val-data> <cvfolds> [cvdata]\n" << std::endl;
		return -1;
	}

	std::ofstream logfile("logfile", std::ios::app );
	try {

		Stopwatch sw1;

	Data trdata;
	std::string filename(argv[1]);
	trdata.Load( filename );

	Data tedata;
	std::string te_fname( argv[2] );
	tedata.Load( te_fname );

	std::cout << "data loaded in: " << sw1.Restart() << " mS" << std::endl;

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


	{
		logfile << tr_labels.Rows() << "\t";
		real_array trprop = ClassProportions(tr_labels);
		logfile << trprop[0] << "\t";
		logfile << te_features.Rows() << "\t";
		real_array teprop = ClassProportions(tedata.Labels());
		logfile << teprop[0] << "\t";
	}

	ClassRatioEstimator cre;
	float bw = cre.BandwidthSelect( tr_features );

	int folds = atoi(argv[3]);

	cvdata_t cvdata;

	if ( argc > 4 )
	{
		std::cout << "loading CV data from: " << argv[4] << std::endl;
		Data::LoadCrossValidationDataSet( argv[4], cvdata );
		std::cout << "cross validation dataset loaded in: " << sw1.Restart() << " mS" << std::endl;
	}
	else
	{
		trdata.GetCrossValidationDataSet(folds, cvdata);
		std::cout << "cross validation dataset constructed: " << sw1.Restart() << " mS" << std::endl;

		std::cout << "saving CV data onto " << argv[1] << ".cv3 ...";
		Data::SaveCrossValidationDataSet( cvdata, std::string(argv[1])+".cv3" );
		std::cout << "saved in " << sw1.Restart() << " mS" << std::endl;
	}

	std::vector<real_array> estimated_props(folds);
	real_array sim_scores(folds);

	for ( int f = 0; f < folds; ++f )
	{
		const Data &train = cvdata[f].test;
		const Data &eval = cvdata[f].train;

		real_array weights;
		cre.BestKernel(train, eval, weights, eModifiedBinaryL1Scorer);

		std::cout << "\nEval True Theta:\n" << ClassProportions(eval.Labels()) << std::endl;
		std::cout << "\nTrain True Theta:\n" << ClassProportions(train.Labels()) << std::endl;

		DenseMatrix Krr, Ker;

		std::cout << "\nU-L Kernels.." << std::endl;

		Stopwatch sw;
		cre.GetKernels( te_features, tr_features, Ker, bw, weights, "" );
		float re_time = sw.Elapsed();

		std::cout << "\nL-L Kernels.." << std::endl;

		sw.Restart();
		cre.GetKernels( tr_features, tr_features, Krr, bw, weights, "" );
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
			std::cout << "L1 Score: " << L1Score(true_prop, prop) << std::endl;
			std::cout << "ModL1 Score: " << ModifiedBinaryL1Score(true_prop, prop) << std::endl;
			std::cout << "Cosine Sim : " << Cosine(true_prop, prop) << std::endl;
			std::cout << "Correlation: " << Correlation(true_prop, prop) << "\n" << std::endl;

			sim_scores[f] = ModifiedBinaryL1Score(true_prop, prop); // L1Score(true_prop, prop);
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

	real_array true_prop = ClassProportions(tedata.Labels());
	if ( conf_denom )
	{
		conf_props /= conf_denom;
		std::cout << "Estimated Final Prop (Confident): \n" << conf_props << std::endl;
		double cscore = ModifiedBinaryL1Score(conf_props, true_prop); // L1Score(conf_props, true_prop);
		std::cout << "L1-score: " << cscore << std::endl;
		logfile << conf_props[0] << "\t" << cscore << "\t";
	}

	props /= denom;
	std::cout << "Estimated Final Prop:\n" << props << std::endl;
	double cscore = ModifiedBinaryL1Score(props, true_prop); // L1Score(props, true_prop);
	std::cout << "L1-score: " << cscore << std::endl;

	if ( !conf_denom )
		logfile << props[0] << "\t" << cscore << "\t";

	logfile << std::endl;

	} catch ( std::exception &e ) {
		std::cerr << e.what() << std::endl;
		logfile << "ERROR: " << e.what() << std::endl;
	}

    return 0;
}
