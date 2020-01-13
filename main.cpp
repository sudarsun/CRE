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
#include "ClassRatioEstimatorRuntime.h"
#include <CLI/CLI.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/support/date_time.hpp>

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

const float kCorrelationThreshold_ModifiedBinaryL1 = 0.90;
const float kCorrelationThreshold_L1 = 0.95f;

void print(double *ptr, int n, int p)
{
	for (int i = 1; i < 10; ++i)
	{
		for (int j = 0; j < 5; ++j)
			printf("%f ", ptr[i*p+j]);

		printf("\n");
	}

	printf("\n\n");
}

static std::string vec2str(real_array &vec) {
	std::string out;
	for (auto v : vec) {
		char str[100];
		snprintf(str, 100, "%f", v);

		out += " ";
		out += str;
	}

	return out;
}

int main(int argc, char **argv)
{
	CLI::App app{"Class Ratio Estimator v5 Nov 2019"};

	std::string trainfile{};
	app.add_option("-r,--train", trainfile, "Labeled training data in libsvm or csv format")->required()->check(CLI::ExistingFile);
	std::string evalfile{};
	app.add_option("-e,--evaluate", evalfile, "Labeled data in libsvm or csv format for evaluation")->check(CLI::ExistingFile);
	std::string testfile{};
	app.add_option("-p,--predict", testfile, "Unlabeled data in libsvm or csv format to make class ratio estimation")->check(CLI::ExistingFile);
	app.add_flag("-u,--use-only-univariate-kernels", "Force usage of univariate kernels only");
	app.add_flag("-m,--use-only-multivariate-kernels", "Force usage of multivariate kernels only");
	std::string logfilename{"logfile"};
	app.add_option("-l,--logfile", logfilename, "Log file path"); //->check(CLI::ExistingFile);
	int verbosity = 6;
	app.add_option("-V,--verbosity", verbosity, "verbosity level [1-6]")->check(CLI::Range(1,6));
	int trainsize = 100;
	app.add_option("-s,--train-size", trainsize, "choose training size in percentage [5-100]")->check(CLI::Range(5,100));
	int cvfolds{5};
	app.add_option("-v,--crossval", cvfolds, "Number of cross-validation folds (>=3)")->check(CLI::Range(3,10));
	float threshold{kCorrelationThreshold_L1};
	app.add_option("-c,--corr-threshold", threshold, "Correlation threshold to use while selecting kernels")->check(CLI::Range(0.8,0.99));

	app.add_flag("-S,--vary-training-size", "Test CRE under different training sizes" );
	app.add_flag("-P,--vary-training-props", "Test CRE under different training class proportions");
	app.add_flag("-k,--use-super-kernel", "Enable usage of Super Kernel");

	CLI11_PARSE(app, argc, argv)

	// get the value of verbosity argument.
	//verbosity = app.get_option("-V")->as<int>();

	auto loglevel =
			verbosity == 1 ? boost::log::trivial::fatal :
			verbosity == 2 ? boost::log::trivial::error :
			verbosity == 3 ? boost::log::trivial::warning :
			verbosity == 4 ? boost::log::trivial::info :
			verbosity == 5 ? boost::log::trivial::debug : boost::log::trivial::trace;

	auto gaussianType =
			app.get_option("-u")->count() >= 1 ? eUnivariateOnly :
			app.get_option("-m")->count() >= 1 ? eMultivariateOnly : eDefault;

	auto useSuperKernel = app.get_option("-k")->count() >= 1;

	bool isPredict = app.get_option("-p")->empty() == 0;

	auto varyTrainSizeFlag = app.get_option("-S")->count() >= 1;
	//auto varyTrainPropFlag = app.get_option("-P")->count() >= 1;

    //boost::log::core::get()->add_global_attribute("Scope", boost::log::attributes::named_scope());
    //boost::log::core::get()->add_global_attribute("CountDown", boost::log::attributes::counter<int>(100, -1));

	boost::log::core::get()->set_filter
	(
		boost::log::trivial::severity >= loglevel
	);

    boost::log::register_simple_formatter_factory<boost::log::trivial::severity_level, char>("Severity");
    boost::log::add_file_log(
            boost::log::keywords::file_name = logfilename + "_%N.log",
            boost::log::keywords::rotation_size = 10 * 1024 * 1024,
            boost::log::keywords::open_mode = std::ios_base::app,
            //boost::log::keywords::format = "[%TimeStamp%] [%AppName%] [%ThreadID%] [%Severity%] [%ProcessID%] [%LineID%] [%MyAttr%] [%CountDown%] %Message%"
            boost::log::keywords::format = "%TimeStamp%|%AppName%|%ThreadID%|%ProcessID%|%Severity%|%Message%",
            boost::log::keywords::auto_flush = true
    );

    boost::log::core::get()->add_global_attribute("AppName", boost::log::attributes::constant<std::string>("C.R.E"));
    boost::log::add_common_attributes();


	try {

	    BOOST_LOG_TRIVIAL(info) << "***** starting the application *****";

        Stopwatch sw1;

        BOOST_LOG_TRIVIAL(trace) << "loading the labeled dataset";
        Data lbl{};
        lbl.Load(trainfile);

        Data ulbl{};
        if (isPredict) {
            BOOST_LOG_TRIVIAL(trace) << "loading the prediction dataset";
            ulbl.Load(testfile);
        }
        else {
            BOOST_LOG_TRIVIAL(trace) << "loading the evaluation dataset";
            ulbl.Load(evalfile);
        }

        if (varyTrainSizeFlag) {
            auto maxTrainSize = lbl.size();

            BOOST_LOG_TRIVIAL(info) << "varying labeled data size in terms of percentage";
            for (int p = 100; p > 5; p-=10) {
                BOOST_LOG_TRIVIAL(info) << "using " << p << "% of the training data";
                ClassRatioEstimatorRuntime cre_rt(gaussianType, threshold, useSuperKernel);
                cre_rt.SetData(lbl, ulbl, cvfolds, maxTrainSize*p/100);

                int count = cre_rt.UnlabeledSampleCount();
                int_array absolute_ids((ulong)count);
                for (int i = 0; i < count; ++i) {
                    absolute_ids[i] = cre_rt.LabeledSampleCount() + i;
                }

                BOOST_LOG_TRIVIAL(info) << "Estimating class proportions for the unlabeled dataset...";
                real_array prps2 = cre_rt.EstimateClassProportions(absolute_ids);
                BOOST_LOG_TRIVIAL(info) << "estimated: " << vec2str(prps2);

                if (!isPredict) {
                    auto evprop{ulbl.classProp()};
                    BOOST_LOG_TRIVIAL(info) << "true: " << vec2str(evprop);
                    BOOST_LOG_TRIVIAL(info) << "score (L1): " << Score(evprop, prps2, eL1Scorer);
                }
            }

            BOOST_LOG_TRIVIAL(info) << "varying labeled data size in terms of absolute count";

            for (int p = 100; p < maxTrainSize/10; p*=2) {
                BOOST_LOG_TRIVIAL(info) << "using " << p << " data points of the training data";
                ClassRatioEstimatorRuntime cre_rt(gaussianType, threshold, useSuperKernel);
                cre_rt.SetData(lbl, ulbl, cvfolds, p);

                int count = cre_rt.UnlabeledSampleCount();
                int_array absolute_ids((ulong)count);
                for (int i = 0; i < count; ++i) {
                    absolute_ids[i] = cre_rt.LabeledSampleCount() + i;
                }

                BOOST_LOG_TRIVIAL(info) << "Estimating class proportions for the unlabeled dataset...";
                real_array prps2 = cre_rt.EstimateClassProportions(absolute_ids);
                BOOST_LOG_TRIVIAL(info) << "estimated: " << vec2str(prps2);

                if (!isPredict) {
                    auto evprop{ulbl.classProp()};
                    BOOST_LOG_TRIVIAL(info) << "true: " << vec2str(evprop);
                    BOOST_LOG_TRIVIAL(info) << "score (L1): " << Score(evprop, prps2, eL1Scorer);
                }
            }

            return 0;
        }


        BOOST_LOG_TRIVIAL(trace) << "initializing Class Ratio Estimator";
        ClassRatioEstimatorRuntime cre_rt(gaussianType, threshold, useSuperKernel);

        auto evprop{ulbl.classProp()};

        //if ()
        //
        cre_rt.SetData(lbl, ulbl, cvfolds, lbl.size()*trainsize/100);
/*
        double *l_features = lbl.features().RawData();
        int ln = lbl.features().mRows;
        int lp = lbl.features().mCols;

        double *l_labels = lbl.labels().RawData();

        double *ul_features = ulbl.features().RawData();
        int uln = ulbl.features().mRows;
        int ulp = ulbl.features().mCols;
        double *ul_labels = ulbl.labels().RawData();

        auto *features = new double[(ln+uln)*lp];
        for (int i = 0; i < ln*lp; ++i)
            features[i] = l_features[i];
        for (int i = 0; i < uln*lp; ++i)
            features[ln*lp+i] = ul_features[i];

        int *labels = new int[ln+uln];
        for (int i = 0; i < ln; ++i)
            labels[i] = (int)l_labels[i];

        // forcibly resetting the labels for evaluation dataset to emulate unlabeled dataset.
        for (int i = ln, j = 0; i < (ln+uln); ++i, ++j)
            labels[i] = 0;//(int)ul_labels[j];

        if (cre_rt.SetData(features, ln+uln, lp, labels, cvfolds) == -1)
        {
            BOOST_LOG_TRIVIAL(error) << "can't apply CRE when unlabeled data is empty\n" << std::endl;
            return -1;
        }
*/
        int count = cre_rt.UnlabeledSampleCount();
        int_array absolute_ids((ulong)count);
        for (int i = 0; i < count; ++i) {
            absolute_ids[i] = cre_rt.LabeledSampleCount() + i;
        }

        BOOST_LOG_TRIVIAL(info) << "Estimating Class Proportions for the entire unlabeled dataset...";
        real_array prps2 = cre_rt.EstimateClassProportions(absolute_ids);
        BOOST_LOG_TRIVIAL(info) << "estimated: " << vec2str(prps2);
        std::cerr << "Estimated: " << vec2str(prps2) << std::endl;

        if (!isPredict) {
            BOOST_LOG_TRIVIAL(info) << "true: " << vec2str(evprop);
            BOOST_LOG_TRIVIAL(info) << "score (L1): " << Score(evprop, prps2, eL1Scorer);
            //BOOST_LOG_TRIVIAL(info) << "score (wL1): " << Score(evprop, prps2, eModifiedBinaryL1Scorer);
        }

        /*
        for (int i = 100; i > 0; i-=5)
        {
            int_array ids = absolute_ids;
            std::random_shuffle(ids.begin(), ids.end());

            int newCount = count * i / 100;
            ids.resize((ulong)newCount);

            int_array relative_ids((ulong)newCount);
            for (int j = 0; j < newCount; ++j)
                relative_ids[j] = ids[j] - cre_rt.LabeledSampleCount();

            //real_array true_prop = cre_rt.UnlabeledDataClassProportions(ids);
            real_array true_prop = ulbl.classProp(relative_ids);
            real_array est_prop = cre_rt.EstimateClassProportions(ids);

            float score = Score(true_prop, est_prop, eL1Scorer);

            std::cout << "size: " << newCount << " (" << i << "%)" << " True: " << true_prop << " Eval: " << est_prop << " Score: " << score << std::endl;
        }*/

    /*
        Data trdata;
        std::string filename(argv[1]);
        trdata.Load( filename );

        Data tedata;
        std::string te_fname( argv[2] );
        tedata.Load( te_fname );

        std::cout << "data loaded in: " << sw1.Restart() << " mS" << std::endl;





        int minLabel = trdata.minLabel();
        int maxLabel = trdata.maxLabel();

        if ( minLabel <= 0 )
        {
            std::cerr << "minLabel=" << minLabel << ": labels are not positive, consider reordering the class labels!\n" << std::endl;
            return -1;
        }

        const Matrix &tr_features = trdata.features();
        const Matrix &tr_labels = trdata.labels();
        const Matrix &te_features = tedata.features();


        {
            logfile << tr_labels.Rows() << "\t";
            real_array trprop = ClassProportions(tr_labels);
            logfile << trprop[0] << "\t";
            logfile << te_features.Rows() << "\t";
            real_array teprop = ClassProportions(tedata.labels());
            logfile << teprop[0] << "\t";
        }

        ClassRatioEstimator cre;
        float bw = cre.BandwidthSelect( tr_features );

        int folds = atoi(argv[3]);

        cvdata_t cvdata;
        ScorerType scorerType = eL1Scorer; //eModifiedBinaryL1Scorer;
        float scorerThreshold = kCorrelationThreshold_L1;

        if ( argc > 4 )
        {
            std::cout << "loading CV data from: " << argv[4] << std::endl;
            Data::LoadCrossValidationDataSet( argv[4], cvdata );
            std::cout << "cross validation dataset loaded in: " << sw1.Restart() << " mS" << std::endl;
        }
        else
        {
            trdata.getCrossValidationDataSet(folds, cvdata);
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
            cre.BestKernel(train, eval, weights, scorerType, scorerThreshold);

            std::cout << "\nEval True Theta:\n" << ClassProportions(eval.labels()) << std::endl;
            std::cout << "\nTrain True Theta:\n" << ClassProportions(train.labels()) << std::endl;

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
                throw std::runtime_error("label cannot have '0'");

                labels += 1;
                noClasses = maxLabel + 1;  // including the 0 as a label.
            }

            real_array true_prop = ClassProportions(tedata.labels());
            std::cout << "\nTrue Prop:\n" << true_prop << std::endl;

            sw.Restart();
            bool isOverfit = false;
            if ( cre.MMD( labels, noClasses, Krr, Ker, estimated_props[f], isOverfit ) )
            {
                const real_array &prop = estimated_props[f];
                std::cout << "Estimated Prop:\n" << prop << std::endl;
                std::cout << "L1 Norm: " << LpNorm(true_prop, prop, 1) << std::endl;
                std::cout << "L1 Score: " << L1Score(true_prop, prop) << std::endl;
                std::cout << "ModL1 Score: " << ModifiedBinaryL1Score(true_prop, prop) << std::endl;
                std::cout << "Cosine Sim : " << Cosine(true_prop, prop) << std::endl;
                std::cout << "Correlation: " << Correlation(true_prop, prop) << "\n" << std::endl;

                sim_scores[f] = Score(true_prop, prop, scorerType);   //ModifiedBinaryL1Score(true_prop, prop); // L1Score(true_prop, prop);
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

            //if ( sim_scores[f] >= 0.95 )
            if ((scorerType == eModifiedBinaryL1Scorer && sim_scores[f] >= kCorrelationThreshold_ModifiedBinaryL1) ||
                (scorerType == eL1Scorer && sim_scores[f] >= kCorrelationThreshold_L1) 	)
            {
                conf_denom += sim_scores[f];
                conf_props += prop;
            }

            props += prop;
            denom += sim_scores[f];
        }

        real_array true_prop = ClassProportions(tedata.labels());
        if ( conf_denom )
        {
            conf_props /= conf_denom;
            std::cout << "Estimated Final Prop (Confident): \n" << conf_props << std::endl;
            double cscore = Score(conf_props, true_prop, scorerType); //ModifiedBinaryL1Score(conf_props, true_prop); // L1Score(conf_props, true_prop);
            std::cout << "Score: " << cscore << std::endl;
            logfile << conf_props[0] << "\t" << cscore << "\t";
        }

        props /= denom;
        std::cout << "Estimated Final Prop:\n" << props << std::endl;
        double cscore = Score(conf_props, true_prop, scorerType); // ModifiedBinaryL1Score(props, true_prop); // L1Score(props, true_prop);
        std::cout << "Score: " << cscore << std::endl;

        if ( !conf_denom )
            logfile << props[0] << "\t" << cscore << "\t";

        logfile << std::endl;
    */
	} catch ( std::exception &e ) {
        std::cerr << "ERROR: " << e.what() << std::endl;
		BOOST_LOG_TRIVIAL(error) << "ERROR: " << e.what() << std::endl;
        BOOST_LOG_TRIVIAL(info) << "***** application terminated abnormally *****";
        return -1;
	}

	BOOST_LOG_TRIVIAL(info) << "***** application terminated normally *****";

    return 0;
}
