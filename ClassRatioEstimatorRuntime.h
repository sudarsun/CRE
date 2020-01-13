#ifndef CLASSRATIOESTIMATORRUNTIME_H
#define CLASSRATIOESTIMATORRUNTIME_H

#include <string>
#include "ClassRatioEstimator.hpp"
#include "Data.hpp"

#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>


class ClassRatioEstimatorRuntime {
private:
	typedef boost::log::sinks::synchronous_sink< boost::log::sinks::text_file_backend > sink_t;
	boost::shared_ptr< sink_t > g_file_sink;

public:
	explicit ClassRatioEstimatorRuntime(GaussianType inGaussian = eDefault, float inThreshold = 0.95,
										bool useSuperKernel = false,
										int verbosity = 6, int minSize = 50, ScorerType scorer = eL1Scorer)
			: mGaussianType{inGaussian}, mBandwidth(0), mScoreThreshold(inThreshold), mScorer(scorer),
			  mNumClasses(0), mOk(false), mRelativeIndexing(false), mMinSize(minSize) {
		if (useSuperKernel) mCRE.EnableSuperKernels();

		auto loglevel =
				verbosity == 1 ? boost::log::trivial::fatal :
				verbosity == 2 ? boost::log::trivial::error :
				verbosity == 3 ? boost::log::trivial::warning :
				verbosity == 4 ? boost::log::trivial::info :
				verbosity == 5 ? boost::log::trivial::debug : boost::log::trivial::trace;

		boost::log::core::get()->set_filter
				(
						boost::log::trivial::severity >= loglevel
				);

		boost::log::register_simple_formatter_factory<boost::log::trivial::severity_level, char>("Severity");


		g_file_sink = boost::log::add_file_log(
				boost::log::keywords::file_name = "cre-logfile_%N.log",
				boost::log::keywords::rotation_size = 10 * 1024 * 1024,
				boost::log::keywords::open_mode = std::ios_base::app,
				//boost::log::keywords::format = "[%TimeStamp%] [%AppName%] [%ThreadID%] [%Severity%] [%ProcessID%] [%LineID%] [%MyAttr%] [%CountDown%] %Message%"
				boost::log::keywords::format = "%TimeStamp%|%AppName%|%ThreadID%|%ProcessID%|%Severity%|%Message%",
				boost::log::keywords::auto_flush = true
		);

		boost::log::core::get()->add_global_attribute("AppName",
													  boost::log::attributes::constant<std::string>("C.R.E"));
		boost::log::core::get()->add_global_attribute("ThreadID", boost::log::attributes::current_thread_id());
		boost::log::core::get()->add_global_attribute("ProcessID", boost::log::attributes::current_process_id());
		boost::log::add_common_attributes();
	}

	~ClassRatioEstimatorRuntime() {
		boost::log::core::get()->remove_sink(g_file_sink);
		g_file_sink.reset();
	}

	int SetData(const double **inData, int n, int p, const int *inLabels, int inFolds = 5);

	int SetData(const double *inData, int n, int p, const int *inLabels, int inFolds = 5);

	int SetData(const std::string &inDataPath, int inFolds = 5);

	int SetData(const Data &lbl, const Data &ulbl, uint cvfolds = 5, uint lcount = 0);

	int SetData(const Data &lbl, const Data &ulbl, uint cvfolds);

	int SetLabeledData(const std::string &inLabeledDataPath, int inFolds = 5);

	int SetUnlabeledData(const std::string &inUnlabeledDataPath);

	real_array EstimateClassProportions(const Matrix &inFeatures);

	[[nodiscard]] real_array EstimateClassProportions(const int_array &inUnlabeledDataPointIds) const;

	struct CRESession {
		int_array ids;
	};

	static CRESession *createCRESession(const int_array &inUNLIds) {
		auto *session = new CRESession;
		session->ids = inUNLIds;
		return session;
	}

	static void deleteSession(CRESession *inSession) {
		delete inSession;
	}

	PropsAtPivot EstimateClassProportionsInSession(CRESession *inSession, int pivot) const;


	[[nodiscard]] real_array LabeledDataClassProportions() const {
		return mLabeled.classProp();
	}

	[[nodiscard]] real_array UnlabeledDataClassProportions() const {
		return mUnlabeled.classProp();
	}

	[[nodiscard]] real_array UnlabeledDataClassProportions(const int_array &ids) const {
		return mUnlabeled.classProp(ids);
	}

	[[nodiscard]] int LabeledSampleCount() const {
		return mLabeled.size();
	}

	[[nodiscard]] int UnlabeledSampleCount() const {
		return mUnlabeled.size();
	}

	[[nodiscard]] bool isOK() const {
		return mOk;
	}

	[[nodiscard]] int nClasses() const {
		if (mOk) return mNumClasses;

		return 0;
	}

	[[nodiscard]] int nFeatures() const {
		if (mOk) return mLabeled.features().Columns();

		return 0;
	}

	[[nodiscard]] real_array ClassPriors() const {
		if (mOk) return ClassProportions(mLabeled.labels());

		return real_array();
	}

private:

	ClassRatioEstimator mCRE;
	GaussianType mGaussianType;
	float mBandwidth;
	Data mLabeled, mUnlabeled;
	float mScoreThreshold;
	real_array mWeights;
	weights_t mWts;
	ScorerType mScorer;
	int mNumClasses;
	bool mOk;
	DenseMatrix mKrr, mKer;

	bool mRelativeIndexing;
	int mMinSize;

	int SetLabeledDataImpl(int inFolds);

	int SetUnlabeledDataImpl();
};


#endif // CLASSRATIOESTIMATORRUNTIME_H
