#include "ClassRatioEstimatorRuntime.h"
#include <boost/thread.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>

int ClassRatioEstimatorRuntime::SetData(const std::string &inDataPath, int inFolds) {
	Data data;
	data.Load(inDataPath);

	// CRE is useful only when unlabeled data is present in the dataset.
	Data *unlabeled = data.getUnlabeled();
	if (!unlabeled->isEmpty()) {
		mLabeled = *data.getLabeled();
		mUnlabeled = *unlabeled;

		std::cout << "labeled data points: " << mLabeled.size() << std::endl;
		std::cout << "unlabeled data points: " << mUnlabeled.size() << std::endl;

		SetLabeledDataImpl(inFolds);
		SetUnlabeledDataImpl();

		mRelativeIndexing = true;
		mOk = true;
		return 0;
	} else {
		std::cerr << "can't continue with CRE, as there are no unlabeled datapoints!!\n" << std::endl;
	}

	return -1;
}

int ClassRatioEstimatorRuntime::SetData(const double **inData, int n, int p, const int *inLabels, int inFolds) {
	Data data(inData, n, p, inLabels);

	// CRE is useful only when unlabeled data is present in the dataset.
	Data *unlabeled = data.getUnlabeled();
	if (!unlabeled->isEmpty()) {
		mLabeled = *data.getLabeled();
		mUnlabeled = *unlabeled;

		std::cout << "labeled data points: " << mLabeled.size() << std::endl;
		std::cout << "unlabeled data points: " << mUnlabeled.size() << std::endl;

		SetLabeledDataImpl(inFolds);
		SetUnlabeledDataImpl();

		mRelativeIndexing = true;
		mOk = true;

		return 0;
	} else {
		std::cerr << "can't continue with CRE, as there are no unlabeled datapoints!!\n" << std::endl;
	}

	return -1;
}

/// Assigns a random subset of data of size inLblCount from labeled set.
int ClassRatioEstimatorRuntime::SetData(const Data &lbl, const Data &ulbl, uint cvfolds, uint inLblCount) {

	//auto rawLabeled = std::make_unique<double>(lbl.features().RawData());
	double *l_features = /*rawLabeled.get();*/ lbl.features().RawData();

	int ln = lbl.features().mRows;
	int lp = lbl.features().mCols;

	double *l_labels = lbl.labels().RawData();

	double *ul_features = ulbl.features().RawData();
	int uln = ulbl.features().mRows;
	int ulp = ulbl.features().mCols;
	double *ul_labels = ulbl.labels().RawData();

	// randomly permute the id list, so that we can pick the top 'inLblCount' items from it.
	auto ids = std::vector<int>(ln);
	for (int i = 0; i < ln; ++i) ids[i] = i;
	std::random_shuffle(ids.begin(), ids.end());

	// create a placeholder for the sample.
	auto *features = new double[(inLblCount + uln) * lp];

	auto nz = 0;
	for (int i = 0; i < inLblCount; ++i)
		for (int j = 0; j < lp; ++j) {
			auto v = l_features[ids[i] * lp + j];
			if (!v) ++nz;
			else features[i * lp + j] = v; //l_features[ids[i] * lp + j];
		}

	for (int i = 0; i < uln * lp; ++i)
		features[inLblCount * lp + i] = ul_features[i];

	int *labels = new int[inLblCount + uln];
	for (int i = 0; i < inLblCount; ++i)
		labels[i] = (int) l_labels[ids[i]];

	// forcibly resetting the labels for evaluation dataset to emulate unlabeled dataset.
	for (int i = inLblCount, j = 0; i < (inLblCount + uln); ++i, ++j)
		labels[i] = 0;//(int)ul_labels[j];

	if (SetData(features, inLblCount + uln, lp, labels, cvfolds) == -1) {
		BOOST_LOG_TRIVIAL(error) << "can't apply CRE when unlabeled data is empty\n" << std::endl;
		return -1;
	}

	return 0;
}

int ClassRatioEstimatorRuntime::SetData(const Data &lbl, const Data &ulbl, uint cvfolds) {
	double *l_features = lbl.features().RawData();
	int ln = lbl.features().mRows;
	int lp = lbl.features().mCols;

	double *l_labels = lbl.labels().RawData();

	double *ul_features = ulbl.features().RawData();
	int uln = ulbl.features().mRows;
	int ulp = ulbl.features().mCols;
	double *ul_labels = ulbl.labels().RawData();

	auto *features = new double[(ln + uln) * lp];
	for (int i = 0; i < ln * lp; ++i)
		features[i] = l_features[i];
	for (int i = 0; i < uln * lp; ++i)
		features[ln * lp + i] = ul_features[i];

	int *labels = new int[ln + uln];
	for (int i = 0; i < ln; ++i)
		labels[i] = (int) l_labels[i];

	// forcibly resetting the labels for evaluation dataset to emulate unlabeled dataset.
	for (int i = ln, j = 0; i < (ln + uln); ++i, ++j)
		labels[i] = 0;//(int)ul_labels[j];

	if (SetData(features, ln + uln, lp, labels, cvfolds) == -1) {
		BOOST_LOG_TRIVIAL(error) << "can't apply CRE when unlabeled data is empty\n" << std::endl;
		return -1;
	}

	return 0;
}

int ClassRatioEstimatorRuntime::SetData(const double *inData, int n, int p, const int *inLabels, int inFolds) {
	Data data(inData, n, p, inLabels);

	// CRE is useful only when unlabeled data is present in the dataset.
	Data *unlabeled = data.getUnlabeled();
	if (!unlabeled->isEmpty()) {
		mLabeled = *data.getLabeled();
		mUnlabeled = *unlabeled;

		BOOST_LOG_TRIVIAL(info) << "labeled data points: " << mLabeled.size();
		BOOST_LOG_TRIVIAL(info) << "unlabeled data points: " << mUnlabeled.size();
		BOOST_LOG_TRIVIAL(info) << "dimensionality: " << p;

		BOOST_LOG_TRIVIAL(debug) << "loading the labeled data points..";
		SetLabeledDataImpl(inFolds);
		BOOST_LOG_TRIVIAL(debug) << "loading the unlabeled data points..";
		SetUnlabeledDataImpl();

		BOOST_LOG_TRIVIAL(debug) << "relative indexing is set to true..";
		mRelativeIndexing = true;
		mOk = true;

		return 0;
	} else {
		BOOST_LOG_TRIVIAL(error) << "can't continue with CRE, as there are no unlabeled datapoints!!\n" << std::endl;
	}

	BOOST_LOG_TRIVIAL(info) << "ClassRatioEstimatorRuntime::SetData error_at_exit";
	return -1;
}

int ClassRatioEstimatorRuntime::SetLabeledData(const std::string &inLabeledDataPath, int inFolds) {
	// load the training data.
	mLabeled.Load(inLabeledDataPath);

	mRelativeIndexing = false;
	return SetLabeledDataImpl(inFolds);
}

int ClassRatioEstimatorRuntime::SetLabeledDataImpl(int inFolds) {
	int maxLabel = mLabeled.maxLabel();
	int minLabel = mLabeled.minLabel();

	if (minLabel == 0)
		throw std::runtime_error("label cannot have '0'");


	mNumClasses = maxLabel;
	DenseMatrix labels = dynamic_cast<const DenseMatrix &>(mLabeled.labels());
	/*if ( minLabel == 0 )
	{
		throw std::runtime_error("label cannot have '0'");

		labels += 1;
		mNumClasses = maxLabel + 1;  // including the 0 as a label.
	}*/

	// set the training corpus bandwidth.
	mBandwidth = mCRE.BandwidthSelect(mLabeled.features());

	// prepare the cross validation dataset.
	cvdata_t cvdata;
	mLabeled.getCrossValidationDataSet(inFolds, cvdata);

	weights_t wghts;
	//real_array weights;
	for (int f = 0; f < inFolds; ++f) {
		BOOST_LOG_TRIVIAL(debug) << "Running fold " << (f + 1) << " of " << inFolds;
		const Data &train = cvdata[f].test;
		const Data &eval = cvdata[f].train;

		//real_array wts;
		//mCRE.BestKernel(train, eval, wts, mScorer, mScoreThreshold);

		weights_t wts2;
		mCRE.BestKernelv2(train, eval, wts2, mGaussianType, mScorer, mScoreThreshold);

		wghts += wts2;
		//weights += wts;
	}

	mWts = wghts;
	//mWeights = weights;

	return 0;
}

int ClassRatioEstimatorRuntime::SetUnlabeledData(const std::string &inUnlabeledDataPath) {
	BOOST_LOG_TRIVIAL(info) << "Loading unlabeled data..." << std::endl;
	mUnlabeled.Load(inUnlabeledDataPath);

	mRelativeIndexing = false;

	return SetUnlabeledDataImpl();
}

int ClassRatioEstimatorRuntime::SetUnlabeledDataImpl() {
	//std::cerr << "Estimating Ker..." << std::endl;
	mCRE.GetKernelsv2(mUnlabeled.features(), mLabeled.features(), mKer, mGaussianType, mBandwidth, mWts);

	//std::cerr << "Estimating Krr..." << std::endl;
	mCRE.GetKernelsv2(mLabeled.features(), mLabeled.features(), mKrr, mGaussianType, mBandwidth, mWts);

	mOk = true;
}

std::string toString(const real_array &prop) {
	std::stringstream ss;
	ss << "[";
	for (auto p : prop) {
		ss << " " << p;
	}
	ss << " ]";

	return ss.str();
}

PropsAtPivot ClassRatioEstimatorRuntime::EstimateClassProportionsInSession(CRESession *inSession, int pivot) const {
	PropsAtPivot result;
	size_t sessionDataLength = inSession->ids.size();

	if (inSession == nullptr)
		BOOST_LOG_TRIVIAL(fatal) << "CRE Session not created.";
	else if (pivot > sessionDataLength)
		BOOST_LOG_TRIVIAL(fatal) << "Pivot point is beyond the data array!";
	else if (pivot == sessionDataLength) {
		// entire data is on the left.
		result.left = EstimateClassProportions(inSession->ids);
		result.right = real_array(result.left.size(), 0.0);
	} else if (pivot == 0) {
		// entire data is on the right.
		result.right = EstimateClassProportions(inSession->ids);
		result.left = real_array(result.right.size(), 0.0);
	} else {
		auto const &data = inSession->ids;
		int_array left(data.begin(), data.begin() + pivot), right(data.begin() + pivot, data.end());
		result.left = pivot <= mMinSize ? ClassPriors() : EstimateClassProportions(left);
		result.right = (sessionDataLength - pivot) <= mMinSize ? ClassPriors() : EstimateClassProportions(right);
	}

	return result;
}

real_array ClassRatioEstimatorRuntime::EstimateClassProportions(const int_array &inUnlabeledDataPointIds) const {
	// set the spread probability for the estimated proportions.
	real_array estimated_props((uword) mNumClasses, float(1.0 / mNumClasses));
	if (mMinSize >= inUnlabeledDataPointIds.size())
		return estimated_props;

	int_array dpts = inUnlabeledDataPointIds;
	if (mRelativeIndexing) {
		int labeledCount = mLabeled.size();
		for (int i = 0; i < inUnlabeledDataPointIds.size(); ++i)
			dpts[i] = inUnlabeledDataPointIds[i] - labeledCount;
	}

	if (mCRE.isCached()) {
		if (!mCRE.MMD_2(mNumClasses, dpts, estimated_props))
			BOOST_LOG_TRIVIAL(error) << "MMD failed";
		BOOST_LOG_TRIVIAL(trace) << "estimate proportion (" << inUnlabeledDataPointIds.size() << ") "
								 << toString(estimated_props);
	} else {
		BOOST_LOG_TRIVIAL(trace) << "estimate proportion (" << inUnlabeledDataPointIds.size() << ") "
								 << toString(estimated_props);

		const DenseMatrix &ker = mKer.Select(dpts, int_array());
		bool isOverfit = false;
		if (!const_cast<ClassRatioEstimator &>(mCRE).MMD(dynamic_cast<const DenseMatrix &>(mLabeled.labels()),
														 mNumClasses, mKrr, ker, estimated_props, isOverfit, true)) {
			BOOST_LOG_TRIVIAL(warning) << (isOverfit ? "MMD Overfit" : "MMD failed") << " ("
									   << inUnlabeledDataPointIds.size() << ")";
			return ClassPriors();
		} else if (isOverfit) {
			// return default class priors instead of spread probability.
			BOOST_LOG_TRIVIAL(warning) << "MMD Overfit (" << inUnlabeledDataPointIds.size() << ")";
			return ClassPriors();
		} else
			BOOST_LOG_TRIVIAL(debug) << "All is well (" << inUnlabeledDataPointIds.size() << ")";
	}

	return estimated_props;
}


real_array ClassRatioEstimatorRuntime::EstimateClassProportions(const Matrix &inFeatures) {
	// set the spread probability for the estimated proportions.
	real_array estimated_props((uword) mNumClasses, float(1.0 / mNumClasses));
	if (mMinSize >= inFeatures.Rows())
		return estimated_props;

	DenseMatrix Krr, Ker;

	mCRE.GetKernelsv2(inFeatures, mLabeled.features(), Ker, mGaussianType, mBandwidth, mWts);
	mCRE.GetKernelsv2(mLabeled.features(), mLabeled.features(), Krr, mGaussianType, mBandwidth, mWts);

	bool isOverfit = false;
	if (!mCRE.MMD(dynamic_cast<const DenseMatrix &>(mLabeled.labels()), mNumClasses, Krr, Ker, estimated_props,
				  isOverfit))
		BOOST_LOG_TRIVIAL(error) << "MMD failed";

	if (isOverfit) {
		BOOST_LOG_TRIVIAL(error) << "MMD Overfit (" << inFeatures.Rows() << ")";
		return ClassPriors();
	}

	return estimated_props;
}

