#include "ClassRatioEstimatorJNI.h"
#include "ClassRatioEstimatorRuntime.h"


#define CORR_THRESHOLD_L1   0.9
#define DONT_USE_SUPERKERNEL    false
#define GAUSSIAN_TYPE   eMultivariateOnly

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    buildJNI
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_buildJNI__Ljava_lang_String_2Ljava_lang_String_2
		(JNIEnv *env, jobject, jstring labeled, jstring unlabeled) {
	auto *cre_rt = new ClassRatioEstimatorRuntime(GAUSSIAN_TYPE, CORR_THRESHOLD_L1, DONT_USE_SUPERKERNEL, 4);

	const char *labeled_fname = env->GetStringUTFChars(labeled, nullptr);
	const char *unlabeled_fname = env->GetStringUTFChars(unlabeled, nullptr);

	try {
		cre_rt->SetLabeledData(labeled_fname);
		env->ReleaseStringUTFChars(labeled, labeled_fname);

		cre_rt->SetUnlabeledData(unlabeled_fname);
		env->ReleaseStringUTFChars(unlabeled, unlabeled_fname);
	}
	catch (std::exception &e) {
		env->ReleaseStringUTFChars(labeled, labeled_fname);
		env->ReleaseStringUTFChars(unlabeled, unlabeled_fname);

		return env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
	catch (...) {
		env->ReleaseStringUTFChars(labeled, labeled_fname);
		env->ReleaseStringUTFChars(unlabeled, unlabeled_fname);

		return env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
							 "unknown exception within CRE runtime while building the model!");
	}

	// don't know why this is done, I just copied the way CRF++ JNI wrapper was implemented.
	jlong result;
	*(ClassRatioEstimatorRuntime **) &result = cre_rt;

	return result;
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    buildJNI
 * Signature: ([DII[I)J
 */
JNIEXPORT jlong JNICALL
Java_ai_buddi_ml_cre_ClassRatioEstimator_buildJNI___3DII_3I(JNIEnv *env, jobject, jdoubleArray dataArray, jint n,
															jint p, jintArray labelsArray) {
	auto *cre_rt = new ClassRatioEstimatorRuntime(GAUSSIAN_TYPE, CORR_THRESHOLD_L1, DONT_USE_SUPERKERNEL, 4);

	jboolean isCopy;
	jdouble *data = env->GetDoubleArrayElements(dataArray, &isCopy);
	jint *labels = env->GetIntArrayElements(labelsArray, &isCopy);

	try {
		if (cre_rt->SetData(data, n, p, labels, 5) == -1) {
			std::cerr << "unlabeled data not available, CRE is unloaded now!" << std::endl;
		}
	}
	catch (std::exception &e) {
		return env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
	catch (...) {
		return env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
							 "unknown exception within CRE runtime while building the model!");
	}

	// don't know why this is done, I just copied the way CRF++ JNI wrapper was implemented.
	jlong result;
	*(ClassRatioEstimatorRuntime **) &result = cre_rt;

	return result;
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    deleteJNI
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_deleteJNI(JNIEnv *, jobject, jlong ref) {
	if (ref != 0) {
		auto *cre_rt = (ClassRatioEstimatorRuntime *) ref;
		delete cre_rt;
	}
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    isOK
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_isOK(JNIEnv *, jobject, jlong ref) {
	return ref != 0 && ((ClassRatioEstimatorRuntime *) ref)->isOK();
}


/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    nFeatures
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_nFeatures
		(JNIEnv *, jobject, jlong ref) {

	// if the reference is ok, return the number of dimensions.
	if (ref != 0) return ((ClassRatioEstimatorRuntime *) ref)->nFeatures();

	return 0L;
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    nClasses
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_nClasses
		(JNIEnv *, jobject, jlong ref) {

	// if the reference is ok, return the number of classes.
	if (ref != 0) return ((ClassRatioEstimatorRuntime *) ref)->nClasses();

	return 0L;
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    classPriors
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_classPriors
		(JNIEnv *env, jobject, jlong ref) {
	if (ref == 0) return env->NewDoubleArray(0);

	try {
		// if the reference is ok, return the class priors of the labeled dataset.
		auto *cre_rt = (ClassRatioEstimatorRuntime *) ref;
		if (!cre_rt->isOK())
			env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "CRE subsystem is not initialized");

		real_array priors = cre_rt->ClassPriors();
		jdoubleArray result = env->NewDoubleArray(priors.size());

		size_t psize = priors.size();
		auto *array = new jdouble[psize];
		for (size_t i = 0; i < psize; ++i) array[i] = priors[i];

		env->SetDoubleArrayRegion(result, 0, psize, array);

		return result;
	}
	catch (std::exception &e) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
	catch (...) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
					  "unknown exception within CRE subsystem while fetching labeled data class priors!");
	}
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioJNI
 * Signature: (J[III)[D
 */
JNIEXPORT jdoubleArray JNICALL
Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioJNI__J_3III(JNIEnv *env, jobject, jlong ref, jintArray dpts,
																	   jint start, jint end) {
	// if the cre_rt reference is not valid, return an empty array.
	if (ref == 0)
		return env->NewDoubleArray(0);

	jboolean isCopy;
	jint *datapoints = env->GetIntArrayElements(dpts, &isCopy);
	//jsize size = env->GetArrayLength(dpts);

	// allocate space for the ids array
	int size = end - start + 1;
	int_array ids(size);

	// copy the given range of data to the ids array
	int i = 0;
	for (jint *ptr = &datapoints[start]; i < size; ++i, ++ptr)
		ids[i] = *ptr;

	try {
		auto *cre_rt = (ClassRatioEstimatorRuntime *) ref;
		if (!cre_rt->isOK())
			env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
						  "CRE subsystem is not initialized due to unavailability of unlabeled data!");

		real_array props = cre_rt->EstimateClassProportions(ids);

		size_t psize = props.size();
		auto *array = new jdouble[psize];
		for (size_t j = 0; j < psize; ++j) array[j] = props[j];

		jdoubleArray result = env->NewDoubleArray(psize);
		env->SetDoubleArrayRegion(result, 0, psize, array);

		return result;
	}
	catch (std::exception &e) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
	catch (...) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
					  "unknown exception within CRE subsystem while estimating class ratio!");
	}
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    createCRESessionJNI
 * Signature: ([I)J
 */
JNIEXPORT jlong JNICALL
Java_ai_buddi_ml_cre_ClassRatioEstimator_createCRESessionJNI(JNIEnv *env, jobject, jintArray dpts) {
	jboolean isCopy;
	jint *datapoints = env->GetIntArrayElements(dpts, &isCopy);
	jsize size = env->GetArrayLength(dpts);

	int_array ids(size);
	int ind = 0;
	for (jint *ptr = datapoints; ind < size; ++ind, ++ptr)
		ids[ind] = *ptr;

	auto session = ClassRatioEstimatorRuntime::createCRESession(ids);

	// don't know why this is done, I just copied the way CRF++ JNI wrapper was implemented.
	jlong result;
	*(ClassRatioEstimatorRuntime::CRESession **) &result = session;

	return result;
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    deleteCRESessionJNI
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_deleteCRESessionJNI(JNIEnv *, jobject, jlong session) {
	if (session != 0) {
		auto *ptr = (ClassRatioEstimatorRuntime::CRESession *) session;
		ClassRatioEstimatorRuntime::deleteSession(ptr);
	}
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioInSessionJNI
 * Signature: (JJI)[D
 */
JNIEXPORT jdoubleArray JNICALL
Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioInSessionJNI(JNIEnv *env, jobject, jlong ref, jlong session,
																		jint pivot) {
	if (ref == 0 || session == 0)
		return env->NewDoubleArray(0);

	auto *cre_rt = (ClassRatioEstimatorRuntime *) ref;
	auto *cre_session = (ClassRatioEstimatorRuntime::CRESession *) session;

	auto props = cre_rt->EstimateClassProportionsInSession(cre_session, pivot);
	size_t psize = props.left.size() + props.right.size();
	auto *array = new jdouble[psize];

	// stitch both the class props as a flattened array.
	size_t i = 0;
	for (float j : props.left)
		array[i++] = j;
	for (float j : props.right)
		array[i++] = j;

	jdoubleArray result = env->NewDoubleArray(psize);
	env->SetDoubleArrayRegion(result, 0, psize, array);

	return result;
}

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioJNI
 * Signature: (J[II)[D
 */
JNIEXPORT jdoubleArray JNICALL
Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioJNI__J_3II(JNIEnv *env, jobject, jlong ref, jintArray dpts,
																	  jint dsize) {
	// if the cre_rt reference is not valid, return an empty array.
	if (ref == 0)
		return env->NewDoubleArray(0);

	jboolean isCopy;
	jint *datapoints = env->GetIntArrayElements(dpts, &isCopy);
	jsize size = env->GetArrayLength(dpts);
	if (dsize > size)
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
					  "input array size is larger than the data container size!");

	int_array ids(dsize);
	int ind = 0;
	for (jint *ptr = datapoints; ind < dsize; ++ind, ++ptr)
		ids[ind] = *ptr;

	try {
		auto *cre_rt = (ClassRatioEstimatorRuntime *) ref;
		if (!cre_rt->isOK())
			env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
						  "CRE subsystem is not initialized due to unavailability of unlabeled data!");

		real_array props = cre_rt->EstimateClassProportions(ids);

		size_t psize = props.size();
		auto *array = new jdouble[psize];
		for (size_t i = 0; i < psize; ++i) array[i] = props[i];

		jdoubleArray result = env->NewDoubleArray(psize);
		env->SetDoubleArrayRegion(result, 0, psize, array);

		return result;
	}
	catch (std::exception &e) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
	catch (...) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
					  "unknown exception within CRE subsystem while estimating class ratio!");
	}
}


/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioJNI
 * Signature: (J[I)[D
 */
JNIEXPORT jdoubleArray JNICALL
Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioJNI__J_3I(JNIEnv *env, jobject, jlong ref, jintArray dpts) {
	// if the cre_rt reference is not valid, return an empty array.
	if (ref == 0)
		return env->NewDoubleArray(0);

	jboolean isCopy;
	jint *datapoints = env->GetIntArrayElements(dpts, &isCopy);
	jsize size = env->GetArrayLength(dpts);

	int_array ids(size);
	int ind = 0;
	for (jint *ptr = datapoints; ind < size; ++ind, ++ptr)
		ids[ind] = *ptr;

	try {
		auto *cre_rt = (ClassRatioEstimatorRuntime *) ref;
		if (!cre_rt->isOK())
			env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
						  "CRE subsystem is not initialized due to unavailability of unlabeled data!");

		real_array props = cre_rt->EstimateClassProportions(ids);

		size_t psize = props.size();
		auto *array = new jdouble[psize];
		for (size_t i = 0; i < psize; ++i) array[i] = props[i];

		jdoubleArray result = env->NewDoubleArray(psize);
		env->SetDoubleArrayRegion(result, 0, psize, array);

		return result;
	}
	catch (std::exception &e) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
	catch (...) {
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
					  "unknown exception within CRE subsystem while estimating class ratio!");
	}
}
