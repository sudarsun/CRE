/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class ai_buddi_ml_cre_ClassRatioEstimator */

#ifndef _Included_ai_buddi_ml_cre_ClassRatioEstimator
#define _Included_ai_buddi_ml_cre_ClassRatioEstimator
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    isOK
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_isOK
		(JNIEnv *, jobject, jlong);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    nFeatures
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_nFeatures
		(JNIEnv *, jobject, jlong);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    nClasses
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_nClasses
		(JNIEnv *, jobject, jlong);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    classPriors
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_classPriors
		(JNIEnv *, jobject, jlong);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    buildJNI
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_buildJNI__Ljava_lang_String_2Ljava_lang_String_2
		(JNIEnv *, jobject, jstring, jstring);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    buildJNI
 * Signature: ([DII[I)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_buildJNI___3DII_3I
		(JNIEnv *, jobject, jdoubleArray, jint, jint, jintArray);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    deleteJNI
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_deleteJNI
		(JNIEnv *, jobject, jlong);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioJNI
 * Signature: (J[I)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioJNI__J_3I
		(JNIEnv *, jobject, jlong, jintArray);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioJNI
 * Signature: (J[II)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioJNI__J_3II
		(JNIEnv *, jobject, jlong, jintArray, jint);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioJNI
 * Signature: (J[III)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioJNI__J_3III
		(JNIEnv *, jobject, jlong, jintArray, jint, jint);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    createCRESessionJNI
 * Signature: ([I)J
 */
JNIEXPORT jlong JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_createCRESessionJNI
		(JNIEnv *, jobject, jintArray);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    deleteCRESessionJNI
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_deleteCRESessionJNI
		(JNIEnv *, jobject, jlong);

/*
 * Class:     ai_buddi_ml_cre_ClassRatioEstimator
 * Method:    estimateClassRatioInSessionJNI
 * Signature: (JJI)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_buddi_ml_cre_ClassRatioEstimator_estimateClassRatioInSessionJNI
		(JNIEnv *, jobject, jlong, jlong, jint);

#ifdef __cplusplus
}
#endif
#endif
