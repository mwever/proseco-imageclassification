package de.upb.crc901.proseco.prototype.imageclassification.benchmark.featureextraction;

import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

/**
 * Evaluator to asses the quality of a feature preprocessing pipeline. This evaluator uses a 10-fold
 * crossvalidation and sets the neighbourhood to be the expected fraction of each class.
 *
 * @author mwever
 */
public class IBkFeatureEvaluator implements FeatureExtractionEvaluator {
  /**
   * Number of folds to perform the cross-validation.
   */
  private int folds = 10;

  /**
   * Standard c'tor.
   */
  public IBkFeatureEvaluator() {
    super();
  }

  /**
   * C'tor defining a customized number of folds.
   *
   * @param folds
   *          The number of folds for the cross-validation.
   */
  public IBkFeatureEvaluator(final int folds) {
    this();
    this.folds = folds;
  }

  @Override
  public double evaluate(final Instances instancesToEvaluate) {
    IBk classifier = new IBk();
    classifier.setKNN(instancesToEvaluate.size() / instancesToEvaluate.numClasses());
    try {
      Evaluation eval = new Evaluation(instancesToEvaluate);
      Object[] out = new Object[] {};
      eval.crossValidateModel(classifier, instancesToEvaluate, this.folds, new Random(123), out);
      return 1 - eval.errorRate();
    } catch (Exception e) {
      e.printStackTrace();
    }
    return 0;
  }

}
