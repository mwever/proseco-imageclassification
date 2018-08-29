package de.upb.crc901.proseco.prototype.imageclassification.benchmark.featureextraction;

import weka.core.Instances;

/**
 * Interface for all feature extraction evaluators.
 *
 * @author mwever
 */
public interface FeatureExtractionEvaluator {

  /**
   * Evaluate the given instances and assess the features in their relation to the assigned labels.
   *
   * @param instancesToEvaluate
   *          The instances to assess their feature representation.
   * @return The accuracy of the evaluator.
   */
  public double evaluate(Instances instancesToEvaluate);

}
