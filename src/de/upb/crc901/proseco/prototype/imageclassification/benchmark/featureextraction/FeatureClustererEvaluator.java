package de.upb.crc901.proseco.prototype.imageclassification.benchmark.featureextraction;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.collections4.MultiValuedMap;
import org.apache.commons.collections4.multimap.HashSetValuedHashMap;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.FarthestFirst;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Evaluator using clustering in order to cluster similar feature representations ignoring their
 * labels and assesses the accuracy of the feature representation evaluating the correct clustering
 * with respect to the labels.
 * 
 * @author mwever
 */
public class FeatureClustererEvaluator implements FeatureExtractionEvaluator {

  @Override
  public double evaluate(final Instances instancesToEvaluate) {
    AttributeStats stats = instancesToEvaluate.attributeStats(instancesToEvaluate.classIndex());
    for (int i = 0; i < stats.nominalCounts.length; i++) {
      System.out.println(i + ": " + stats.nominalCounts[i]);
    }

    Instances copyOfInstances;
    Remove af = new Remove();
    try {
      if (instancesToEvaluate.classIndex() < 0) {
        copyOfInstances = new Instances(instancesToEvaluate);
      } else {
        af.setAttributeIndices("" + (instancesToEvaluate.classIndex() + 1));
        af.setInvertSelection(false);
        af.setInputFormat(instancesToEvaluate);
        copyOfInstances = Filter.useFilter(instancesToEvaluate, af);
      }

      MultiValuedMap<Integer, Instance> clusters = new HashSetValuedHashMap<>();
      Map<Integer, Integer> clusterToClassIndex = new HashMap<>();

      FarthestFirst clusterAlg = new FarthestFirst();

      clusterAlg.setNumClusters(instancesToEvaluate.numClasses());
      AbstractClusterer clusterer = clusterAlg; // original clusterer
      clusterer.buildClusterer(copyOfInstances);

      int[] assignments = new int[copyOfInstances.size()];
      for (int i = 0; i < copyOfInstances.size(); i++) {
        assignments[i] = clusterer.clusterInstance(copyOfInstances.get(i));
      }

      for (int i = 0; i < assignments.length; i++) {
        clusters.put(assignments[i], instancesToEvaluate.get(i));
      }

      int misclassified = 0;
      for (Map.Entry<Integer, Collection<Instance>> cluster : clusters.asMap().entrySet()) {
        int[] classInstanceCounter = new int[instancesToEvaluate.numClasses()];

        cluster.getValue().stream().forEach(x -> classInstanceCounter[(int) x.classValue()]++);

        int classForCluster = 0;
        for (int i = 0; i < classInstanceCounter.length; i++) {
          if (classInstanceCounter[i] >= classInstanceCounter[classForCluster]) {
            classForCluster = i;
          }
        }

        System.out.print("Cluster " + cluster.getKey() + " ");
        for (int i = 0; i < classInstanceCounter.length; i++) {
          System.out.print(i + "=" + classInstanceCounter[i] + " ");
          if (i != classForCluster) {
            misclassified += classInstanceCounter[i];
          }
        }
        System.out.println();

        clusterToClassIndex.put(cluster.getKey(), classForCluster);
      }

      return 1.0 - ((double) misclassified / instancesToEvaluate.size());
    } catch (Exception e) {
      e.printStackTrace();
    }

    return 0;
  }

}
