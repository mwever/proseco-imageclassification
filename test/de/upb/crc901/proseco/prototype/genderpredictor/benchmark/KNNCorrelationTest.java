package de.upb.crc901.proseco.prototype.genderpredictor.benchmark;

import Catalano.Imaging.FastBitmap;
import Catalano.Imaging.Filters.Photometric.DifferenceOfGaussian;
import Catalano.Imaging.Filters.Photometric.IPhotometricFilter;
import Catalano.Imaging.Filters.Photometric.SelfQuocientImage;
import Catalano.Imaging.Filters.Photometric.SingleScaleRetinex;
import Catalano.Imaging.Filters.Photometric.TanTriggsNormalization;
import Catalano.Imaging.Texture.BinaryPattern.GradientLocalBinaryPattern;
import Catalano.Imaging.Texture.BinaryPattern.IBinaryPattern;
import Catalano.Imaging.Texture.BinaryPattern.ImprovedLocalBinaryPattern;
import Catalano.Imaging.Texture.BinaryPattern.LocalBinaryPattern;
import Catalano.Imaging.Texture.BinaryPattern.LocalGradientCoding;
import Catalano.Imaging.Tools.ImageHistogram;

import de.upb.crc901.proseco.prototype.imageclassification.benchmark.featureextraction.FeatureExtractionEvaluator;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.featureextraction.IBkFeatureEvaluator;

import jaicore.ml.WekaUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class KNNCorrelationTest {
  private static final int ILBP_GRANULARITY = 5;
  private static final int MIN = 250;
  private static final int FOLDS = 5;
  private static final int ESTIMATE_SET_SIZE = 50;

  private static final String CLASSIFIER_STATS_LOCATION = "X:\\data\\CRC901\\";

  private static final IPhotometricFilter[] IMAGE_FILTERS = { new SelfQuocientImage(), new SingleScaleRetinex(), new TanTriggsNormalization(), new DifferenceOfGaussian() };
  private static final IBinaryPattern[] BINARY_PATTERNS = { new GradientLocalBinaryPattern(), new ImprovedLocalBinaryPattern(), new LocalBinaryPattern(), new LocalBinaryPattern(),
      new LocalGradientCoding() };
  /**
   * Object variable for the feature extractor (to be defined).
   */
  private static IBinaryPattern bp = null;

  /**
   * Object variable for the Instances meta data
   */
  private Instances metadata;

  private static int innerPipelineCounter = 0;
  private static Lock monitorWriteLock = new ReentrantLock();
  private static Lock performanceLock = new ReentrantLock();

  public static void main(final String[] args) {
    // final BestFirst<TFDNode, String> searchAlgo = new BestFirst<>(MLUtil.getGraphGenerator(new
    // File("prototypes\\imageclassification\\strategies\\htn\\imagefilter.searchspace")),
    // new RandomizedDepthFirstEvaluator<>(new Random(12345)));
    // SimpleGraphVisualizationWindow<Node<TFDNode, Integer>> vis = new
    // SimpleGraphVisualizationWindow<>(searchAlgo.getEventBus());
    // vis.getPanel().setTooltipGenerator(new TFDTooltipGenerator());
    // System.out.println(searchAlgo.nextSolution());

    File imageDir = new File("tmp");
    imageDir.mkdir();
    unzipDataFile(new File("lfwgender_dataset.zip"), imageDir.toPath());

    /* read labels */
    final Map<String, String> labels = getLabelMap(imageDir.toPath());
    List<String> classes = new LinkedList<>();
    classes.addAll(labels.values().stream().collect(Collectors.toSet()));

    File[] fileArray = imageDir.listFiles(new FileFilter() {
      @Override
      public boolean accept(final File pathname) {
        if (!FilenameUtils.isExtension(pathname.getName(), "jpg")) {
          return false;
        }

        String label = labels.get(pathname.getName());
        if (label == null || label.equals("")) {
          return false;
        }
        return true;
      }
    });

    Map<String, FastBitmap> imageData = new HashMap<>();
    List<Integer> availableFilters = IntStream.range(0, IMAGE_FILTERS.length).mapToObj(x -> (Integer) x).collect(Collectors.toList());
    List<List<Integer>> allCombinations = getAllCombinations(availableFilters);
    Collections.shuffle(allCombinations);
    for (List<Integer> imagePreprocessingCombo : allCombinations) {
      System.out.print((allCombinations.indexOf(imagePreprocessingCombo) + 1) + "/" + allCombinations.size() + " ");
      innerPipelineCounter = 0;
      imageData.clear();
      Arrays.stream(fileArray).parallel().forEach(imageFile -> {
        // Placeholder for applying image filters to the fast bitmap object
        FastBitmap fb = new FastBitmap(imageFile.getAbsolutePath());
        new Catalano.Imaging.Filters.Grayscale().applyInPlace(fb);
        imageData.put(imageFile.getName(), fb);
      });
      executePipeline(imageData, classes, labels, imagePreprocessingCombo);
    }

  }

  private static List<List<Integer>> getAllCombinations(final List<Integer> availableFilters) {
    List<List<Integer>> integerList = new LinkedList<>();

    List<List<Integer>> latestAdded = new LinkedList<>();
    for (Integer x : availableFilters) {
      List<Integer> atomList = new LinkedList<>();
      atomList.add(x);
      latestAdded.add(atomList);
    }

    for (int i = 1; i < availableFilters.size(); i++) {
      List<List<Integer>> newLatestAdded = new LinkedList<>();
      for (List<Integer> latestAddedItem : latestAdded) {
        for (Integer x : availableFilters) {
          List<Integer> copyOfLatestAddedItem = new LinkedList<>(latestAddedItem);
          copyOfLatestAddedItem.add(x);
          newLatestAdded.add(copyOfLatestAddedItem);
        }
      }
      integerList.addAll(latestAdded);
      latestAdded.clear();
      latestAdded.addAll(newLatestAdded);
    }

    return integerList;
  }

  private static void executePipeline(final Map<String, FastBitmap> images, final List<String> classes, final Map<String, String> labels, final List<Integer> imagePreprocessing) {

    System.out.print(" Start Preprocessing ");
    images.values().stream().parallel().forEach(img -> {
      imagePreprocessing.stream().forEach(filterIx -> IMAGE_FILTERS[filterIx].applyInPlace(img));
    });
    System.out.print(" Preprocessing DONE ");

    Classifier[] classifierPortfolio = { new BayesNet(), new NaiveBayes(), new NaiveBayesMultinomial(), new GaussianProcesses(), new LinearRegression(), // new Logistic(),
        // new MultilayerPerceptron(),
        new SGD(), new SimpleLinearRegression(), new SimpleLogistic(), new VotedPerceptron(), new IBk(), new KStar(), // new DecisionTable(),
        new JRip(), new M5Rules(), new OneR(), new PART(), new ZeroR(), new DecisionStump(), new J48(), // new LMT(),
        new M5P(), new RandomForest(), new RandomTree(), new REPTree() };

    ExecutorService pool = Executors.newFixedThreadPool(4);
    for (IBinaryPattern featureExtraction : BINARY_PATTERNS) {
      try {

        bp = featureExtraction;
        String pipeline = "";
        for (Integer ipIx : imagePreprocessing) {
          if (!pipeline.equals("")) {
            pipeline += "->";
          }
          pipeline += IMAGE_FILTERS[ipIx].getClass().getName();
        }
        pipeline += "=>" + bp.getClass().getName();

        System.out.println("Compute instances from fastbitmaps...");
        Instances allInstances = getEmptyDataset(classes, getILBPAttributes(classes));
        images.keySet().stream().parallel().forEach(key -> {
          allInstances.add(applyBP(images.get(key), allInstances, labels.get(key)));
        });
        System.out.println("Done with computing instances from fastbitmaps");

        List<Instances> stratifiedSplit = WekaUtil.getStratifiedSplit(allInstances, new Random(), 0.75);
        Instances trainInstances = stratifiedSplit.get(0);
        Instances testInstances = stratifiedSplit.get(1);

        Instances estimateInstances = getEmptyDataset(classes, getILBPAttributes(classes));

        System.out.println("Choose estimation instances");
        if (ESTIMATE_SET_SIZE >= trainInstances.size()) {
          estimateInstances.addAll(trainInstances);
        } else {
          Random rand = new Random();
          while (estimateInstances.size() < ESTIMATE_SET_SIZE) {
            Instance sampledInstance = trainInstances.get(rand.nextInt(trainInstances.size()));
            if (!estimateInstances.contains(sampledInstance)) {
              estimateInstances.add(sampledInstance);
              System.out.print(estimateInstances.size() + " ");
            }
          }
          System.out.println();
        }
        System.out.println("Chose estimation set of size " + estimateInstances.size());

        String setup = "folds=" + FOLDS + ";estimateSize=" + estimateInstances.size() + ";ilbp_granularity=" + ILBP_GRANULARITY;

        System.out.println("knn Estimate");
        FeatureExtractionEvaluator eval = new IBkFeatureEvaluator(FOLDS);
        double estimate = (1 - eval.evaluate(estimateInstances));
        System.out.println("knn estimate done.");

        logPerformance(setup, pipeline, "knnEstiamte=" + estimate);

        int hashCodeOfTrainingData = new HashCodeBuilder().append(trainInstances.toString()).toHashCode();
        writeDataSet(hashCodeOfTrainingData, trainInstances);

        final String lockedPipeline = pipeline;

        for (Classifier classifier : classifierPortfolio) {
          pool.submit(new Runnable() {

            @Override
            public void run() {
              System.out.println(Thread.currentThread().getName() + ": Start evaluation of classifier " + classifier.getClass().getSimpleName());
              long startTime = System.currentTimeMillis();
              double accuracy = 0;
              try {
                classifier.buildClassifier(trainInstances);
                double errorRate = computeErrorRate(classifier, testInstances);
                logPerformance(setup, lockedPipeline, classifier.getClass().getName() + "=" + errorRate);
                accuracy = 1 - errorRate;
              } catch (Exception e) {
                logPerformance(setup, lockedPipeline, classifier.getClass().getName() + "=-1");
              }
              monitorPerformanceStats(hashCodeOfTrainingData, WekaUtil.getClassifierDescriptor(classifier), (System.currentTimeMillis() - startTime), accuracy);
              innerPipelineCounter++;
              System.out.println(Thread.currentThread().getName() + ": Finished evaluation of " + classifier.getClass().getSimpleName());
            }

          });
        }
      } catch (Exception e) {
        System.err.println("Exception occured...");
        e.printStackTrace();
      }
    }
    pool.shutdown();
    try {
      System.out.println("Await Pool Termination of pool " + pool.toString());
      pool.awaitTermination(120, TimeUnit.MINUTES);
      System.out.println("Pool shut down.");
    } catch (InterruptedException e) {
      e.printStackTrace();
      pool.shutdownNow();
    }
  }

  /**
   * Computes the error rate of the classifier with respect to the given test instances.
   *
   * @param testInstances
   *          Instances to make predictions for with the built classifier.
   * @return Returns the accuracy
   * @throws Exception
   */
  private static double computeErrorRate(final Classifier classifier, final Instances testInstances) throws Exception {
    int correctPredictions = 0;

    for (int i = 0; i < testInstances.numInstances(); i++) {
      double pred = classifier.classifyInstance(testInstances.instance(i));
      if (testInstances.classAttribute().value((int) testInstances.instance(i).classValue()).equals(testInstances.classAttribute().value((int) pred))) {
        correctPredictions++;
      }
    }
    double accuracy = correctPredictions * 1f / testInstances.numInstances();
    return accuracy;
  }

  private static void writeDataSet(final int hashCodeOfTrainingData, final Instances trainInstances) {
    try (BufferedWriter bw = new BufferedWriter(new FileWriter(CLASSIFIER_STATS_LOCATION + File.separator + "data" + File.separator + hashCodeOfTrainingData + ".arff"))) {
      bw.write(trainInstances.toString());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private static void monitorPerformanceStats(final int hashCodeOfTrainingData, final String classifierDescriptor, final long runtime, final double accuracy) {
    double roundedAccuracy = Math.round(accuracy * 10000) / 100;
    monitorWriteLock.lock();
    try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File(CLASSIFIER_STATS_LOCATION + File.separator + "stats2.csv"), true))) {
      bw.write(hashCodeOfTrainingData + "," + classifierDescriptor + "," + runtime + "," + roundedAccuracy + "\n");
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      monitorWriteLock.unlock();
    }
  }

  private static void logPerformance(final String setup, final String pipeline, final String measuredPerformance) {
    performanceLock.lock();
    try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File("knn.log"), true))) {
      bw.write(setup + ";" + pipeline + ";" + measuredPerformance + "\n");
    } catch (IOException e1) {
      e1.printStackTrace();
    } finally {
      performanceLock.unlock();
    }
  }

  /**
   * Extracts a zip file and stores its contents to the outputFolder path.
   *
   * @param zipFile
   *          Zip file to extract.
   * @param outputFolder
   *          Output folder to extract the files contained in zipFile to.
   */
  protected static void unzipDataFile(final File zipFile, final Path outputFolder) {
    final byte[] buffer = new byte[1024];
    try {
      // create output directory is not exists
      if (!outputFolder.toFile().exists()) {
        outputFolder.toFile().mkdir();
      }

      // get the zip file content
      final ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile));
      // get the zipped file list entry
      ZipEntry ze = zis.getNextEntry();

      while (ze != null) {
        final String fileName = ze.getName();
        final File newFile = new File(outputFolder.toFile().getAbsolutePath() + File.separator + fileName);

        // create all non exists folders
        // else you will hit FileNotFoundException for compressed folder
        new File(newFile.getParent()).mkdirs();
        final FileOutputStream fos = new FileOutputStream(newFile);
        int len;
        while ((len = zis.read(buffer)) > 0) {
          fos.write(buffer, 0, len);
        }
        fos.close();
        ze = zis.getNextEntry();
      }
      zis.closeEntry();
      zis.close();
    } catch (final IOException ex) {
      ex.printStackTrace();
    }
  }

  private static ArrayList<Attribute> getILBPAttributes(final List<String> classes) {
    /* compute number of features */
    int numberOfFeatures;
    if (bp instanceof LocalBinaryPattern || bp instanceof Catalano.Imaging.Texture.BinaryPattern.GradientLocalBinaryPattern
        || bp instanceof Catalano.Imaging.Texture.BinaryPattern.LocalGradientCoding || bp instanceof Catalano.Imaging.Texture.BinaryPattern.MultiblockLocalBinaryPattern) {
      numberOfFeatures = 256;
    } else if (bp instanceof Catalano.Imaging.Texture.BinaryPattern.CenterSymmetricLocalBinaryPattern) {
      numberOfFeatures = 16;
    } else {
      numberOfFeatures = 511;
    }

    final int n = numberOfFeatures * ILBP_GRANULARITY * ILBP_GRANULARITY; // 511 is the number of
                                                                          // features in each square
    final ArrayList<Attribute> attributes = new ArrayList<>(n + 1);
    for (int i = 0; i < n; i++) {
      attributes.add(new Attribute("p" + i));
    }
    attributes.add(new Attribute("class", classes));
    return attributes;
  }

  public static Instance applyBP(final FastBitmap fb, final Instances dataset, final String classValue) {
    /* go through boxes and compute ilbp */

    final int[][] matrix = fb.toMatrixGrayAsInt();
    List<Integer> attributeVals = new ArrayList<>();

    /* compute ilbp histogram for each square */
    final int length = Math.min(fb.getWidth(), fb.getHeight());
    final int stepSize = (int) Math.floor(length * 1f / ILBP_GRANULARITY);
    for (int xSquare = 0; xSquare < ILBP_GRANULARITY; xSquare++) {
      for (int ySquare = 0; ySquare < ILBP_GRANULARITY; ySquare++) {

        /* determine the submatrix of this square */
        final int[][] excerpt = new int[stepSize][stepSize];
        for (int i = 0; i < stepSize; i++) {
          for (int j = 0; j < stepSize; j++) {
            excerpt[i][j] = matrix[xSquare * stepSize + i][ySquare * stepSize + j];
          }
        }

        /* create fast bitmap and apply ilbp */
        FastBitmap fb2 = new FastBitmap(excerpt);
        final ImageHistogram hist = bp.ComputeFeatures(fb2);
        final int[] attributesForSquare = hist.getValues();
        for (final int val : attributesForSquare) {
          attributeVals.add(val);
          // JOptionPane.showMessageDialog(null, fb.toIcon(), "Result", JOptionPane.PLAIN_MESSAGE);
        }
      }
    }

    /* now create instance object */
    final Instance inst = new DenseInstance(attributeVals.size() + 1);
    inst.setDataset(dataset);

    /* set attribute values */
    for (int i = 0; i < attributeVals.size(); i++) {
      inst.setValue(i, attributeVals.get(i));
    }

    /* if there is a class assigned */
    try {
      inst.setValue(attributeVals.size(), classValue);
    } catch (IllegalArgumentException e) {
      System.out.println("Class value: " + classValue);
      e.printStackTrace();
    }

    return inst;
  }

  public static void processDataAndAddToDataset(final File imageFile, final Instances dataset, final String classValue) {
    /* create matrix representation of image */
    FastBitmap fb = new FastBitmap(imageFile.getAbsolutePath());
    // Placeholder for applying image filters to the fast bitmap object
    final int min = Math.min(fb.getWidth(), fb.getHeight());
    new Catalano.Imaging.Filters.Grayscale().applyInPlace(fb);
    /* $imagefilter$ */

    Instance inst = applyBP(fb, dataset, classValue);
    dataset.add(inst);
  }

  protected static Map<String, String> getLabelMap(final Path folder) {
    Map<String, String> labels = new HashMap<>();

    try {
      final BufferedReader br = new BufferedReader(new FileReader(new File(folder.toFile().getAbsolutePath() + File.separator + "labels.txt")));
      String line;
      while ((line = br.readLine()) != null) {
        final String[] split = line.split(",");
        if (split.length == 2 && !split[1].isEmpty()) {
          labels.put(split[0], split[1]);
        }
      }
      br.close();
    } catch (final FileNotFoundException e1) {
      e1.printStackTrace();
    } catch (final IOException e) {
      e.printStackTrace();
    }

    return labels;
  }

  public static Instances getEmptyDataset(final List<String> classes, final ArrayList<Attribute> attributes) {
    final Instances data = new Instances("images", attributes, 0);
    data.setClassIndex(data.numAttributes() - 1);
    return data;
  }

  private static void log(final String msg) {
    System.out.println("Gender Predictor: " + msg);
  }

}
