package de.upb.crc901.proseco.prototype.imageclassification.benchmark.runner;

import de.upb.crc901.proseco.PrototypeProperties;
import de.upb.crc901.proseco.prototype.imageclassification.GroundingRoutine;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.InstancesTuple;

import jaicore.basic.FileUtil;
import jaicore.basic.PerformanceLogger;
import jaicore.ml.WekaUtil;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.lang.ProcessBuilder.Redirect;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import weka.classifiers.Classifier;

public class ClassifierBenchmarkRunner extends AbstractBenchmarkRunner {
  private static final PrototypeProperties PROPS = new PrototypeProperties(".." + File.separator + "config" + File.separator + "benchmarkservice.conf");
  private static final String CLASSIFIER_MODEL_FILE = PROPS.getProperty("classifier_model_file");
  private static final String MONITOR_OUTPUT = PROPS.getProperty("monitoring_folder");

  private static final String FVALUE_FILE = PROPS.getProperty("name_fvaluefile");
  private static final File BENCHMARK_INSTANCES_FILE = new File(PROPS.getProperty("benchmark_instances"));

  private final int hashCodeOfTrainingData;

  private final InstancesTuple instances;

  public ClassifierBenchmarkRunner(final BenchmarkTask pTask, final GroundingRoutine pGroundingRoutine, final File taskTempFolder, final InstancesTuple tuple) {
    super(pTask, pGroundingRoutine, taskTempFolder);
    this.instances = tuple;
    int hashCode = 0;
    try {
      hashCode = new HashCodeBuilder().append(FileUtil.readFileAsString(this.instances.trainingData.getParent() + File.separator + "train.arff")).toHashCode();
    } catch (IOException e) {
      e.printStackTrace();
    }
    this.hashCodeOfTrainingData = hashCode;
  }

  @Override
  public void run() {
    try {
      this.getGroundingRoutine().codeAssembly();

      this.getGroundingRoutine().compile();

      long start = System.currentTimeMillis();
      this.getGroundingRoutine().trainModel(this.instances.trainingData);
      long end = System.currentTimeMillis();

      // Test trained instance against validation set
      PerformanceLogger.logStart("computeFValue");
      log("Compute f value for current testbed ... ", false);
      int accuracyTmp = (int) Math.round(this.computeAccuracy() * 10000);
      double h = 100;
      double accuracy = accuracyTmp / h;
      log("DONE");
      PerformanceLogger.logEnd("computeFValue");

      /* write stats */
      log("Writing stat file ... ", false);
      Classifier usedClassifier = null;
      File classifierFile = new File(this.getTaskTempFolder().getAbsolutePath() + File.separator + CLASSIFIER_MODEL_FILE);
      try (ObjectInputStream br = new ObjectInputStream(new BufferedInputStream(new FileInputStream(classifierFile)))) {
        usedClassifier = (Classifier) br.readObject();
      } catch (final Exception e) {
        e.printStackTrace();
      }
      this.writeStats(usedClassifier, end - start, accuracy);
      log("DONE");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private double computeAccuracy() {
    double acc = 0;
    if (new File(this.getTaskTempFolder().getAbsolutePath() + File.separator + CLASSIFIER_MODEL_FILE).exists()) {
      final ProcessBuilder pb = new ProcessBuilder(this.getTaskTempFolder().getAbsolutePath() + File.separator + "test.bat", this.instances.validationData.getAbsolutePath());
      pb.redirectError(Redirect.INHERIT);

      try {
        final Process fValueProcess = pb.start();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(fValueProcess.getInputStream()), 1)) {
          final String fValueString = br.readLine();
          if (fValueString != null) {

            final String[] fValueStringSplit = fValueString.split("=");
            if (fValueStringSplit.length == 2) {
              acc = Double.parseDouble(fValueStringSplit[1]);
            }
          }
        }
        fValueProcess.waitFor();
      } catch (final IOException e) {
        e.printStackTrace();
      } catch (final InterruptedException e) {
        e.printStackTrace();
      }
    }

    try (BufferedWriter bw = new BufferedWriter(new FileWriter(this.getTaskTempFolder() + File.separator + FVALUE_FILE))) {
      bw.write(acc + "\n");
    } catch (final IOException e) {
      e.printStackTrace();
    }
    return acc;
  }

  private static void log(final String msg) {
    log(msg, true);
  }

  private static boolean lastLineBreak = true;

  private static void log(final String msg, final boolean linebreak) {
    final String prefix = "[" + Thread.currentThread().getName() + "] BenchmarkService>ClassifierBenchmarkRunner: ";
    String printString;
    if (lastLineBreak) {
      printString = prefix + msg;
    } else {
      printString = msg;
    }

    if (linebreak) {
      System.out.println(printString);
    } else {
      System.out.print(printString);
    }
    lastLineBreak = linebreak;
  }

  private void writeStats(final Classifier algorithm, final long runtime, final double accuracy) {

    try {

      File folder = new File(new File(MONITOR_OUTPUT) + File.separator + "data");

      /* copy the data to the stats data folder if it is not there already */
      File dataTarget = new File(folder.getAbsolutePath() + File.separator + this.hashCodeOfTrainingData + ".arff");
      if (!dataTarget.exists()) {
        File dataSrc = new File(this.instances.trainingData.getParent() + File.separator + "train.arff");
        try {
          FileUtils.copyFile(dataSrc, dataTarget);
        } catch (IOException e) {
          e.printStackTrace();
        }
      }

      /* write data file if not done yet */
      try (FileWriter fw = new FileWriter(new File(MONITOR_OUTPUT) + File.separator + "stats.csv", true)) {
        if (!folder.exists()) {
          folder.mkdirs();
        }
        fw.write(this.hashCodeOfTrainingData + ", " + WekaUtil.getClassifierDescriptor(algorithm) + ", " + runtime + ", " + accuracy + "\n");
      } catch (IOException e) {
        e.printStackTrace();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
