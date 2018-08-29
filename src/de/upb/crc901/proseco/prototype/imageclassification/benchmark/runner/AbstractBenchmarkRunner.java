package de.upb.crc901.proseco.prototype.imageclassification.benchmark.runner;

import de.upb.crc901.proseco.prototype.imageclassification.GroundingRoutine;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask;

import java.io.File;

public abstract class AbstractBenchmarkRunner {

  private final BenchmarkTask task;
  private final GroundingRoutine groundingRoutine;
  private final File taskTempFolder;

  protected AbstractBenchmarkRunner(final BenchmarkTask task, final GroundingRoutine groundingRoutine, final File taskTempFolder) {
    this.task = task;
    this.groundingRoutine = groundingRoutine;
    this.taskTempFolder = taskTempFolder;
  }

  protected BenchmarkTask getTask() {
    return this.task;
  }

  protected GroundingRoutine getGroundingRoutine() {
    return this.groundingRoutine;
  }

  protected File getTaskTempFolder() {
    return this.taskTempFolder;
  }

  public abstract void run();

}
