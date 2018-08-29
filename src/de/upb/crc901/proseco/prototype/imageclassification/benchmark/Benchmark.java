package de.upb.crc901.proseco.prototype.imageclassification.benchmark;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import de.upb.crc901.proseco.PrototypeProperties;
import de.upb.crc901.proseco.prototype.imageclassification.GroundingRoutine;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.featureextraction.IBkFeatureEvaluator;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.runner.AbstractBenchmarkRunner;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.runner.ClassifierBenchmarkRunner;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.runner.FeatureExtractionBenchmarkRunner;
import jaicore.basic.FileUtil;
import jaicore.basic.PerformanceLogger;

public class Benchmark extends Thread {

	private static final PrototypeProperties PROPS = new PrototypeProperties(
			".." + File.separator + "config" + File.separator + "benchmarkservice.conf");

	private static final File WAITING_TASK_DIR = new File(PROPS.getProperty("waiting_task_dir"));
	private static final File FINISHED_TASK_DIR = new File(PROPS.getProperty("finished_task_dir"));
	private static final File TESTBED_DIR = new File(PROPS.getProperty("testbed_dir"));

	private static final File SOURCE_INPUT_FOLDER = new File(PROPS.getProperty("prototype_source_code"));

	private static final int NUMBER_OF_THREADS = Integer.parseInt(PROPS.getProperty("number_of_threads"));
	private static final String TASK_FILE_EXT = "task";

	private static final String[] INSTANCES_HASH_PLACEHOLDERS = { "imagefilter" };

	private static final File DATA_FILE = new File(
			".." + File.separator + "interview_data/interview_resources" + File.separator + "data.zip");
	private static final File GROUNDING_CONFIG = new File(
			".." + File.separator + "config" + File.separator + "groundingroutine.conf");

	private volatile boolean keepRunning = true;
	private final List<String> processedFileNames;
	private final Lock fileNameListLock;
	private File taskTempFolder;

	private static final String MONITOR_OUTPUT = "X:\\data\\CRC901\\stats.csv";

	/**
	 * Lock that needs to be acquired first before instancesLockMap is allowed
	 * to be accessed.
	 */
	private final Lock instancesLockMapLock;
	/**
	 * Map containing the locks to access the tuples of instances. First thread
	 * acquiring a lock for not yet existing instances tuple executes the
	 * buildInstances of the prototype
	 */
	private final Map<Integer, Lock> instancesLockMap;
	/**
	 * Map containing tuples of instances that are already serialized.
	 */
	private final Map<Integer, InstancesTuple> instancesTupleMap;

	public Benchmark(final String pName, final List<String> processedFileNames, final Lock fileNameListLock,
			final Lock instancesLockMapLock, final Map<Integer, Lock> instancesLockMap,
			final Map<Integer, InstancesTuple> instancesTupleMap) {
		super(pName);
		this.processedFileNames = processedFileNames;
		this.fileNameListLock = fileNameListLock;
		this.instancesLockMapLock = instancesLockMapLock;
		this.instancesLockMap = instancesLockMap;
		this.instancesTupleMap = instancesTupleMap;
	}

	@Override
	public void run() {
		try {
			System.out.println(Thread.currentThread().getName() + ": Thread is running.");
			PerformanceLogger.logStart("Uptime");

			while (this.keepRunning) {
				final File[] fileList = WAITING_TASK_DIR.listFiles();

				if (fileList == null) {
					continue;
				}

				int numberOfProcessedTasks = 0;
				for (final File taskFile : fileList) {
					if (taskFile.isDirectory()
							|| !FilenameUtils.isExtension(taskFile.getAbsolutePath(), TASK_FILE_EXT)) {
						continue;
					}

					BenchmarkTask task;
					if ((task = this.getNextTask(taskFile)) == null) {
						continue;
					}

					GroundingRoutine groundingRoutine = new GroundingRoutine(task.getCandidateFolder(),
							SOURCE_INPUT_FOLDER.getCanonicalFile(), this.taskTempFolder, GROUNDING_CONFIG);

					AbstractBenchmarkRunner benchmarkRunner = null;
					switch (task.getBuildPhase()) {
					case FEATURE_EXTRACTION:
						benchmarkRunner = new FeatureExtractionBenchmarkRunner(task, groundingRoutine,
								this.taskTempFolder, DATA_FILE.getCanonicalFile(), new IBkFeatureEvaluator());
						break;
					case CLASSIFIER_DEF:
						InstancesTuple tuple = this.getInstances(task);
						benchmarkRunner = new ClassifierBenchmarkRunner(task, groundingRoutine, this.taskTempFolder,
								tuple);
						break;
					}

					log("Start to benchmark task " + taskFile.getAbsolutePath() + " for candidate "
							+ task.getCandidateFolder().getAbsolutePath());
					benchmarkRunner.run();

					String[] ignoreFilesForMoving = { "compile.bat", "libs", "test.bat", "train.bat",
							"validationInstances.serialized" };
					Set<String> ignoreFilesForMovingSet = Arrays.stream(ignoreFilesForMoving)
							.collect(Collectors.toSet());

					// move task specific files to task directory
					log("Benchmark Service: Move files from " + this.taskTempFolder + " to "
							+ task.getCandidateFolder().getAbsolutePath(), false);
					for (final File testBedFile : this.taskTempFolder.listFiles()) {
						if (!ignoreFilesForMovingSet.contains(testBedFile.getName())) {
							final File candidateFile = new File(task.getCandidateFolder().getAbsolutePath()
									+ File.separator + testBedFile.getName());
							if (candidateFile.exists()) {
								candidateFile.delete();
							}

							if (testBedFile.isFile()) {
								FileUtils.copyFile(testBedFile, candidateFile);
							}
						}
					}
					FileUtils.copyFile(taskFile,
							new File(FINISHED_TASK_DIR.getAbsolutePath() + File.separator + taskFile.getName()));
					log("DONE.");

					// XXX delete temporary directory
					// FileUtils.deleteDirectory(this.taskTempFolder);
					PerformanceLogger.logEnd("PerformBenchmarkForCandidate");
					log("Finished task " + taskFile.getName());
					numberOfProcessedTasks++;
				}

				if (numberOfProcessedTasks == 0) {
					// wait for new tasks and go to sleep for some millis
					Thread.sleep(1000);
				}
			}
		} catch (final IOException e) {
			e.printStackTrace();
		} catch (final InterruptedException e) {
			log("Woke up by interrupt.");
		} finally {
			log("Service shutting down, saving global performance log to file");
			PerformanceLogger.logEnd("Uptime");
			PerformanceLogger.saveGlobalLogToFile(new File("../InternalBenchmark.log"));
		}
	}

	private InstancesTuple getInstances(final BenchmarkTask benchmarkTask) {
		Integer instancesHashValue = this.getHashValueForBenchmarkTask(benchmarkTask);
		// get lock for the instances tuple
		Lock instancesTupleLock;
		this.instancesLockMapLock.lock();
		try {
			instancesTupleLock = this.instancesLockMap.get(instancesHashValue);
			if (instancesTupleLock == null) {
				instancesTupleLock = new ReentrantLock();
				this.instancesLockMap.put(instancesHashValue, instancesTupleLock);
			}
		} finally {
			this.instancesLockMapLock.unlock();
		}

		InstancesTuple instances;
		// lock the instancesTupleLock and get instancesTuple
		instancesTupleLock.lock();
		try {
			instances = this.instancesTupleMap.get(instancesHashValue);
			if (instances == null) {
				instances = this.serializeInstancesForBenchmarkTask(benchmarkTask, instancesHashValue);
				this.instancesTupleMap.put(instancesHashValue, instances);
			}
		} finally {
			instancesTupleLock.unlock();
		}

		return instances;
	}

	private InstancesTuple serializeInstancesForBenchmarkTask(final BenchmarkTask benchmarkTask,
			final Integer instancesHashValue) {
		File buildInstancesDir = new File(instancesHashValue + "");
		File instancesCacheDir = new File("cachedInstances" + File.separator + instancesHashValue);
		instancesCacheDir.mkdirs();

		try {
			FileUtils.copyDirectory(TESTBED_DIR, buildInstancesDir);

			GroundingRoutine grounding = new GroundingRoutine(benchmarkTask.getCandidateFolder(), SOURCE_INPUT_FOLDER,
					buildInstancesDir, GROUNDING_CONFIG);

			grounding.codeAssembly();

			grounding.compile();

			grounding.buildInstances(DATA_FILE.getCanonicalFile(), -1);

			// move files from buildinstancesdir to cache dir
			String[] filesToMove = { "allInstances.arff", "train.arff", "contTrain.serialized", "instances.serialized",
					"test.serialized", "train.serialized", "validation.serialized" };

			Map<String, File> serializedFiles = new HashMap<>();

			for (String fileToMove : filesToMove) {
				serializedFiles.put(fileToMove,
						new File(instancesCacheDir.getAbsolutePath() + File.separator + fileToMove));
				FileUtils.copyFile(new File(buildInstancesDir.getAbsolutePath() + File.separator + fileToMove),
						serializedFiles.get(fileToMove));
			}

			FileUtils.deleteDirectory(buildInstancesDir);

			return new InstancesTuple(serializedFiles.get(filesToMove[4]), serializedFiles.get(filesToMove[5]),
					serializedFiles.get(filesToMove[1]), serializedFiles.get(filesToMove[3]));

		} catch (IOException e) {
			e.printStackTrace();
		}

		return null;
	}

	private int getHashValueForBenchmarkTask(final BenchmarkTask benchmarkTask) {
		HashCodeBuilder hcb = new HashCodeBuilder();
		for (String placeholder : INSTANCES_HASH_PLACEHOLDERS) {
			try {
				String fileAsString = FileUtil
						.readFileAsString(benchmarkTask.getCandidateFolder() + File.separator + placeholder);
				hcb.append(fileAsString);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return hcb.toHashCode();
	}

	private BenchmarkTask getNextTask(final File taskFile) {
		if (taskFile.isDirectory() || !FilenameUtils.isExtension(taskFile.getAbsolutePath(), TASK_FILE_EXT)) {
			return null;
		}

		BenchmarkTask task;
		this.fileNameListLock.lock();
		try {
			if (!this.processedFileNames.contains(taskFile.getName())) {
				this.processedFileNames.add(taskFile.getName());
				task = BenchmarkTask.readFromTaskFile(taskFile);
				this.taskTempFolder = new File(
						TESTBED_DIR.getAbsoluteFile() + "_" + task.getCandidateFolder().getName());
				FileUtils.copyDirectory(TESTBED_DIR, this.taskTempFolder);
				return task;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			this.fileNameListLock.unlock();
		}
		return null;
	}

	@Override
	public void interrupt() {
		this.keepRunning = false;
	}

	public static void main(final String[] args) {
		ExecutorService threadPool = Executors.newFixedThreadPool(NUMBER_OF_THREADS);
		List<String> taskFilenameList = new LinkedList<>();
		Lock taskFileLock = new ReentrantLock(true);

		Lock instancesLockMapLock = new ReentrantLock();
		Map<Integer, Lock> instancesLockMap = new HashMap<>();
		Map<Integer, InstancesTuple> instancesTupleMap = new HashMap<>();

		File cachedInstancesCache = new File("cachedInstances");
		if (cachedInstancesCache.exists() && cachedInstancesCache.listFiles() != null) {
			log("Detected cached instances folder now search for cached data");
			for (File cachedInstancesDir : cachedInstancesCache.listFiles()) {
				Integer index = Integer.parseInt(cachedInstancesDir.getName());
				log("Index: " + index);

				InstancesTuple tuple = new InstancesTuple(
						new File(cachedInstancesDir.getAbsolutePath() + File.separator + "train.serialized"),
						new File(cachedInstancesDir.getAbsolutePath() + File.separator + "validation.serialized"),
						new File(cachedInstancesDir.getAbsolutePath() + File.separator + "contTrain.serialized"),
						new File(cachedInstancesDir.getAbsolutePath() + File.separator + "test.serialized"));

				if (!tuple.trainingData.exists() || !tuple.validationData.exists()
						|| !tuple.continuedTrainingData.exists() || !tuple.testData.exists()) {
					System.err.println("WARN: skipped cached instances with index " + index
							+ ", because some required files do not exist.");
					continue;
				}
				instancesLockMap.put(index, new ReentrantLock());
				instancesTupleMap.put(index, tuple);
				System.out.println("Read in cached instances entry with index " + index);
			}
		} else {
			log("No cached instances folder, so continue.");
		}

		IntStream.range(0, NUMBER_OF_THREADS).forEach(x -> threadPool.submit(new Benchmark("BenchmarkWorker#" + x,
				taskFilenameList, taskFileLock, instancesLockMapLock, instancesLockMap, instancesTupleMap)));

		System.err.println("Service up and running");
		String line;
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while ((line = br.readLine()) != null) {
				switch (line.trim()) {
				case "q":
					System.out.println("Shutdown benchmark worker threads");
					threadPool.shutdownNow();
					threadPool.awaitTermination(2000, TimeUnit.MILLISECONDS);
					System.exit(0);
					break;
				}
			}
		} catch (final IOException e) {
			e.printStackTrace();
		} catch (final InterruptedException e) {
			e.printStackTrace();
		}
	}

	private static void log(final String msg) {
		log(msg, true);
	}

	private static boolean lastLineBreak = true;

	private static void log(final String msg, final boolean linebreak) {
		final String prefix = "[" + Thread.currentThread().getName() + "] Benchmark Service: ";
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
}
