package de.upb.crc901.proseco.prototype.imageclassification.benchmark;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;

import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EBuildPhase;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EDataFraction;

public class BenchmarkRcv {
	private static final String TASK_DIRECTORY = "task/";
	private static final String TASK_FILE_PREFIX = "task_";
	private static final String TASK_FILE_EXT = ".task";
	private static final String TASK_FILE_TEMP_EXT = ".temptask";

	public static void main(final String[] args) {
		if (args.length != 3) {
			System.out.println("Correct Usage: java BenchmarkRcv [phase] [candidateDirectory] [SAMPLE|FULL]");
			System.exit(1);
		}

		BenchmarkTask task = new BenchmarkTaskBuilder().setCandidateFolder(new File(args[1]))
				.setBuildPhase(EBuildPhase.valueOf(args[0])).setDataFraction(EDataFraction.valueOf(args[2])).build();

		if (!task.getCandidateFolder().exists() || !task.getCandidateFolder().isDirectory()) {
			System.out.println("Given candidate directory does not exist or is not a directory.");
		}

		String taskFilename = TASK_DIRECTORY + TASK_FILE_PREFIX + task.getCandidateFolder().getName();

		final File taskTempFile = new File(taskFilename + TASK_FILE_TEMP_EXT);
		final File taskFile = new File(taskFilename + TASK_FILE_EXT);
		taskTempFile.getParentFile().mkdirs();

		if (taskFile.exists()) {
			return;
		}

		try {
			BenchmarkTask.writeTaskToFile(task, taskTempFile);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		try {
			FileUtils.moveFile(taskTempFile, taskFile);
		} catch (final IOException e) {
			e.printStackTrace();
		}
	}

}
