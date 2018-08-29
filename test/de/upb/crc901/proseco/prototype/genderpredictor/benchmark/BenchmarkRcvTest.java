package de.upb.crc901.proseco.prototype.genderpredictor.benchmark;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.junit.AfterClass;
import org.junit.Test;

import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkRcv;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTaskBuilder;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EBuildPhase;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EDataFraction;

public class BenchmarkRcvTest {

	@Test
	public void mainTest() throws FileNotFoundException, IOException {
		File candidateFolder = new File("testrsc/temp/candidate0");
		candidateFolder.mkdirs();

		BenchmarkTask task = new BenchmarkTaskBuilder().setCandidateFolder(candidateFolder)
				.setBuildPhase(EBuildPhase.FEATURE_EXTRACTION).setDataFraction(EDataFraction.SAMPLE).build();

		// setup arguments
		String[] args = new String[3];
		args[0] = task.getBuildPhase().toString();
		args[1] = task.getCandidateFolder().getAbsolutePath();
		args[2] = task.getDataFraction().toString();

		File taskFolder = new File("task/");
		if (taskFolder.exists()) {
			try {
				FileUtils.deleteDirectory(taskFolder);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		BenchmarkRcv.main(args);
		assert (taskFolder.exists()) : "Task output folder was not created.";

		File[] taskFiles = taskFolder.listFiles();
		BenchmarkTask taskDes = BenchmarkTask.readFromTaskFile(taskFiles[0]);

		assert (taskDes.equals(task)) : "Deserialized task does not equal original task!";

	}

	@AfterClass
	public static void cleanUp() throws IOException {
		FileUtils.deleteDirectory(new File("task/"));
		FileUtils.deleteDirectory(new File("testrsc/temp/"));
	}

}
