package de.upb.crc901.proseco.prototype.imageclassification.benchmark.runner;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.lang.ProcessBuilder.Redirect;

import de.upb.crc901.proseco.PrototypeProperties;
import de.upb.crc901.proseco.prototype.imageclassification.GroundingRoutine;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.featureextraction.FeatureExtractionEvaluator;
import weka.core.Instances;

public class FeatureExtractionBenchmarkRunner extends AbstractBenchmarkRunner {

	private final File dataFile;
	private final int NUMBER_OF_INSTANCES = 50;
	private final FeatureExtractionEvaluator evaluator;
	
	private static final PrototypeProperties PROPS = new PrototypeProperties(
			".." + File.separator + "config" + File.separator + "benchmarkservice.conf");
	private static final String FVALUE_FILE = PROPS.getProperty("name_fvaluefile");

	public FeatureExtractionBenchmarkRunner(final BenchmarkTask pTask, final GroundingRoutine pGroundingRoutine,
			final File pTaskTempFolder, final File pDataFile, final FeatureExtractionEvaluator pEvaluator) {
		super(pTask, pGroundingRoutine, pTaskTempFolder);
		this.dataFile = pDataFile;
		this.evaluator = pEvaluator;
	}

	@Override
	public void run() {
		this.getGroundingRoutine().codeAssembly();

		this.getGroundingRoutine().compile();

		try {
			Process createInstances = new ProcessBuilder()
					.command(this.getTaskTempFolder().getAbsolutePath() + File.separator + "buildInstances.bat",
							this.dataFile.getAbsolutePath(), this.NUMBER_OF_INSTANCES + "")
					.redirectError(Redirect.INHERIT).redirectOutput(Redirect.INHERIT).start();
			createInstances.waitFor();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		double fValue = 0.0;
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
				new File(this.getTaskTempFolder().getAbsolutePath() + File.separator + "instances.serialized")))) {
			Instances builtInstances = (Instances) ois.readObject();
			fValue = this.evaluator.evaluate(builtInstances);
			System.out.println(fValue);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

		this.writeFValue(fValue);

	}

	private void writeFValue(final double fValue) {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(
				new File(this.getTaskTempFolder().getAbsolutePath() + File.separator + FVALUE_FILE)))) {
			bw.write(fValue + "\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
