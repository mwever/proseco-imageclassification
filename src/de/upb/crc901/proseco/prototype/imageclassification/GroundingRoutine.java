package de.upb.crc901.proseco.prototype.imageclassification;

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
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ProcessBuilder.Redirect;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import de.upb.crc901.proseco.PrototypeProperties;
import de.upb.crc901.proseco.prototype.GroundingUtil;
import jaicore.basic.PerformanceLogger;
import jaicore.ml.WekaUtil;
import weka.core.Instances;

/**
 * 
 * GroundingRoutine, grounding routine of the ImageClassification prototype.
 *
 */
public class GroundingRoutine {

	private static final String JAVA_EXTENSION = ".java";
	private static final String SERVICE_SRC_FILE = "ImageClassifier.java";
	private static final String BUILD_INSTANCES_SCRIPT = "buildInstances.bat";

	private static final String INSTANCES_PT_OUT = "instances.serialized";

	private static final String TRAINING_INSTANCES_FILE = "train.serialized";
	private static final String CONT_TRAINING_INSTANCES_FILE = "contTrain.serialized";
	private static final String VALIDATION_INSTANCES_FILE = "validation.serialized";
	private static final String TEST_INSTANCES_FILE = "test.serialized";

	private static final double VALIDATION_INSTANCES_FRACTION = 0.25;
	private static final double TEST_INSTANCES_FRACTION = 0.25;

	private final File placeHolderDir;
	private final File sourceInputDir;
	private final File sourceOutputDir;
	private final File serviceSourceFile;

	private String serviceSourceFileContent;

	/**
	 * This object maintains all configurations provided by the configuration
	 * file located in the config folder of the prototype.
	 */
	private final PrototypeProperties properties;

	/**
	 * Class for the grounding routine. The grounding routine is responsible for
	 * substituting code templates with given values. Furthermore, it compiles
	 * the code template and trains the resulting classifier.
	 *
	 * @param placeHolderDir
	 *            The directory the files for the placeholders of the code
	 *            template are located.
	 * @param sourceInputDir
	 *            The directory the plain code template is located.
	 * @param sourceOutputDir
	 *            The directory, the grounded code template shall be copied to
	 *            and where the source code is compiled to binaries and
	 *            executed.
	 * @param configFile
	 *            The file containing the configuration values for the grounding
	 *            routine.
	 */
	public GroundingRoutine(final File placeHolderDir, final File sourceInputDir, final File sourceOutputDir,
			final File configFile) {
		this.placeHolderDir = placeHolderDir;
		this.sourceInputDir = sourceInputDir;
		this.sourceOutputDir = sourceOutputDir;
		this.serviceSourceFile = new File(this.sourceInputDir.getAbsolutePath() + "/" + SERVICE_SRC_FILE);
		this.properties = new PrototypeProperties(configFile);

		log("Read service source file");
		this.serviceSourceFileContent = "";
		try (BufferedReader br = new BufferedReader(new FileReader(this.serviceSourceFile))) {
			String line;
			while ((line = br.readLine()) != null) {
				this.serviceSourceFileContent += line + "\n";
			}
		} catch (final FileNotFoundException e) {
			e.printStackTrace();
		} catch (final IOException e) {
			e.printStackTrace();
		}

		// copy all source files from the source input directory to the source
		// output directory
		File[] otherSourceFiles = sourceInputDir.listFiles(new FileFilter() {
			@Override
			public boolean accept(final File arg0) {
				return FilenameUtils.isExtension(arg0.getName(), "java") && !arg0.getName().equals(SERVICE_SRC_FILE);
			}
		});
		for (File otherSrcFile : otherSourceFiles) {
			try {
				FileUtils.copyFile(otherSrcFile,
						new File(sourceOutputDir.getAbsolutePath() + File.separator + otherSrcFile.getName()));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		// DONE with copying

		if (!placeHolderDir.exists() || !placeHolderDir.isDirectory()) {
			log("ERROR placeholder folder does not exist or is no directory");
			System.exit(-1);
		}

		if (!sourceOutputDir.exists() || !sourceOutputDir.isDirectory()) {
			log("ERROR source folder does not exist or is no directory");
			System.exit(-1);
		}
	}

	public void codeAssembly() {
		for (final File placeholder : this.placeHolderDir.listFiles()) {
			if (placeholder.isFile()) {
				final String placeholderVar = "/* $" + placeholder.getName() + "$ */";
				if (this.serviceSourceFileContent.contains(placeholderVar)) {
					try (final BufferedReader br = new BufferedReader(new FileReader(placeholder))) {
						String placeholderValue = "";
						String line;
						while ((line = br.readLine()) != null) {
							placeholderValue += line + "\n";
						}
						this.serviceSourceFileContent = this.serviceSourceFileContent.replace(placeholderVar,
								placeholderValue);
					} catch (final FileNotFoundException e) {
						e.printStackTrace();
					} catch (final IOException e) {
						e.printStackTrace();
					}
				}
			}
		}

		final File sourceOutputFile = new File(this.sourceOutputDir.getAbsolutePath() + "/" + SERVICE_SRC_FILE);
		try (final BufferedWriter bw = new BufferedWriter(new FileWriter(sourceOutputFile))) {
			bw.write(this.serviceSourceFileContent);
		} catch (final IOException e) {
			e.printStackTrace();
		}

		log("Code assembly DONE.");
	}

	public void compile() {
		try {
			final ProcessBuilder pb = new ProcessBuilder(
					GroundingUtil.compileJava("*" + JAVA_EXTENSION, this.properties.getProperty("classpath")))
							.directory(this.sourceOutputDir).redirectError(Redirect.INHERIT)
							.redirectOutput(Redirect.INHERIT);
			pb.start().waitFor();
			log("Compile solution DONE.");
		} catch (final IOException e) {
			e.printStackTrace();
		} catch (final InterruptedException e) {
			e.printStackTrace();
		}
	}

	public void trainModel(final File trainingData) {
		try {
			String[] command = GroundingUtil.executeJava(this.properties.getProperty("classname"),
					"-t " + trainingData.getCanonicalPath(), this.properties.getProperty("classpath"));
			final ProcessBuilder pb = new ProcessBuilder(command).directory(this.sourceOutputDir)
					.redirectError(Redirect.INHERIT).redirectOutput(Redirect.INHERIT);
			pb.start().waitFor();
			log("Train model DONE.");
		} catch (final IOException e) {
			e.printStackTrace();
		} catch (final InterruptedException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Use the compiled prototype to build numberOfInstancesToBuild many
	 * instances from the given dataFile.
	 *
	 * @param dataFile
	 *            The zip file containing the data that shall be transformed
	 *            into instances.
	 * @param numberOfInstancesToBuild
	 *            Number of instances that are desired to be built from the
	 *            given data file. A negative value or a value of zero indicates
	 *            to transform all data from the dataFile into instances.
	 */
	public void buildInstances(final File dataFile, final int numberOfInstancesToBuild) {
		PerformanceLogger.logStart("BuildInstances");
		try {
			// get the command for building the instances in order to execute
			// this command as a process
			String[] command = GroundingUtil.executeJava(this.properties.getProperty("classname"),
					"-i " + dataFile.getCanonicalPath() + " " + numberOfInstancesToBuild,
					this.properties.getProperty("classpath"));
			final ProcessBuilder pb = new ProcessBuilder(
					this.sourceOutputDir.getAbsolutePath() + File.separator + BUILD_INSTANCES_SCRIPT,
					dataFile.getCanonicalPath()).redirectError(Redirect.INHERIT).redirectOutput(Redirect.INHERIT);
			pb.start().waitFor();
			log("Buildind instances done.");

			if (numberOfInstancesToBuild <= 0) {
				try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
						this.sourceOutputDir.getAbsolutePath() + File.separator + INSTANCES_PT_OUT))) {
					Instances allInstances = (Instances) ois.readObject();
					List<Instances> stratifiedInstances = WekaUtil.getStratifiedSplit(allInstances, new Random(123),
							(1 - VALIDATION_INSTANCES_FRACTION - TEST_INSTANCES_FRACTION),
							VALIDATION_INSTANCES_FRACTION);

					final Instances trainingInstances = stratifiedInstances.get(0);
					final Instances validationInstances = stratifiedInstances.get(1);
					final Instances testInstances = stratifiedInstances.get(2);

					final Instances contTrainingInstances = new Instances(trainingInstances);
					contTrainingInstances.addAll(validationInstances);

					System.out.print("GroundingRoutine: Serialize instances...");
					writeInstances(trainingInstances,
							new File(this.sourceOutputDir + File.separator + TRAINING_INSTANCES_FILE));
					writeInstances(validationInstances,
							new File(this.sourceOutputDir + File.separator + VALIDATION_INSTANCES_FILE));
					writeInstances(testInstances,
							new File(this.sourceOutputDir + File.separator + TEST_INSTANCES_FILE));
					writeInstances(contTrainingInstances,
							new File(this.sourceOutputDir + File.separator + CONT_TRAINING_INSTANCES_FILE));
					System.out.println("DONE.");

					/* save examples as ARFF */
					File arffTrainExport = new File(this.sourceOutputDir + File.separator + "train.arff");
					try (BufferedWriter bw = new BufferedWriter(new FileWriter(arffTrainExport))) {
						bw.write(trainingInstances.toString());
					} catch (IOException e) {
						e.printStackTrace();
					}
					File arffExport = new File(this.sourceOutputDir + File.separator + "allInstances.arff");
					try (BufferedWriter bw = new BufferedWriter(new FileWriter(arffExport))) {
						bw.write(allInstances.toString());
					} catch (IOException e) {
						e.printStackTrace();
					}

				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				}
			}
		} catch (final IOException e) {
			e.printStackTrace();
		} catch (final InterruptedException e) {
			e.printStackTrace();
		}
		PerformanceLogger.logEnd("BuildInstances");
	}

	private static void writeInstances(final Instances instances, final File file) {
		if (file.getParentFile() != null) {
			file.getParentFile().mkdirs();
		}

		try (final ObjectOutputStream objectStream = new ObjectOutputStream(new FileOutputStream(file))) {
			objectStream.writeObject(instances);
		} catch (final FileNotFoundException e) {
			e.printStackTrace();
		} catch (final IOException e) {
			e.printStackTrace();
		}
	}

	public static void main(final String[] args) {
		long startTime;
		if (args.length != 3) {
			// TODO: Correct usage message
			log("Correct usage: ");
			System.exit(-1);
		}

		PerformanceLogger.logStart("TotalRuntime");

		final File placeholderFolder = new File(args[0]);
		final File sourceInputFolder = new File(args[1]);
		final File sourceOutputFolder = new File(args[2]);
		final File configFile = new File("config/groundingroutine.conf");

		GroundingRoutine gr = new GroundingRoutine(placeholderFolder, sourceInputFolder, sourceOutputFolder,
				configFile);

		/* Assemble the code by substituting the placeholders */
		gr.codeAssembly();

		/* Compile the assembled code */
		gr.compile();

		gr.trainModel(new File(".." + File.separator + "params" + File.separator + "classifierdef" + File.separator
				+ "instances.serialized"));

		PerformanceLogger.saveGlobalLogToFile(new File("GroundingRoutine.log"));
	}

	private static void log(final String msg) {
		final String prefix = "Grounding Routine: ";
		System.out.println(prefix + msg);
	}

}
