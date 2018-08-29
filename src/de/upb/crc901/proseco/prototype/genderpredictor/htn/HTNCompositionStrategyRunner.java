package de.upb.crc901.proseco.prototype.genderpredictor.htn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.ProcessBuilder.Redirect;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.proseco.PrototypeProperties;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EBuildPhase;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EDataFraction;
import de.upb.crc901.taskconfigurator.core.CodePlanningUtil;
import de.upb.crc901.taskconfigurator.core.MLUtil;
import de.upb.crc901.taskconfigurator.core.SolutionEvaluator;
import de.upb.crc901.taskconfigurator.search.algorithms.PipelineSearcher;
import de.upb.crc901.taskconfigurator.search.evaluators.RandomCompletionEvaluator;
import jaicore.basic.PerformanceLogger;
import jaicore.graphvisualizer.SimpleGraphVisualizationWindow;
import jaicore.planning.graphgenerators.task.ceoctfd.CEOCTFDGraphGenerator;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.planning.graphgenerators.task.tfd.TFDTooltipGenerator;
import jaicore.planning.model.ceoc.CEOCAction;
import jaicore.search.algorithms.parallel.parallelexploration.distributed.interfaces.SerializableNodeEvaluator;
import jaicore.search.algorithms.standard.bestfirst.BestFirst;
import jaicore.search.algorithms.standard.core.NodeEvaluator;
import jaicore.search.structure.core.Node;
import weka.core.Instances;

/**
 * This program searches the "inputs/classifierdef" folder of its execution for
 * a file called "instances.serialized". This file is supposed to store a
 * serialized object of the WEKA class Instances.
 *
 * It then invokes the AutoML machine to construct a good chain of WEKA tools.
 *
 * In a final step, this chain is serialized into Java code that is stored to
 * "classifierdef" of the "outputs" folder
 *
 */
public class HTNCompositionStrategyRunner implements SolutionEvaluator {

	private static final Logger logger = LoggerFactory.getLogger(HTNCompositionStrategyRunner.class);
	private static final PrototypeProperties PROPS = new PrototypeProperties("conf/htncompositionstrategyrunner.conf");

	private static final boolean SHOW_GRAPH = Boolean.parseBoolean(PROPS.getProperty("show_graph"));
	private static final int NUMBER_OF_CONSIDERED_SOLUTIONS = Integer
			.parseInt(PROPS.getProperty("number_of_considered_solutions"));
	private static final int EVALUATION_SAMPLE_SIZE = Integer.parseInt(PROPS.getProperty("evaluation_sample_size"));

	private final static String NAME_PLACEHOLDER_IMAGEFILTER = PROPS.getProperty("name_placeholder_imagefilter");
	private final static String NAME_PLACEHOLDER_FEATUREEXTRACTION = PROPS
			.getProperty("name_placeholder_featureextraction");
	private final static String NAME_PLACEHOLDER_CLASSIFICATION = PROPS.getProperty("name_placeholder_classification");
	private final static String NAME_PARAM = PROPS.getProperty("name_param");
	private static final String NAME_FVALUE = PROPS.getProperty("name_fvalue");

	private final File benchmarkFile;
	private final File outputFolder;
	private final Map<Map<String, String>, Integer> fValueMap = new HashMap<>();
	private String preprocessingSolution;
	private int bestPreprocessingPlaceholderFScore = Integer.MAX_VALUE;
	private String featureExtractingSolution;

	public HTNCompositionStrategyRunner(final File outputFolder, final File benchmarkFile) {
		super();
		this.outputFolder = outputFolder;
		this.benchmarkFile = benchmarkFile;
	}

	public static void main(final String[] args) throws ClassNotFoundException, FileNotFoundException, IOException {
		Thread.currentThread().setName("HTNCompositionStrategy Main");

		/* read in input and output folder specifications */
		if (args.length != 3) {
			logger.error(
					"Invalid usage of composition Strategy. Provide three params: \"input folder\", \"output folder\", and \"benchmark executable\"");
			return;
		}
		// final File paramFile = new File(args[0] + File.separator + NAME_PLACEHOLDER +
		// File.separator + NAME_PARAM);
		// if (!paramFile.exists()) {
		// if (!paramFile.getParentFile().exists()) {
		// logger.error("Invalid usage of composition Strategy. Please make sure that
		// the first param (input folder) exists");
		// } else {
		// logger.error("Invalid usage of composition Strategy. Please make sure that
		// the input folder contains the file " + NAME_PLACEHOLDER);
		// }
		// return;
		// }
		PerformanceLogger.logStart("StrategyTotalRun");
		logger.info("Running HTN Composition Strategy.");

		/* get data */
		@SuppressWarnings("resource")
		// final Instances data = (Instances) new ObjectInputStream(new
		// BufferedInputStream(new FileInputStream(paramFile))).readObject();

		/* compute java code */
		final HTNCompositionStrategyRunner strategy = new HTNCompositionStrategyRunner(new File(args[1]),
				new File(args[2]));
		final Map<String, String> javaCode = strategy.getPlaceholderValues();
		System.out.println("Final solution: " + javaCode);
		strategy.writeSolution(javaCode);

		PerformanceLogger.logEnd("StrategyTotalRun");
		PerformanceLogger.saveGlobalLogToFile(new File("HTNCompositionStrategy.log"));
		System.out.println("Strategy is ready ...");
		
	}

	public Map<String, String> getPlaceholderValues() {

		/* solve composition problem */
		final Random random = new Random(0);
		final Map<String, String> placeholderValues = new HashMap<>(2);

		/* PHASE 1: Searching for best preprocessor */
		this.getPreprocessingPipeline();

		System.out.println("Proceeding with next phase...");

		/* PHASE 2: Searching for best classifier */
		final SerializableNodeEvaluator<TFDNode, Integer> nodeEval = new RandomCompletionEvaluator(random,
				EVALUATION_SAMPLE_SIZE, this, true);
		final BestFirst<TFDNode, String> searchAlgo = new BestFirst<>(
				MLUtil.getGraphGenerator(new File("htn.searchspace")), nodeEval);
		final PipelineSearcher optimizer = new PipelineSearcher(searchAlgo, random, NUMBER_OF_CONSIDERED_SOLUTIONS,
				SHOW_GRAPH);
		
		/* get the first entry in the list of solutions (they are ordered by the f-value) */
		final List<CEOCAction> pipelineDescription = optimizer.getPipelineDescriptions().get(0);

		/* derive Java code from the plan (this is the recipe) */
		placeholderValues.put(NAME_PLACEHOLDER_IMAGEFILTER, this.preprocessingSolution);
		placeholderValues.put(NAME_PLACEHOLDER_FEATUREEXTRACTION, this.featureExtractingSolution);
		placeholderValues.put(NAME_PLACEHOLDER_CLASSIFICATION, MLUtil.getJavaCodeFromPlan(pipelineDescription));
		return placeholderValues;
	}

	private void getPreprocessingPipeline() {
		CEOCTFDGraphGenerator generator = MLUtil.getGraphGenerator(new File("imagefilter.searchspace"));
		NodeEvaluator<TFDNode, Integer> nodeEval = new RandomCompletionEvaluator(new Random(0), 1, new SolutionEvaluator() {
			
			@Override
			public void setTrainingData(Instances train) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public void setControlData(Instances validation) {
				// TODO Auto-generated method stub
				
			}
			
			@Override
			public int getSolutionScore(List<CEOCAction> plan) throws Exception {

				System.out.println("Code:\n----------------------------------\n");
				List<List<CEOCAction>> planParts = getCodeParts(plan);
				String imageFilterCode = getFeatureExtractionCode(planParts.get(0));
				String featureExctractionCode = getFeatureExtractionCode(planParts.get(1));
				System.out.println(featureExctractionCode);

				/* invoke benchmark */
				final String key = String.valueOf(System.currentTimeMillis());
				final Map<String, String> placeholderValues = new HashMap<>();
				placeholderValues.put(NAME_PLACEHOLDER_IMAGEFILTER, imageFilterCode);
				System.out.println("#####IMAGE FILTER CODE " + imageFilterCode);
				placeholderValues.put(NAME_PLACEHOLDER_FEATUREEXTRACTION, featureExctractionCode);
				
				try {
					int score = callBenchmark(placeholderValues, key);
					if (score < bestPreprocessingPlaceholderFScore) {
						preprocessingSolution = imageFilterCode;
						bestPreprocessingPlaceholderFScore = score;
						featureExtractingSolution = featureExctractionCode;
						System.out.println("updated preprocessing solution");
					}
					System.out.println("SCORE IS: " + score);
					return score;
				} catch (Exception e) {
					System.err.println("Cannot compute score due to exception:");
					e.printStackTrace();
					return Integer.MAX_VALUE;
				}
			}
		}, false);
		BestFirst<TFDNode, String> bf = new BestFirst<>(generator, nodeEval);
		SimpleGraphVisualizationWindow<Node<TFDNode, Integer>> window = new SimpleGraphVisualizationWindow<>(
				bf.getEventBus());
		window.getPanel().setTooltipGenerator(new TFDTooltipGenerator());
		
		/* identify next solution. The actually used code fragments are stored directly in the HTNCompositionStrategyRunner by the node evaluation function */
		List<TFDNode> solution = bf.nextSolution();
	}

	private String getFeatureExtractionCode(final List<CEOCAction> plan) {
		StringBuilder codeBuilder = new StringBuilder();
		for (CEOCAction a : plan) {
			switch (a.getOperation().getName()) {
			case "scale": {
				String input = a.getParameters().get(0).getName();
				codeBuilder.append("new Catalano.Imaging.Filters.Crop(0, 0, min, min).ApplyInPlace(" + input + ");\n");
				codeBuilder.append("new Catalano.Imaging.Filters.Resize(250, 250).applyInPlace(" + input + ");\n");
				break;
			}
			default: {
				String codeForAction = CodePlanningUtil.getCodeForAction(a);
				codeBuilder.append(codeForAction);
				codeBuilder.append("\n");

				/*
				 * append code snippet to assign the binary pattern stuff to the respective
				 * variable in the code
				 */
				if (a.getOperation().getName().startsWith("Catalano.Imaging.Texture.BinaryPattern.")) {
					String[] split = codeForAction.split(" ");
					codeBuilder.append("bp = " + split[1] + ";\n");
				}
			}
			}
		}
		return codeBuilder.toString();
	}

	@Override
	public int getSolutionScore(final List<CEOCAction> plan) throws Exception {
		if (this.preprocessingSolution == null) {
			throw new IllegalStateException("Trying to compute solution score before preprocessing has been fixed!");
		}

		PerformanceLogger.logStart("getF");

		/* write down solution (the preprocessing is fixed here) */
		final String key = String.valueOf(System.currentTimeMillis());
		final Map<String, String> placeholderValues = new HashMap<>();
		placeholderValues.put(NAME_PLACEHOLDER_IMAGEFILTER, this.preprocessingSolution);
		placeholderValues.put(NAME_PLACEHOLDER_FEATUREEXTRACTION, this.featureExtractingSolution);
		placeholderValues.put(NAME_PLACEHOLDER_CLASSIFICATION, MLUtil.getJavaCodeFromPlan(plan));
		return this.callBenchmark(placeholderValues, key);
	}

	private int callBenchmark(final Map<String, String> placeholderValues, final String key) {

		/* write code for solution and define folder where we expect the results */
		this.writeSolution(placeholderValues, key);
		final File candidateFolder = new File(this.outputFolder.getAbsolutePath() + File.separator + key);

		/*
		 * define benchmark call depending on whether the classifier has been defined or
		 * not
		 */
		final ProcessBuilder pb;
		if (placeholderValues.containsKey(NAME_PLACEHOLDER_CLASSIFICATION)) {
			pb = new ProcessBuilder(this.benchmarkFile.getAbsolutePath(), EBuildPhase.CLASSIFIER_DEF.toString(),
					candidateFolder.getAbsolutePath(), EDataFraction.FULL.toString()).redirectError(Redirect.INHERIT)
							.redirectOutput(Redirect.INHERIT);
		} else {
			pb = new ProcessBuilder(this.benchmarkFile.getAbsolutePath(), EBuildPhase.FEATURE_EXTRACTION.toString(),
					candidateFolder.getAbsolutePath(), EDataFraction.SAMPLE.toString()).redirectError(Redirect.INHERIT)
							.redirectOutput(Redirect.INHERIT);
		}

		System.out.println("Executing: " + pb.command());

		/* call benchmark and await termination */
		System.out.println("Compute f value for current testbed");
		Process fValueProcess;
		try {
			fValueProcess = pb.start();
			fValueProcess.waitFor();

			final File fValueFile = new File(candidateFolder.getAbsolutePath() + File.separator + NAME_FVALUE);
			boolean resultAvailable = false;

			while (!resultAvailable) {
				if (fValueFile.exists()) {
					resultAvailable = true;
				} else {
					Thread.sleep(100);
				}
			}

			if (fValueFile.exists() && fValueFile.isFile()) {
				try (BufferedReader br = new BufferedReader(new FileReader(fValueFile))) {
					PerformanceLogger.logEnd("getF");
					final int fValue = (int) ((1 - Double.parseDouble(br.readLine())) * FVALUE_ACCURACY);
					this.fValueMap.put(placeholderValues, fValue);
					return fValue;
				}
			} else {
				System.out.println("Could not compute f Value");
			}

		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
		PerformanceLogger.logEnd("getF");
		System.out.println("Finished computation ...");

		/* read in result of the benchmark process */
		this.fValueMap.put(placeholderValues, 10000);
		return 10000;
	}

	private static final double FVALUE_ACCURACY = 10000f;

	public void writeSolution(final Map<String, String> placeholderValues) {
		this.writeSolution(placeholderValues, "");

		try (FileWriter fw = new FileWriter(new File(this.outputFolder + File.separator + NAME_FVALUE))) {
			System.out.println(this.fValueMap.get(placeholderValues));
			fw.write((1 - (this.fValueMap.get(placeholderValues) / FVALUE_ACCURACY)) + "\n");
		} catch (final IOException e) {
			System.out.println("Failed to write fvalue");
			e.printStackTrace();
		}
	}

	public void writeSolution(final Map<String, String> placeholderValues, final String subfolder) {

		if (placeholderValues.containsValue(null)) {
			throw new IllegalArgumentException("Placeholder values must not be null!");
		}

		/* create folder */
		File folder = new File(this.outputFolder + (!(subfolder.equals("")) ? File.separator + subfolder : ""));
		if (!folder.exists()) {
			folder.mkdirs();
		}

		// /* copy image filter bat */
		// try {
		// FileUtils.copyFile(new File("imagefilter"), new File(folder + File.separator
		// + "imagefilter"));
		// } catch (IOException e1) {
		// e1.printStackTrace();
		// }

		/* write placeholder values */
		for (String placeholder : placeholderValues.keySet()) {
			final File targetFile = new File(folder + File.separator + placeholder);
			try (FileWriter fw = new FileWriter(targetFile)) {
				fw.write(this.rewriteJavaCode(placeholderValues.get(placeholder)));
			} catch (final IOException e) {
				System.out.println("Failed to write solution " + targetFile.getAbsolutePath());
				e.printStackTrace();
			}
		}
	}

	public String rewriteJavaCode(String code) {
		if (code == null) {
			throw new IllegalArgumentException("Code must not be NULL");
		}

		/* append classifier definition */
		final Pattern p = Pattern.compile("weka\\.classifiers\\.[^ ]*([^=]*)");
		final Matcher m = p.matcher(code);
		if (m.find()) {
			final String classifierVar = m.group(1).trim();
			code += "c = " + classifierVar + ";\n";
		} else {
			System.err.println("No classifier definition found. Cannot assign variable to c.");
		}
		return code;
	}

	private List<List<CEOCAction>> getCodeParts(final List<CEOCAction> plan) {
		List<CEOCAction> part1 = new ArrayList<>();
		List<CEOCAction> part2 = new ArrayList<>();
		boolean sepReached = false;
		for (CEOCAction a : plan) {
			if (a.getOperation().getName().equals("separator")) {
				sepReached = true;
			} else {
				if (!sepReached) {
					part1.add(a);
				} else {
					part2.add(a);
				}
			}
		}

		List<List<CEOCAction>> parts = new ArrayList<>();
		parts.add(part1);
		parts.add(part2);
		return parts;
	}

	@Override
	public void setTrainingData(final Instances train) {

		/*
		 * we ignore this here, because the training and test data is already contained
		 * in the benchmark anyway
		 */
	}

	@Override
	public void setControlData(final Instances validation) {

		/*
		 * we ignore this here, because the training and test data is already contained
		 * in the benchmark anyway
		 */
	}
}
