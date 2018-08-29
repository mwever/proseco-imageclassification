package de.upb.crc901.proseco.prototype.imageclassification.benchmark;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.commons.lang3.builder.EqualsBuilder;

public class BenchmarkTask {

	public enum EBuildPhase {
		FEATURE_EXTRACTION, CLASSIFIER_DEF;
	}

	public enum EDataFraction {
		SAMPLE, FULL;
	}

	private final File candidateFolder;
	private final EBuildPhase buildPhase;
	private final EDataFraction dataFraction;

	public BenchmarkTask(final File pCandidateFolder, final EBuildPhase pBuildPhase, final EDataFraction pDataFraction) {
		this.candidateFolder = pCandidateFolder;
		this.buildPhase = pBuildPhase;
		this.dataFraction = pDataFraction;
	}

	public File getCandidateFolder() {
		return this.candidateFolder;
	}

	public EBuildPhase getBuildPhase() {
		return this.buildPhase;
	}

	public EDataFraction getDataFraction() {
		return this.dataFraction;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		sb.append("candidateFolder=" + this.candidateFolder.getAbsolutePath()+"\n");
		sb.append("buildPhase=" + this.buildPhase+"\n");
		sb.append("dataFraction=" + this.dataFraction+"\n");

		return sb.toString();
	}

	@Override
	public boolean equals(final Object obj) {
		if(!(obj instanceof BenchmarkTask)) {
			return false;
		}
		BenchmarkTask other = (BenchmarkTask) obj;

		return new EqualsBuilder().append(this.candidateFolder.getAbsolutePath(), other.candidateFolder.getAbsolutePath()).append(this.dataFraction, other.dataFraction).append(this.buildPhase, other.buildPhase).isEquals();
	}

	public static BenchmarkTask readFromTaskFile(final File taskFile) throws FileNotFoundException, IOException {
		File candidateFolder = null;
		EBuildPhase buildPhase = null;
		EDataFraction dataFraction = null;

		try(BufferedReader br = new BufferedReader(new FileReader(taskFile))) {
			String line;
			while((line = br.readLine()) != null) {
				if(line.trim().equals("") || line.trim().startsWith("#")) {
					continue;
				}

				String[] lineSplit = line.split("=");
				switch(lineSplit[0].trim()) {
				case "candidateFolder":
					candidateFolder = new File(lineSplit[1].trim());
					break;
				case "buildPhase":
					buildPhase = EBuildPhase.valueOf(lineSplit[1].trim());
					break;
				case "dataFraction":
					dataFraction = EDataFraction.valueOf(lineSplit[1].trim());
					break;
				}
			}
		}
		return new BenchmarkTask(candidateFolder, buildPhase, dataFraction);
	}

	public static void writeTaskToFile(final BenchmarkTask pTask, final File pTargetFile) throws IOException {
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(pTargetFile))) {
			bw.write(pTask.toString()+"\n");
		}
	}

}
