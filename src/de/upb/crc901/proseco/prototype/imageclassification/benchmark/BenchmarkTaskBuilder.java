package de.upb.crc901.proseco.prototype.imageclassification.benchmark;

import java.io.File;

import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EBuildPhase;
import de.upb.crc901.proseco.prototype.imageclassification.benchmark.BenchmarkTask.EDataFraction;

public class BenchmarkTaskBuilder {

	private File candidateFolder = null;
	private EDataFraction dataFraction = EDataFraction.FULL;
	private EBuildPhase buildPhase = EBuildPhase.CLASSIFIER_DEF;

	public BenchmarkTaskBuilder() {

	}

	public BenchmarkTaskBuilder setCandidateFolder(final File pCandidateFolder) {
		this.candidateFolder = pCandidateFolder;
		return this;
	}

	public BenchmarkTaskBuilder setDataFraction(final EDataFraction pDataFraction) {
		this.dataFraction = pDataFraction;
		return this;
	}

	public BenchmarkTaskBuilder setBuildPhase(final EBuildPhase pBuildPhase) {
		this.buildPhase = pBuildPhase;
		return this;
	}

	public BenchmarkTask build() {
		if(this.candidateFolder==null) {
			throw new IllegalArgumentException("Candidate folder needs to be set before building.");
		}
		return new BenchmarkTask(this.candidateFolder, this.buildPhase, this.dataFraction);
	}

}
