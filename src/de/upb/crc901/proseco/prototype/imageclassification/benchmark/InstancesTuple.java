package de.upb.crc901.proseco.prototype.imageclassification.benchmark;

import java.io.File;

public class InstancesTuple {

	public final File trainingData;
	public final File validationData;
	public final File continuedTrainingData;
	public final File testData;

	public InstancesTuple(final File pTrainingData, final File pValidationData, final File pContinuedTrainingData,
			final File pTestData) {
		this.trainingData = pTrainingData;
		this.validationData = pValidationData;
		this.continuedTrainingData = pContinuedTrainingData;
		this.testData = pTestData;
	}

}
