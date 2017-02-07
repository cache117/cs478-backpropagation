package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.Matrix;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class StaticStrategy extends LearningStrategy
{
    private Matrix testData;

    public StaticStrategy(LearnerData learnerData) throws Exception
    {
        super(learnerData);
        testData = new Matrix();
        testData.loadArff(learnerData.getEvalParameter());
        if (learnerData.isNormalized())
            testData.normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!
    }

    @Override
    public Matrix getTrainingData()
    {
        return new Matrix(getArffData(), 0, 0, getTrainingSetSize(), getArffData().cols());
    }

    @Override
    public Matrix getTestingData()
    {
        return testData;
    }

    @Override
    public Matrix getValidationData()
    {
        return new Matrix(getArffData(), getTrainingSetSize(), 0, getValidationSetSize(), getArffData().cols());
    }
}
