package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.Matrix;
import edu.byu.cstaheli.cs478.toolkit.exception.MatrixException;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class CrossValidationStrategy extends LearningStrategy
{
    private int begin;
    private int end;
    private Matrix trainingData;

    public CrossValidationStrategy(LearnerData learnerData) throws Exception
    {
        this(learnerData, 0, 0);

    }

    public CrossValidationStrategy(LearnerData learnerData, int begin, int end) throws Exception
    {
        super(learnerData);
        this.begin = begin;
        this.end = end;
        this.trainingData = getInitialTrainingData();
    }

    private Matrix getInitialTrainingData() throws MatrixException
    {
        Matrix matrix = new Matrix(getArffData(), 0, 0, begin, getArffData().cols());
        matrix.add(getArffData(), end, 0, getArffData().rows() - end);
        return matrix;
    }

    @Override
    public Matrix getTrainingData()
    {
        return new Matrix(trainingData, 0, 0, getTrainingSetSize() - 1, trainingData.cols());
    }

    @Override
    public Matrix getTestingData()
    {
        return new Matrix(getArffData(), begin, 0, end - begin, getArffData().cols());
    }

    @Override
    public Matrix getValidationData()
    {
        return new Matrix(trainingData, getTrainingSetSize(), 0, getValidationSetSize() - 1, trainingData.cols());
    }

    @Override
    protected int getTrainSize()
    {
        try
        {
            return getInitialTrainingData().rows();
        }
        catch (MatrixException e)
        {
            e.printStackTrace();
        }
        return 0;
    }
}
