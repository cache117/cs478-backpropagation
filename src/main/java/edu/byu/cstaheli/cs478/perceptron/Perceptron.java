package edu.byu.cstaheli.cs478.perceptron;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.Matrix;
import edu.byu.cstaheli.cs478.toolkit.RandomLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

import java.util.Random;

/**
 * Created by cstaheli on 1/17/2017.
 */
public class Perceptron extends RandomLearner
{
    private static final int EPOCHS_WITHOUT_SIGNIFICANT_IMPROVEMENT = 5;
    private double[] weights;
    private int epochsWithoutSignificantImprovement;

    public Perceptron(Random rand, MLSystemManager manager)
    {
        super(rand, manager);
        setLearningRate(.1);
        this.setManager(manager);
    }

    @Override
    public void train(LearningStrategy strategy) throws Exception
    {
        super.train(strategy);
    }

    protected double getExpected(Matrix labels, int row)
    {
        return labels.get(row, 0);
    }

    protected double getActivation(double[] rowWeights, double[] row)
    {
        double sum = 0;
        for (int i = 0; i < row.length; ++i)
        {
            sum += (rowWeights[i] * row[i]);
        }
        return sum > 0 ? 1 : 0;
    }

    private boolean isAccuracyChangeLargeEnough(double previousAccuracy, double currentAccuracy)
    {
        return (currentAccuracy - previousAccuracy) >= .005;
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        labels[0] = getActivation(getWeights(), features);
    }

    @Override
    protected void initializeWeights(int features, int outputs)
    {
        setWeights(new double[features]);
        for (int i = 0; i < features; ++i)
        {
            getWeights()[i] = getRandomWeight();
        }
    }

    @Override
    protected void analyzeInputRow(double[] row, double expectedOutput)
    {
        for (int i = 0; i < row.length; ++i)
        {
            double input = row[i];
            double actual = getActivation(getWeights(), row);
            double newWeight = calcNewWeight(getWeights()[i], getLearningRate(), expectedOutput, actual, input);
            getWeights()[i] = newWeight;
        }
    }

    @Override
    protected boolean isThresholdValidationAccuracyMet(double previousAccuracy, double validationAccuracy)
    {
        if (!isAccuracyChangeLargeEnough(previousAccuracy, validationAccuracy))
        {
            if (++epochsWithoutSignificantImprovement >= EPOCHS_WITHOUT_SIGNIFICANT_IMPROVEMENT)
            {
                return true;
            }
        }
        else
        {
            epochsWithoutSignificantImprovement = 0;
            return false;
        }
        return false;
    }

    private double calcNewWeight(double oldWeight, double learningRate, double expected, double actual, double input)
    {
        return oldWeight - learningRate * (actual - expected) * input;
    }

    public double[] getWeights()
    {
        return weights;
    }

    public void setWeights(double[] weights)
    {
        this.weights = weights;
    }
}
