package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.Matrix;
import edu.byu.cstaheli.cs478.toolkit.RandomLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

import java.util.*;

/**
 * Created by cstaheli on 1/31/2017.
 */
public class BackPropagation extends RandomLearner
{
    private List<Node> hiddenLayer;
    private List<Node> outputLayer;
    private List<Double> inputs;

    public BackPropagation(Random rand, MLSystemManager manager)
    {
        super(rand, manager);
    }

    @Override
    public void train(LearningStrategy strategy) throws Exception
    {
        Matrix trainingFeatures = strategy.getTrainingFeatures();
        Matrix trainingLabels = strategy.getTrainingLabels();
        initializeWeights(trainingFeatures.cols(), trainingFeatures.valueCount(0));
        boolean keepTraining = true;
        //for each epoch
        while (keepTraining)
        {
            //for each training data instance
            for (int i = 0; i < trainingFeatures.rows(); ++i)
            {
                analyzeInputRow(trainingFeatures.row(i));
                //propagate error through the network
                //adjust the weights
                //calculate the accuracy over training data
            }
            //for each validation data instance
            Matrix validationFeatures = strategy.getValidationFeatures();
            Matrix validationLabels = strategy.getValidationLabels();
            //calculate the accuracy over the validation data
            double validationAccuracy = measureAccuracy(validationFeatures, validationLabels, null);
            //if the threshold validation accuracy is met
            if (isThresholdValidationAccuracyMet(validationAccuracy))
            {
                //exit training
                keepTraining = false;
            }
            else
            {
                //continue training
                keepTraining = true;
            }
        }
    }

    private void initializeWeights(int features, int outputs)
    {
        int numberOfNodesInHiddenLayer = getNumberOfNodesInHiddenLayer(features, outputs);
        hiddenLayer = new ArrayList<>(numberOfNodesInHiddenLayer);
        outputLayer = new ArrayList<>(outputs);
        for (int i = 0; i < numberOfNodesInHiddenLayer; ++i)
        {
            hiddenLayer.add(new Node(getRandomWeight(), getRandomWeight()));
        }
    }

    private void analyzeInputRow(double[] row)
    {
        for (double item : row)
        {
            inputs.add(item);
        }
        for (Node hiddenNode: hiddenLayer)
        {

        }
        for (Node outputNode: outputLayer)
        {

        }
        for (Node hiddenNode: hiddenLayer)
        {

        }
    }

    private int getNumberOfNodesInHiddenLayer(int features, int outputs)
    {
        return (features * 2) + outputs;
    }

    private boolean isThresholdValidationAccuracyMet(double validationAccuracy)
    {
        return validationAccuracy > 0;
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {

    }


}
