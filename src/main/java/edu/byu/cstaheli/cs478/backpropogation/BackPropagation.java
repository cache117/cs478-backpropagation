package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.Matrix;
import edu.byu.cstaheli.cs478.toolkit.SupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

/**
 * Created by cstaheli on 1/31/2017.
 */
public class BackPropagation extends SupervisedLearner
{
    @Override
    public void train(LearningStrategy strategy) throws Exception
    {
        Matrix trainingFeatures = strategy.getTrainingFeatures();
        Matrix trainingLabels = strategy.getTrainingLabels();
        boolean keepTraining = true;
        //for each epoch
        while (keepTraining)
        {
            //for each training data instance
            for (int i = 0; i < trainingFeatures.rows(); ++i)
            {
                for (int j = 0; j < trainingFeatures.cols(); ++j)
                {
                    analyzeInput(i, j);
                    //propagate error through the network
                    //adjust the weights
                    //calculate the accuracy over training data
                }
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

    private void analyzeInput(int i, int j)
    {

    }

    private boolean isThresholdValidationAccuracyMet(double validationAccuracy)
    {
        return validationAccuracy > 0;
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {

    }

    private class Node
    {

    }
}
