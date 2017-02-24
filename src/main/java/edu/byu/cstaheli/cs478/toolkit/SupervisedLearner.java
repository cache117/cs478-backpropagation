package edu.byu.cstaheli.cs478.toolkit;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

public abstract class SupervisedLearner
{
    private int totalEpochs;
    private MLSystemManager manager;
    private double learningRate;

    protected SupervisedLearner(MLSystemManager manager)
    {
        totalEpochs = 0;
        this.setManager(manager);
        setLearningRate(.1);
    }

    public void train(LearningStrategy strategy) throws Exception
    {
        Matrix trainingFeatures = strategy.getTrainingFeatures();
        Matrix trainingLabels = strategy.getTrainingLabels();
        initializeWeights(trainingFeatures.cols(), trainingLabels.valueCount(0));
        //Get a baseline accuracy
        double validationAccuracy = calculateValidationSetAccuracy(strategy);
        double bestAccuracy = validationAccuracy;
        completeEpoch(0, validationAccuracy);
        boolean keepTraining = true;
        //for each epoch
        while (keepTraining)
        {
            //for each training data instance
            trainingFeatures = strategy.getTrainingFeatures();
            trainingLabels = strategy.getTrainingLabels();
            for (int i = 0; i < trainingFeatures.rows(); ++i)
            {
                analyzeInputRow(trainingFeatures.row(i), trainingLabels.get(i, 0));
                //propagate error through the network
                //adjust the weights
                //calculate the accuracy over training data
            }
            //for each validation data instance
            //calculate the accuracy over the validation data
            validationAccuracy = calculateValidationSetAccuracy(strategy);
            //if the threshold validation accuracy is met, stop training, else continue
            keepTraining = !isThresholdValidationAccuracyMet(validationAccuracy, bestAccuracy);
            if (validationAccuracy > bestAccuracy)
            {
                bestAccuracy = validationAccuracy;
            }
            incrementTotalEpochs();
            completeEpoch(getTotalEpochs(), validationAccuracy);
        }
    }

    // A feature vector goes in. A label vector comes out. (Some supervised
    // learning algorithms only support one-dimensional label vectors. Some
    // support multi-dimensional label vectors.)
    public abstract void predict(double[] features, double[] labels) throws Exception;

    // The model must be trained before you call this method. If the label is nominal,
    // it returns the predictive accuracy. If the label is continuous, it returns
    // the root mean squared error (RMSE). If confusion is non-NULL, and the
    // output label is nominal, then confusion will hold stats for a confusion matrix.
    public double measureAccuracy(Matrix features, Matrix labels, Matrix confusion) throws Exception
    {
        if (features.rows() != labels.rows())
            throw (new Exception("Expected the features and labels to have the same number of rows"));
        if (labels.cols() != 1)
            throw (new Exception("Sorry, this method currently only supports one-dimensional labels"));
        if (features.rows() == 0)
            throw (new Exception("Expected at least one row"));

        int labelValues = labels.valueCount(0);
        if (labelValues == 0) // If the label is continuous...
        {
            // The label is continuous, so measure root mean squared error
            double[] prediction = new double[1];
            double sse = 0.0;
            for (int i = 0; i < features.rows(); i++)
            {
                double[] feat = features.row(i);
                double[] targ = labels.row(i);
                prediction[0] = 0.0; // make sure the prediction is not biased by a previous prediction
                predict(feat, prediction);
                double delta = targ[0] - prediction[0];
                sse += (delta * delta);
            }
            return Math.sqrt(sse / features.rows());
        }
        else
        {
            // The label is nominal, so measure predictive accuracy
            if (confusion != null)
            {
                confusion.setSize(labelValues, labelValues);
                for (int i = 0; i < labelValues; i++)
                    confusion.setAttrName(i, labels.attrValue(0, i));
            }
            int correctCount = 0;
            double[] prediction = new double[1];
            for (int i = 0; i < features.rows(); i++)
            {
                double[] feat = features.row(i);
                int targ = (int) labels.get(i, 0);
                if (targ >= labelValues)
                    throw new Exception("The label is out of range");
                predict(feat, prediction);
                int pred = (int) prediction[0];
                if (confusion != null)
                    confusion.set(targ, pred, confusion.get(targ, pred) + 1);
                if (pred == targ)
                    correctCount++;
            }
            return (double) correctCount / features.rows();
        }
    }

    public int getTotalEpochs()
    {
        return totalEpochs;
    }

    public void incrementTotalEpochs()
    {
        ++this.totalEpochs;
    }

    public MLSystemManager getManager()
    {
        return manager;
    }

    public void setManager(MLSystemManager manager)
    {
        this.manager = manager;
    }

    public double getLearningRate()
    {
        return learningRate;
    }

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    protected void completeEpoch(int epoch, double currentAccuracy)
    {
        manager.completeEpoch(epoch, currentAccuracy);
    }

    protected double calculateValidationSetAccuracy(LearningStrategy strategy) throws Exception
    {
        Matrix validationFeatures = strategy.getValidationFeatures();
        Matrix validationLabels = strategy.getValidationLabels();
        return measureAccuracy(validationFeatures, validationLabels, new Matrix());
    }

    protected abstract void initializeWeights(int features, int outputs);

    protected abstract void analyzeInputRow(double[] row, double expectedOutput);

    protected abstract boolean isThresholdValidationAccuracyMet(double validationAccuracy, double bestAccuracy);
}
