package edu.byu.cstaheli.cs478.toolkit;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

import java.io.FileWriter;
import java.io.IOException;

public abstract class SupervisedLearner
{
    private static final boolean OUTPUT_EACH_EPOCH = false;

    private int totalEpochs;
    private double learningRate;
    private String outputFile;

    protected SupervisedLearner()
    {
        totalEpochs = 0;
        setLearningRate(.1);
    }

    public void train(LearningStrategy strategy) throws Exception
    {
        Matrix trainingFeatures = strategy.getTrainingFeatures();
        Matrix trainingLabels = strategy.getTrainingLabels();
        initializeWeights(trainingFeatures.cols(), trainingLabels.valueCount(0));
        //Get a baseline accuracy
        double trainingMSE = calcMeanSquaredError(trainingFeatures, trainingLabels);
        double validationMSE = calcMeanSquaredError(strategy.getValidationFeatures(), strategy.getValidationLabels());
        double validationAccuracy = calculateValidationSetAccuracy(strategy);
        double bestAccuracy = validationAccuracy;
        completeEpoch(0, trainingMSE, validationMSE, validationAccuracy);
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
            }
            //calculate the accuracy over training data
            trainingMSE = calcMeanSquaredError(trainingFeatures, trainingLabels);
            //for each validation data instance
            //calculate the accuracy over the validation data
            validationMSE = calcMeanSquaredError(strategy.getValidationFeatures(), strategy.getValidationLabels());
            validationAccuracy = calculateValidationSetAccuracy(strategy);
            //if the threshold validation accuracy is met, stop training, else continue
            keepTraining = !isThresholdValidationAccuracyMet(validationAccuracy, bestAccuracy);
            bestAccuracy = getBestAccuracy(validationAccuracy, bestAccuracy);
            incrementTotalEpochs();
            completeEpoch(getTotalEpochs(), trainingMSE, validationMSE, validationAccuracy);
        }
        double testMSE = calcMeanSquaredError(strategy.getTestingFeatures(), strategy.getTestingLabels());
        double testAccuracy = measureAccuracy(strategy.getTestingFeatures(), strategy.getTestingLabels(), new Matrix());
        outputFinalAccuracies(getTotalEpochs(), trainingMSE, validationMSE, testMSE, validationAccuracy, testAccuracy);
    }

    protected double getBestAccuracy(double newValue, double previousBest)
    {
        if (newValue > previousBest)
        {
            previousBest = newValue;
        }
        return previousBest;
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

    public double getLearningRate()
    {
        return learningRate;
    }

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    protected void completeEpoch(int epoch, double classificationAccuracy)
    {
        if (outputFile != null && OUTPUT_EACH_EPOCH)
        {
            try (FileWriter writer = new FileWriter(outputFile, true))
            {
                writer.append(String.format("%s, %s\n", epoch, classificationAccuracy));
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    protected void completeEpoch(int epoch, double trainingMSE, double validationMSE, double classificationAccuracy)
    {
        if (outputFile != null && OUTPUT_EACH_EPOCH)
        {
            try (FileWriter writer = new FileWriter(outputFile, true))
            {
                writer.append(String.format("%s, %s, %s, %s\n", epoch, trainingMSE, validationMSE, classificationAccuracy));
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    protected void outputFinalAccuracies(int epoch, double trainingMSE, double validationMSE, double testMSE, double validationClassificationAccuracy, double testClassificationAccuracy)
    {
        if (outputFile != null)
        {
            try (FileWriter writer = new FileWriter(outputFile, true))
            {
                writer.append(String.format("%s, %s, %s, %s, %s, %s\n", epoch, trainingMSE, validationMSE, testMSE, validationClassificationAccuracy, testClassificationAccuracy));
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
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

    public abstract void writeAccuraciesAndFinalWeights(double trainAccuracy, double testAccuracy);

    private double calcMeanSquaredError(Matrix features, Matrix labels) throws Exception
    {
        assert features.rows() == labels.rows();
        double mse = 0;
        for (int i = 0; i < features.rows(); ++i)
        {
            double[] row = features.row(i);
            double output = labels.get(i, 0);
            double[] label = new double[1];
            predict(row, label);
            double predicted = label[0];
            mse += calcMeanSquaredError(output, predicted);
        }
        return mse / features.rows();
    }

    private double calcMeanSquaredError(double expected, double actual)
    {
        return (expected * actual) * (expected * actual);
    }

    public void setOutputFile(String outputFile)
    {
        this.outputFile = outputFile;
    }
}
