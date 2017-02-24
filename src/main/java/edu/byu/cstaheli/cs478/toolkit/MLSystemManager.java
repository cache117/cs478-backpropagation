package edu.byu.cstaheli.cs478.toolkit;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------


import edu.byu.cstaheli.cs478.backpropogation.BackPropagation;
import edu.byu.cstaheli.cs478.baseline.BaselineLearner;
import edu.byu.cstaheli.cs478.perceptron.Perceptron;
import edu.byu.cstaheli.cs478.toolkit.strategy.*;

import java.util.Random;


public class MLSystemManager
{
    private Random random;
    private SupervisedLearner learner;

    public MLSystemManager()
    {
        setRandom(new Random());
    }

    public static void main(String[] args) throws Exception
    {
        MLSystemManager ml = new MLSystemManager();
        ml.run(args);
    }

    /**
     * When you make a new learning algorithm, you should add a line for it to this method.
     */
    public SupervisedLearner getLearner(String model, Random rand) throws Exception
    {
        switch (model)
        {
            case "baseline":
                return new BaselineLearner();
            case "perceptron":
                return new Perceptron(rand, this);
            case "backpropagation":
                return new BackPropagation(rand, this);
//            case "neuralnet":
//                return new NeuralNet(rand);
//            case "decisiontree":
//                return new DecisionTree();
//            case "knn":
//                return new InstanceBasedLearner();
            default:
                throw new Exception("Unrecognized model: " + model);
        }
    }

    public SupervisedLearner getLearner()
    {
        return learner;
    }

    public void setLearner(SupervisedLearner learner)
    {
        this.learner = learner;
    }

    public void run(String[] args) throws Exception
    {

        //args = new String[]{"-L", "baseline", "-A", "data/iris.arff", "-E", "cross", "10", "-N"};

        //Parse the command line arguments
        ArgParser parser = new ArgParser(args);
        determineEvalMethod(parser);
    }

    public void setRandomSeed(long seed)
    {
        //Random rand = new Random(1234); // Use a seed for deterministic results (makes debugging easier)
        setRandom(new Random(seed));
    }

    private void determineEvalMethod(ArgParser parser) throws Exception
    {
        // Load the model
        if (learner == null)
        {
            learner = getLearner(parser.getLearner(), getRandom());
        }

        // Load the ARFF file
        Matrix arffData = new Matrix();
        arffData.loadArff(parser.getARFF());
        if (parser.isNormalized())
        {
            System.out.println("Using normalized data\n");
            arffData.normalize();
        }

        printStats(parser.getARFF(), parser.getLearner(), parser.getEvaluation(), arffData);

        LearnerData learnerData = new LearnerData(getRandom(), parser, arffData);
        switch (parser.getEvaluation())
        {
            case "training":
                calcTraining(learner, learnerData);
                break;
            case "static":
                calcStatic(learner, learnerData);
                break;
            case "random":
                calcRandom(learner, learnerData);
                break;
            case "cross":
                calcCrossValidation(learner, learnerData);
                break;
        }
    }

    private void calcTraining(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        System.out.println("Calculating accuracy on training set...");
        LearningStrategy strategy = new TrainingStrategy(learnerData);
//        Matrix features = new Matrix(learnerData.getArffData(), 0, 0, learnerData.getArffData().rows(), learnerData.getArffData().cols() - 1);
//        Matrix labels = new Matrix(learnerData.getArffData(), 0, learnerData.getArffData().cols() - 1, learnerData.getArffData().rows(), 1);
        Matrix confusion = new Matrix();
        double startTime = System.currentTimeMillis();
        learner.train(strategy);
        double elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
        double accuracy = learner.measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), confusion);
        System.out.println("Training set accuracy: " + accuracy);
        if (learnerData.isVerbose())
        {
            System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
            confusion.print();
            System.out.println("\n");
        }
        System.out.println("Total number of epochs: " + learner.getTotalEpochs());
    }

    private void calcStatic(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        LearningStrategy strategy = new StaticStrategy(learnerData);

        System.out.println("Calculating accuracy on separate test set...");
        System.out.println("Test set name: " + learnerData.getEvalParameter());
        System.out.println("Number of test instances: " + strategy.getTestingData().rows());
        double startTime = System.currentTimeMillis();
        learner.train(strategy);
        double elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
        double trainAccuracy = learner.measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), null);
        System.out.println("Training set accuracy: " + trainAccuracy);
        Matrix testFeatures = strategy.getTestingFeatures();
        Matrix testLabels = strategy.getTestingLabels();
        Matrix confusion = new Matrix();
        double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
        System.out.println("Test set accuracy: " + testAccuracy);
        if (learnerData.isVerbose())
        {
            System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
            confusion.print();
            System.out.println("\n");
        }
        System.out.println("Total number of epochs: " + learner.getTotalEpochs());
    }

    private void calcRandom(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        LearningStrategy strategy = new RandomStrategy(learnerData);
        System.out.println("Calculating accuracy on a random hold-out set...");
        double trainPercent = Double.parseDouble(learnerData.getEvalParameter());
        if (trainPercent < 0 || trainPercent > 1)
            throw new Exception("Percentage for random evaluation must be between 0 and 1");
        System.out.println("Percentage used for training: " + trainPercent);
        System.out.println("Percentage used for testing: " + (1 - trainPercent));
        learnerData.getArffData().shuffle(learnerData.getRandom());
        Matrix testFeatures = strategy.getTestingFeatures();
        Matrix testLabels = strategy.getTestingLabels();
        double startTime = System.currentTimeMillis();
        learner.train(strategy);
        double elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
        double trainAccuracy = learner.measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), null);
        System.out.println("Training set accuracy: " + trainAccuracy);
        Matrix confusion = new Matrix();
        double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
        System.out.println("Test set accuracy: " + testAccuracy);
        if (learnerData.isVerbose())
        {
            System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
            confusion.print();
            System.out.println("\n");
        }
        learner.writeAccuraciesAndFinalWeights(trainAccuracy, testAccuracy);
        System.out.println("Total number of epochs: " + learner.getTotalEpochs());
    }

    private void calcCrossValidation(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        LearningStrategy strategy;
        System.out.println("Calculating accuracy using cross-validation...");
        int folds = Integer.parseInt(learnerData.getEvalParameter());
        if (folds <= 0)
            throw new Exception("Number of folds must be greater than 0");
        System.out.println("Number of folds: " + folds);
        int reps = 1;
        double sumAccuracy = 0.0;
        double elapsedTime = 0.0;
        for (int j = 0; j < reps; j++)
        {
            learnerData.getArffData().shuffle(learnerData.getRandom());
            for (int i = 0; i < folds; i++)
            {
                int begin = i * learnerData.getArffData().rows() / folds;
                int end = (i + 1) * learnerData.getArffData().rows() / folds;
                strategy = new CrossValidationStrategy(learnerData, begin, end);
                Matrix testFeatures = strategy.getTestingFeatures();
                Matrix testLabels = strategy.getTestingLabels();
                double startTime = System.currentTimeMillis();
                learner.train(strategy);
                elapsedTime += System.currentTimeMillis() - startTime;
                Matrix confusion = new Matrix();
                double accuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
                sumAccuracy += accuracy;
                System.out.println("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
                if (learnerData.isVerbose())
                {
                    System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
                    confusion.print();
                    System.out.println("\n");
                }
            }
        }
        elapsedTime /= (reps * folds);
        System.out.println("Average time to train (in seconds): " + elapsedTime / 1000.0);
        System.out.println("Mean accuracy=" + (sumAccuracy / (reps * folds)));
        System.out.println("Total number of epochs: " + learner.getTotalEpochs());
    }

    private void printStats(String fileName, String learnerName, String evalMethod, Matrix data)
    {
        // Print some stats
        System.out.println();
        System.out.println("Dataset name: " + fileName);
        System.out.println("Number of instances: " + data.rows());
        System.out.println("Number of attributes: " + data.cols());
        System.out.println("Learning algorithm: " + learnerName);
        System.out.println("Evaluation method: " + evalMethod);
        System.out.println();
    }

    public Random getRandom()
    {
        return random;
    }

    private void setRandom(Random random)
    {
        this.random = random;
    }

    private String getArrayString(double[] array)
    {
        StringBuilder builder = new StringBuilder();
        for (double anArray : array)
        {
            builder.append(",").append(anArray);
        }
        return builder.toString();
    }
}
