package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.Matrix;
import edu.byu.cstaheli.cs478.toolkit.RandomLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by cstaheli on 1/31/2017.
 */
public class BackPropagation extends RandomLearner
{
    private List<Node> hiddenLayer;
    private List<Node> outputLayer;

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
                analyzeInputRow(trainingFeatures.row(i), trainingLabels.get(0, 0));
                //propagate error through the network
                //adjust the weights
                //calculate the accuracy over training data
            }
            //for each validation data instance
            Matrix validationFeatures = strategy.getValidationFeatures();
            Matrix validationLabels = strategy.getValidationLabels();
            //calculate the accuracy over the validation data
            double validationAccuracy = measureAccuracy(validationFeatures, validationLabels, null);
            //if the threshold validation accuracy is met, stop training, else continue
            keepTraining = !isThresholdValidationAccuracyMet(validationAccuracy);
        }
    }

    private void initializeWeights(int features, int outputs)
    {
        int numberOfNodesInHiddenLayer = getNumberOfNodesInHiddenLayer(features, outputs);
        hiddenLayer = new ArrayList<>(numberOfNodesInHiddenLayer);
        outputLayer = new ArrayList<>(outputs);
        int id = 0;
        for (int i = 0; i < numberOfNodesInHiddenLayer; ++i, ++id)
        {
            hiddenLayer.add(new Node(id, features, getRandom()));
        }
        for (int i = 0; i < outputs; ++i, ++id)
        {
            outputLayer.add(new Node(id, numberOfNodesInHiddenLayer, getRandom()));
        }
    }

    private void analyzeInputRow(double[] row, double expectedOutput)
    {
        List<Double> hiddenLayerOutputs = getLayerOutputs(convertInputRow(row), hiddenLayer);
        List<Double> outputLayerErrors = new ArrayList<>(outputLayer.size());
        for (Node outputNode : outputLayer)
        {
            double nodeOutput = calcNodeOutput(outputNode, hiddenLayerOutputs);
            double gradient = outputNode.calcGradient(nodeOutput);
            double error = outputNode.calcOutputNodeError(expectedOutput, nodeOutput, gradient);
            outputLayerErrors.add(error);
            outputNode.calcWeightChanges(getLearningRate(), error);
        }
        assert hiddenLayer.size() == hiddenLayerOutputs.size() && outputLayerErrors.size() == outputLayer.size();
        for (int i = 0; i < hiddenLayer.size(); ++i)
        {
            List<Double> outputLayerWeights = getParentWeights(i);
            Node hiddenNode = hiddenLayer.get(i);
            double gradient = hiddenNode.calcGradient(hiddenLayerOutputs.get(i));
            double error = hiddenNode.calcHiddenNodeError(gradient, outputLayerErrors, outputLayerWeights);
            hiddenNode.calcWeightChanges(getLearningRate(), error);
        }
    }

    private List<Double> getLayerOutputs(List<Double> inputs, List<Node> layer)
    {
        List<Double> layerOutputs = new ArrayList<>(layer.size());
        for (Node hiddenNode : layer)
        {
            double nodeOutput = calcNodeOutput(hiddenNode, inputs);
            layerOutputs.add(nodeOutput);
        }
        return layerOutputs;
    }

    private List<Double> convertInputRow(double[] row)
    {
        return Arrays.stream(row).boxed().collect(Collectors.toList());
    }

    private double calcNodeOutput(Node node, List<Double> inputs)
    {
        double net = node.calcNet(inputs);
        return node.calcOutput(net);
    }

    private List<Double> getParentWeights(int index)
    {
        List<Double> outputLayerWeights =  new ArrayList<>(outputLayer.size());
        for (Node outputNode : outputLayer)
        {
            outputLayerWeights.add(outputNode.getInputWeight(index));
        }
        return outputLayerWeights;
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
        List<Double> hiddenLayerOutputs = getLayerOutputs(convertInputRow(features), hiddenLayer);
        List<Double> outputLayerOutputs = getLayerOutputs(hiddenLayerOutputs, outputLayer);
        labels[0] = determineFinalOutput(outputLayerOutputs);
    }

    private double determineFinalOutput(List<Double> outputs)
    {
        return outputs.get(0);
    }


}
