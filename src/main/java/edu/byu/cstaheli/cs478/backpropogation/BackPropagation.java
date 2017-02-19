package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.RandomLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
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
        super.train(strategy);
    }

    protected void initializeWeights(int features, int outputs)
    {
        int numberOfNodesInHiddenLayer = getNumberOfNodesInHiddenLayer(features, outputs);
        hiddenLayer = new ArrayList<>(numberOfNodesInHiddenLayer);
        outputLayer = new ArrayList<>(outputs);
        int id = 0;
        for (int i = 0; i < numberOfNodesInHiddenLayer; ++i, ++id)
        {
            hiddenLayer.add(new Node(features, getRandom()));
        }
        for (int i = 0; i < outputs; ++i, ++id)
        {
            outputLayer.add(new Node(numberOfNodesInHiddenLayer, getRandom()));
        }
    }

    protected void analyzeInputRow(double[] row, double expectedOutput)
    {
        List<Double> hiddenLayerOutputs = getLayerOutputs(convertInputRow(row), hiddenLayer);
        List<Double> outputLayerErrors = new ArrayList<>(outputLayer.size());
        for (Node outputNode : outputLayer)
        {
            double nodeOutput = calcNodeOutput(outputNode, hiddenLayerOutputs);
            double gradient = outputNode.calcGradient(nodeOutput);
            double error = outputNode.calcOutputNodeError(gradient, expectedOutput, nodeOutput);
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
        List<Double> outputLayerWeights = new ArrayList<>(outputLayer.size());
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

    protected boolean isThresholdValidationAccuracyMet(double previousAccuracy, double validationAccuracy)
    {
        return validationAccuracy <= previousAccuracy;
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
