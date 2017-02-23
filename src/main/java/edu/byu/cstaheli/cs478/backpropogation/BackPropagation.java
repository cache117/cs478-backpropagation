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
    private static final int EPOCHS_WITHOUT_SIGNIFICANT_IMPROVEMENT = 5;

    private List<Node> hiddenLayer;
    private List<Node> outputLayer;
    private int epochsWithoutSignificantImprovement;
    private double momentum;

    public BackPropagation(Random rand, MLSystemManager manager)
    {
        super(rand, manager);
        momentum = 0;
    }

    public double getMomentum()
    {
        return momentum;
    }

    public void setMomentum(double momentum)
    {
        this.momentum = momentum;
    }

    public List<Node> getHiddenLayer()
    {
        return hiddenLayer;
    }

    public void setHiddenLayer(List<Node> hiddenLayer)
    {
        this.hiddenLayer = hiddenLayer;
    }

    public List<Node> getOutputLayer()
    {
        return outputLayer;
    }

    public void setOutputLayer(List<Node> outputLayer)
    {
        this.outputLayer = outputLayer;
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
        for (int i = 0; i < numberOfNodesInHiddenLayer; ++i)
        {
            hiddenLayer.add(new Node(features, getRandom()));
        }

        outputLayer = new ArrayList<>(outputs);
        for (int i = 0; i < outputs; ++i)
        {
            outputLayer.add(new Node(numberOfNodesInHiddenLayer, getRandom()));
        }
    }

    protected void analyzeInputRow(double[] row, double expectedOutput)
    {
        List<Double> hiddenLayerOutputs = getLayerOutputs(convertInputRow(row), hiddenLayer);
        List<Double> outputLayerErrors = getOutputLayerErrors(expectedOutput, hiddenLayerOutputs);
        updateHiddenLayerWeights(hiddenLayerOutputs, outputLayerErrors);
        updateOutputLayerWeights(outputLayerErrors);
    }

    private void updateOutputLayerWeights(List<Double> outputLayerErrors)
    {
        assert outputLayerErrors.size() == outputLayer.size();
        for (int i = 0; i < outputLayer.size(); ++i)
        {
            Node outputNode = outputLayer.get(i);
            double error = outputLayerErrors.get(i);
            outputNode.calcWeightChanges(getLearningRate(), error, momentum);
        }
    }

    private void updateHiddenLayerWeights(List<Double> hiddenLayerOutputs, List<Double> outputLayerErrors)
    {
        assert hiddenLayer.size() == hiddenLayerOutputs.size() && outputLayerErrors.size() == outputLayer.size();
        for (int i = 0; i < hiddenLayer.size(); ++i)
        {
            List<Double> outputLayerWeights = getParentWeights(i, outputLayer);
            Node hiddenNode = hiddenLayer.get(i);
            double gradient = hiddenNode.calcGradient(hiddenLayerOutputs.get(i));
            double error = hiddenNode.calcHiddenNodeError(gradient, outputLayerErrors, outputLayerWeights);
            hiddenNode.calcWeightChanges(getLearningRate(), error, momentum);
        }
    }

    private List<Double> getOutputLayerErrors(double expectedOutput, List<Double> hiddenLayerOutputs)
    {
        List<Double> outputLayerErrors = new ArrayList<>(outputLayer.size());
        for (Node outputNode : outputLayer)
        {
            double nodeOutput = calcNodeOutput(outputNode, hiddenLayerOutputs);
            double gradient = outputNode.calcGradient(nodeOutput);
            double error = outputNode.calcOutputNodeError(gradient, expectedOutput, nodeOutput);
            outputLayerErrors.add(error);
//            outputNode.calcWeightChanges(getLearningRate(), error);
        }
        return outputLayerErrors;
    }

    private List<Double> getLayerOutputs(List<Double> inputs, List<Node> layer)
    {
        List<Double> layerOutputs = new ArrayList<>(layer.size());
        for (Node node : layer)
        {
            double nodeOutput = calcNodeOutput(node, inputs);
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

    private List<Double> getParentWeights(int index, List<Node> parentLayer)
    {
        List<Double> outputLayerWeights = new ArrayList<>(parentLayer.size());
        for (Node outputNode : parentLayer)
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
        if (validationAccuracy <= previousAccuracy)
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

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        List<Double> hiddenLayerOutputs = getLayerOutputs(convertInputRow(features), hiddenLayer);
        List<Double> outputLayerOutputs = getLayerOutputs(hiddenLayerOutputs, outputLayer);
        labels[0] = determineFinalOutput(outputLayerOutputs);
    }

    private double determineFinalOutput(List<Double> outputs)
    {
        //return the index of the node that activated
        int index = 0;
        double maxOutput = outputs.get(0);
        for (int i = 1; i < outputs.size(); ++i)
        {
            double output = outputs.get(i);
            if (output > maxOutput)
            {
                maxOutput = output;
                index = i;
            }
        }
        return index;
    }
}
