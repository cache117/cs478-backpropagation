package edu.byu.cstaheli.cs478.backpropogation;

import java.util.List;

/**
 * Created by cstaheli on 2/16/2017.
 */
public class Node
{
    private double weight;
    private double biasWeight;
    protected static final Node ZERO_NODE = new Node(0, 0);

    public Node(double weight, double biasWeight)
    {
        this.weight = weight;
        this.biasWeight = biasWeight;
    }

    public double calcNet(List<Double> inputs)
    {
        double net = 0;
        for (double input: inputs)
        {
            net += weight * input;
        }
        net += weight * biasWeight;
        return net;
    }

    public double calcOutput(double net)
    {
        return (1d / (1 + Math.exp(-net)));
    }

    public double calcGradient(double output)
    {
        return output * (1 - output);
    }

    public double calcWeightChange(double learningRate, double nodeInput, double nodeError)
    {
        return learningRate * nodeInput * nodeError;
    }

    public double calcOutputNodeError(double expectedOutput, double actualOutput, double gradient)
    {
        return (expectedOutput - actualOutput) * gradient;
    }

    public double calcHiddenNodeError(double gradient, List<Double> outputErrors, List<Double> weightsToOutputs)
    {
        return gradient * calcSumOfOutputErrors(outputErrors, weightsToOutputs);
    }

    private double calcSumOfOutputErrors(List<Double> outputErrors, List<Double> weightsToOutputs)
    {
        assert outputErrors.size() == weightsToOutputs.size();
        double sum = 0;
        for (int i = 0; i < outputErrors.size(); ++i)
        {
            sum += outputErrors.get(i) * weightsToOutputs.get(i);
        }
        return sum;
    }
}
