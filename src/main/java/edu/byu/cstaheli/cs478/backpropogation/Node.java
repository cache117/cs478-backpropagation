package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.RandomWeightGenerator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by cstaheli on 2/16/2017.
 */
public class Node
{
    /**
     * Used for testing.
     * <p>
     * This node is initialized with zeros for weights;
     */
    protected static final Node ZERO_NODE = new Node(0, new RandomWeightGenerator(new Random(1234)));
    private double biasWeight;
    private List<InputWeight> inputWeights;
    private RandomWeightGenerator random;

    public Node(int numberOfInputs, RandomWeightGenerator random)
    {
        this.random = random;
        generateWeights(numberOfInputs);
        this.biasWeight = getRandomWeight();
    }

    private double getRandomWeight()
    {
        return random.getRandomWeight();
    }

    private void generateWeights(int numberOfInputs)
    {
        inputWeights = new ArrayList<>(numberOfInputs);
        for (int i = 0; i < numberOfInputs; ++i)
        {
            inputWeights.add(new InputWeight(getRandomWeight()));
        }
    }

    public double getBiasWeight()
    {
        return biasWeight;
    }

    public double calcNet(List<Double> inputs)
    {
        addInputs(inputs);
        double net = 0;
        for (InputWeight inputWeight : inputWeights)
        {
            net += inputWeight.calcNet();
        }
        net += biasWeight * 1;
        return net;
    }

    private void addInputs(List<Double> inputs)
    {
        assert inputs.size() == inputWeights.size();
        for (int i = 0; i < inputs.size(); ++i)
        {
            inputWeights.get(i).setInput(inputs.get(i));
        }
    }

    public double calcOutput(double net)
    {
        return (1d / (1 + Math.exp(-net)));
    }

    public double calcGradient(double output)
    {
        return output * (1 - output);
    }

    public void calcWeightChanges(double learningRate, double outputError)
    {
        for (InputWeight inputWeight : inputWeights)
        {
            inputWeight.changeWeight(learningRate, outputError);
        }
    }

    public double calcOutputNodeError(double gradient, double expectedOutput, double actualOutput)
    {
        return gradient * (expectedOutput - actualOutput);
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


    public double getInputWeight(int index)
    {
        return inputWeights.get(index).getWeight();
    }
}
