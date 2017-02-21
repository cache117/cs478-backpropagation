package edu.byu.cstaheli.cs478.backpropogation;

/**
 * Created by cstaheli on 2/18/2017.
 */
public class InputWeight
{
    private double input;
    private double weight;

    public InputWeight(double weight)
    {
        this.weight = weight;
    }

    public InputWeight(double input, double weight)
    {
        this(weight);
        this.input = input;
    }

    public double getInput()
    {
        return input;
    }

    public void setInput(double input)
    {
        this.input = input;
    }

    public double getWeight()
    {
        return weight;
    }

    public double calcNet()
    {
        return input * weight;
    }

    public void changeWeight(double learningRate, double outputError)
    {
        double weightDelta = Node.calculateWeightDelta(learningRate, outputError, input);
        applyWeightChange(weightDelta);
    }

    private void applyWeightChange(double deltaWeight)
    {
        this.weight += deltaWeight;
    }
}
