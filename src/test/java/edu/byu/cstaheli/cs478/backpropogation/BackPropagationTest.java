package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by cstaheli on 2/12/2017.
 */
class BackPropagationTest
{
    private static void assertNumberBetween(double number, double lowerBound, double upperBound)
    {
        assertTrue(number >= lowerBound && number <= upperBound, String.format("Actual: %s. Expected Bounds[%s, %s].", number, lowerBound, upperBound));
    }

    @Test
    public void testBackPropagation() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        String datasetsLocation = "src/test/resources/datasets/";
        System.out.println("Training");
        args = ("-L backpropagation -A " + datasetsLocation + "vowel.arff -E training").split(" ");
        manager.run(args);
        System.out.println("Training");
        args = ("-L backpropagation -A " + datasetsLocation + "voting.arff -E training").split(" ");
        manager.run(args);
        System.out.println("Cross Fold Validation");
        args = ("-L backpropagation -A " + datasetsLocation + "voting.arff -E cross 25 -V").split(" ");
        manager.run(args);
    }

    @Test
    public void testAnalyzeInputRow() throws Exception
    {
        List<InputWeight> inputWeights = getInputWeights();
        List<Node> hiddenNodes = getHiddenNodes(inputWeights);
        List<Node> outputNodes = getOutputNodes(inputWeights);

        BackPropagation backPropagation = buildTestBackProp(hiddenNodes, outputNodes);

        double[] row = new double[2];
        row[0] = 0;
        row[1] = 0;
        double output = 1;
        backPropagation.analyzeInputRow(row, output);
        hiddenNodes = backPropagation.getHiddenLayer();
        Node node = hiddenNodes.get(0);
        double weight = node.getInputWeight(0);
        assertNumberBetween(weight, -.001, .001);
        outputNodes = backPropagation.getOutputLayer();
        node = outputNodes.get(0);
        weight = node.getInputWeight(0);
        assertEquals(0, weight);

        inputWeights = getInputWeights();
        hiddenNodes = getHiddenNodes(inputWeights);
        outputNodes = getOutputNodes(inputWeights);

        backPropagation = buildTestBackProp(hiddenNodes, outputNodes);

        row = new double[2];
        row[0] = 0;
        row[1] = 1;
        output = 0;
        backPropagation.analyzeInputRow(row, output);
        hiddenNodes = backPropagation.getHiddenLayer();
        outputNodes = backPropagation.getOutputLayer();

        //Real test
        inputWeights = getInputWeights();
        hiddenNodes = getHiddenNodes(inputWeights);
        outputNodes = getOutputNodes(inputWeights);

        backPropagation = buildTestBackProp(hiddenNodes, outputNodes);

        row = new double[2];
        row[0] = 0;
        row[1] = 0;
        output = 1;
        backPropagation.analyzeInputRow(row, output);
        hiddenNodes = backPropagation.getHiddenLayer();
        outputNodes = backPropagation.getOutputLayer();

        row = new double[2];
        row[0] = 0;
        row[1] = 1;
        output = 0;
        backPropagation.analyzeInputRow(row, output);
        hiddenNodes = backPropagation.getHiddenLayer();
        outputNodes = backPropagation.getOutputLayer();

    }

    private BackPropagation buildTestBackProp(List<Node> hiddenNodes, List<Node> outputNodes)
    {
        BackPropagation backPropagation;
        backPropagation = new BackPropagation(new Random(1234), new MLSystemManager());
        backPropagation.setHiddenLayer(hiddenNodes);
        backPropagation.setOutputLayer(outputNodes);
        return backPropagation;
    }

    private List<Node> getOutputNodes(List<InputWeight> inputWeights)
    {
        List<Node> outputNodes = new ArrayList<>(1);
        outputNodes.add(new Node(inputWeights, 1));
        return outputNodes;
    }

    private List<Node> getHiddenNodes(List<InputWeight> inputWeights)
    {
        List<Node> hiddenNodes = new ArrayList<>(2);
        for (int i = 0; i < 2; ++i)
        {
            Node node = new Node(inputWeights, 1);
            hiddenNodes.add(node);
        }
        return hiddenNodes;
    }

    private List<InputWeight> getInputWeights()
    {
        List<InputWeight> inputWeights = new ArrayList<>(3);
        for (int i = 0; i < 2; ++i)
        {
            inputWeights.add(new InputWeight(1));
        }
        return inputWeights;
    }
}