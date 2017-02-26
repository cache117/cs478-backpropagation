package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import org.junit.jupiter.api.Test;

import java.io.File;
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
//        System.out.println("Training");
//        args = ("-L backpropagation -A " + datasetsLocation + "vowel.arff -E training -V").split(" ");
//        manager.run(args);
//        System.out.println("Training");
//        args = ("-L backpropagation -A " + datasetsLocation + "voting.arff -E training -V").split(" ");
//        manager.run(args);
//        System.out.println("Cross Fold Validation");
//        args = ("-L backpropagation -A " + datasetsLocation + "voting.arff -E cross 25").split(" ");
//        manager.run(args);
        System.out.println("");
        BackPropagation backPropagation = new BackPropagation(new Random(1234));
//        assertTrue((new File(datasetsLocation + "irisResults.csv").delete()));
//        backPropagation.setOutputFile(datasetsLocation + "irisResults.csv");
//        manager.setLearner(backPropagation);
//        args = ("-L backpropagation -A " + datasetsLocation + "iris.arff -E random .75").split(" ");
//        manager.run(args);
//        backPropagation = new BackPropagation(new Random(1234));
//        assertTrue((new File(datasetsLocation + "vowelResults.csv").delete()));
//        backPropagation.setOutputFile(datasetsLocation + "vowelResults.csv");
//        manager.setLearner(backPropagation);
//        args = ("-L backpropagation -A " + datasetsLocation + "vowel.arff -E random .75").split(" ");
//        manager.run(args);
        assertTrue((new File(datasetsLocation + "modifiedVowelResults.csv").delete()));
        for (int i = 0; i < 3; ++i)
        {
            backPropagation = new BackPropagation(new Random(1234));
            backPropagation.setOutputFile(datasetsLocation + "modifiedVowelResults.csv");
            manager.setLearner(backPropagation);
            args = ("-L backpropagation -A " + datasetsLocation + "modifiedVowel.arff -E random .75").split(" ");
            manager.run(args);
        }
        assertTrue((new File(datasetsLocation + "modifiedVowel1.csv").delete()));
        for (int i = 0; i < 3; ++i)
        {
            backPropagation = new BackPropagation(new Random(1234));
            backPropagation.setOutputFile(datasetsLocation + "modifiedVowel1.csv");
            backPropagation.setLearningRate(.5);
            manager.setLearner(backPropagation);
            args = ("-L backpropagation -A " + datasetsLocation + "modifiedVowel.arff -E random .75").split(" ");
            manager.run(args);
        }
        assertTrue((new File(datasetsLocation + "modifiedVowel2.csv").delete()));
        for (int i = 0; i < 3; ++i)
        {
            backPropagation = new BackPropagation(new Random(1234));
            backPropagation.setOutputFile(datasetsLocation + "modifiedVowel2.csv");
            backPropagation.setLearningRate(.001);
            manager.setLearner(backPropagation);
            args = ("-L backpropagation -A " + datasetsLocation + "modifiedVowel.arff -E random .75").split(" ");
            manager.run(args);
        }
        assertTrue((new File(datasetsLocation + "modifiedVowel3.csv").delete()));
        for (int i = 0; i < 3; ++i)
        {
            backPropagation = new BackPropagation(new Random(1234));
            backPropagation.setOutputFile(datasetsLocation + "modifiedVowel3.csv");
            backPropagation.setLearningRate(1);
            manager.setLearner(backPropagation);
            args = ("-L backpropagation -A " + datasetsLocation + "modifiedVowel.arff -E random .75").split(" ");
            manager.run(args);
        }
        assertTrue((new File(datasetsLocation + "modifiedVowel4.csv").delete()));
        for (int i = 0; i < 3; ++i)
        {
            backPropagation = new BackPropagation(new Random(1234));
            backPropagation.setOutputFile(datasetsLocation + "modifiedVowel4.csv");
            backPropagation.setLearningRate(5);
            manager.setLearner(backPropagation);
            args = ("-L backpropagation -A " + datasetsLocation + "modifiedVowel.arff -E random .75").split(" ");
            manager.run(args);
        }
    }

    @Test
    public void testAnalyzeInputRow() throws Exception
    {
        List<Node> hiddenNodes = getHiddenNodes(getInputWeights());
        List<Node> outputNodes = getOutputNodes(getInputWeights());

        BackPropagation backPropagation = buildTestBackProp(hiddenNodes, outputNodes);

        double[] row = new double[2];
        row[0] = 0;
        row[1] = 0;
        double output = 1;
        backPropagation.analyzeInputRow(row, output);
        hiddenNodes = backPropagation.getHiddenLayer();
        Node node = hiddenNodes.get(0);
        double weight = node.getInputWeight(0);
        assertEquals(1, weight);
        outputNodes = backPropagation.getOutputLayer();
        node = outputNodes.get(0);
        weight = node.getInputWeight(1);
        assertNumberBetween(weight, 1.00415, 1.00416);
        weight = node.getBiasWeight();
        assertNumberBetween(weight, 1.00568, 1.00569);

        hiddenNodes = getHiddenNodes(getInputWeights());
        outputNodes = getOutputNodes(getInputWeights());

        backPropagation = buildTestBackProp(hiddenNodes, outputNodes);

        row = new double[2];
        row[0] = 0;
        row[1] = 1;
        output = 0;
        backPropagation.analyzeInputRow(row, output);
//        hiddenNodes = backPropagation.getHiddenLayer();
//        node = hiddenNodes.get(0);
//        weight = node.getInputWeight(0);
//        assertEquals(1, weight);
//        outputNodes = backPropagation.getOutputLayer();
//        node = outputNodes.get(0);
//        weight = node.getInputWeight(1);
//        assertNumberBetween(weight, 1.00415, 1.00416);
//        weight = node.getBiasWeight();
//        assertNumberBetween(weight, 1.00568, 1.00569);

        //Real test
        hiddenNodes = getHiddenNodes(getInputWeights());
        outputNodes = getOutputNodes(getInputWeights());

        backPropagation = buildTestBackProp(hiddenNodes, outputNodes);

        row = new double[2];
        row[0] = 0;
        row[1] = 0;
        output = 1;
        backPropagation.analyzeInputRow(row, output);
        hiddenNodes = backPropagation.getHiddenLayer();
        node = hiddenNodes.get(0);
        weight = node.getInputWeight(0);
        assertEquals(1, weight);
        outputNodes = backPropagation.getOutputLayer();
        node = outputNodes.get(0);
        weight = node.getInputWeight(1);
        assertNumberBetween(weight, 1.00415, 1.00416);
        weight = node.getBiasWeight();
        assertNumberBetween(weight, 1.00568, 1.00569);

        row = new double[2];
        row[0] = 0;
        row[1] = 1;
        output = 0;
        backPropagation.analyzeInputRow(row, output);
        hiddenNodes = backPropagation.getHiddenLayer();
        node = hiddenNodes.get(0);
        weight = node.getInputWeight(0);
        assertEquals(1, weight);
        outputNodes = backPropagation.getOutputLayer();
        node = outputNodes.get(0);
        weight = node.getInputWeight(1);
        assertNumberBetween(weight, 1.00415, 1.00416);
        weight = node.getBiasWeight();
        assertNumberBetween(weight, 1.00568, 1.00569);

    }

    private BackPropagation buildTestBackProp(List<Node> hiddenNodes, List<Node> outputNodes)
    {
        BackPropagation backPropagation;
        backPropagation = new BackPropagation(new Random(1234));
        backPropagation.setHiddenLayer(hiddenNodes);
        backPropagation.setOutputLayer(outputNodes);
        backPropagation.setLearningRate(1);
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