package edu.byu.cstaheli.cs478.backpropogation;

import edu.byu.cstaheli.cs478.toolkit.RandomWeightGenerator;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by cstaheli on 2/16/2017.
 */
class NodeTest
{
    private static void assertNumberBetween(double number, double lowerBound, double upperBound)
    {
        assertTrue(number >= lowerBound && number <= upperBound);
    }

    @Test
    void testCalcOutput()
    {
        Node node = Node.ZERO_NODE;
        double output = node.calcOutput(0);
        assertEquals(.5, output);

        output = node.calcOutput(2.5);
        assertNumberBetween(output, .923, .925);

        output = node.calcOutput(-6);
        assertNumberBetween(output, .00246, .00248);

        output = node.calcOutput(5);
        assertNumberBetween(output, .9932, .9934);

        output = node.calcOutput(1);
        assertNumberBetween(output, .73, .7311);
    }

    @Test
    void testCalcGradient()
    {
        Node node = Node.ZERO_NODE;
        double gradient = node.calcGradient(.921);
        assertNumberBetween(gradient, .0727589, .072759);

        gradient = node.calcGradient(.941);
        assertNumberBetween(gradient, .055518, .05552);
    }

    @Test
    void testCalcWeightChange()
    {

    }

    @Test
    void testCalcOutputNodeError()
    {

    }

    @Test
    void testCalcHiddenNodeError()
    {

    }

    @Test
    void testCalcNet()
    {
        List<Double> inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(0d);

        Node node = new Node(0, inputs.size(), new RandomWeightGenerator(1234));
        double net = node.calcNet(inputs);
        assertNumberBetween(net, .356, .358);

        inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(1d);
        net = node.calcNet(inputs);
        assertNumberBetween(net, .807, .809);

        inputs = new ArrayList<>();
        inputs.add(1d);
        inputs.add(1d);
        net = node.calcNet(inputs);
        assertNumberBetween(net, .955, .956);

        inputs = new ArrayList<>();
        inputs.add(1d);
        inputs.add(0d);
        node = new Node(0, inputs.size(), new RandomWeightGenerator(1234));
        net = node.calcNet(inputs);
        assertNumberBetween(net, .504, .505);

        inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(1d);
        inputs.add(1d);
        inputs.add(1d);
        inputs.add(0d);
        node = new Node(0, inputs.size(), new RandomWeightGenerator(1234));
        net = node.calcNet(inputs);
        assertNumberBetween(net, .471, .472);
    }
}