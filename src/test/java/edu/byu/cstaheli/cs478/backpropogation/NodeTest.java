package edu.byu.cstaheli.cs478.backpropogation;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

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
        Node node = new Node(1, 1);
        double net = node.calcNet(inputs);
        assertEquals(1, net);

        inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(1d);
        net = node.calcNet(inputs);
        assertEquals(2, net);

        inputs = new ArrayList<>();
        inputs.add(1d);
        inputs.add(1d);
        net = node.calcNet(inputs);
        assertEquals(3, net);

        inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(1d);
        node = new Node(.5, 1);
        net = node.calcNet(inputs);
        assertEquals(1, net);

        inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(0d);
        node = new Node(.5, .5);
        net = node.calcNet(inputs);
        assertEquals(.25, net);
    }
}