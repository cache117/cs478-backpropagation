package edu.byu.cstaheli.cs478.toolkit;

import java.util.Random;

/**
 * Created by cstaheli on 2/16/2017.
 */
public abstract class RandomLearner extends SupervisedLearner
{
    private RandomWeightGenerator random;

    public RandomLearner(Random random, MLSystemManager manager)
    {
        super(manager);
        this.setRandom(random);
    }

    protected double getRandomWeight()
    {
        // Gives numbers between -.5 and .5
        return random.getRandomWeight();
    }

    protected RandomWeightGenerator getRandom()
    {
        return random;
    }

    protected void setRandom(Random random)
    {
        this.random = new RandomWeightGenerator(random);
    }
}
