package edu.byu.cstaheli.cs478.toolkit;

import java.util.Random;

/**
 * Created by cstaheli on 2/18/2017.
 */
public class RandomWeightGenerator
{
    private Random random;

    public RandomWeightGenerator(Random random)
    {
        this.random = random;
    }

    public RandomWeightGenerator(long seed)
    {
        random = new Random(seed);
    }

    public double getRandomWeight()
    {
        return random.nextDouble() - 0.5;
    }

    public int getRandomInt(int bound)
    {
        return random.nextInt(bound);
    }
}
