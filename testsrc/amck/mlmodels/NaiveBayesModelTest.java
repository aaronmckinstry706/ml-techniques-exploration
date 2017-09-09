package amck.mlmodels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class NaiveBayesModelTest {
    
    private static final Logger LOGGER = Logger.getLogger(NaiveBayesModelTest.class.getName());
    
    private NaiveBayesModel model;
    
    @Before
    public void setup() {
        model = new NaiveBayesModel.Builder().inputDimension(2).numberOfClasses(2).build();
    }
    
    @Test
    public void testUntrainedModel() {
        Double uniformLogProbability = new Double(Math.log(1.0/model.getNumberOfClasses())
                + model.getInputDimension()*Math.log(0.5));
        ArrayList<Double> expected = new ArrayList<Double>(model.getNumberOfClasses());
        expected.add(uniformLogProbability);
        expected.add(uniformLogProbability);
        boolean[] input = new boolean[model.getInputDimension()];
        for (int featureIndex = 0; featureIndex < input.length; ++featureIndex) {
            input[featureIndex] = false;
        }
        Assert.assertEquals(expected, model.predict(input));
        input[model.getNumberOfClasses()/2] = true;
        Assert.assertEquals(expected, model.predict(input));
    }
    
    @Test
    public void testTrainedModel() {
        boolean[][] samples = new boolean[][] {
            // Class 0 examples:
            new boolean[] {true, true},
            new boolean[] {true, true},
            new boolean[] {true, true},
            new boolean[] {true, false},
            new boolean[] {true, false},
            new boolean[] {true, false},
            new boolean[] {true, false},
            new boolean[] {true, false},
            new boolean[] {false, false},
            new boolean[] {false, false},
            new boolean[] {false, false},
            new boolean[] {false, false},
            /**
             * Correct conditional probabilities for C==0 are:
             *  P(x0==true  | C==0) = 2/3
             *  P(x0==false | C==0) = 1/3
             *  P(x1==true  | C==0) = 1/4
             *  P(x1==false | C==0) = 3/4
             */
            
            // Class 1 examples:
            new boolean[] {true, true},
            new boolean[] {true, true},
            new boolean[] {false, true},
            new boolean[] {false, true},
            new boolean[] {false, false}
            /**
             * Correct conditional probabilities for C==1 are:
             *  P(x0==true  | C==1) = 2/5
             *  P(x0==false | C==1) = 3/5
             *  P(x1==true  | C==1) = 4/5
             *  P(x1==false | C==1) = 1/5
             */
        };
        
        /**
         * Correct priors are:
         *  P(C==0) = 5/17
         *  P(C==1) = 12/17
         */
        
        int[] labels = new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        
        model.train(samples, labels);
        
        boolean[][] testInputs = new boolean[][] {
            new boolean[] {true, true},
            new boolean[] {true, false},
            new boolean[] {false, true},
            new boolean[] {false, false}
        };
        
        List<Double> logPriors = Arrays.asList(Math.log(12.0/17.0), Math.log(5.0/17.0));
        
        List<List<Double>> expectedPredictions = new ArrayList<List<Double>>();
        expectedPredictions.add(Arrays.asList(
                Math.log(2.0/3.0) + Math.log(1.0/4.0) + logPriors.get(0),
                Math.log(2.0/5.0) + Math.log(4.0/5.0) + logPriors.get(1)));
        expectedPredictions.add(Arrays.asList(
                Math.log(2.0/3.0) + Math.log(3.0/4.0) + logPriors.get(0),
                Math.log(2.0/5.0) + Math.log(1.0/5.0) + logPriors.get(1)));
        expectedPredictions.add(Arrays.asList(
                Math.log(1.0/3.0) + Math.log(1.0/4.0) + logPriors.get(0),
                Math.log(3.0/5.0) + Math.log(4.0/5.0) + logPriors.get(1)));
        expectedPredictions.add(Arrays.asList(
                Math.log(1.0/3.0) + Math.log(3.0/4.0) + logPriors.get(0),
                Math.log(3.0/5.0) + Math.log(1.0/5.0) + logPriors.get(1)));
        
        for (int sampleIndex = 0; sampleIndex < Math.pow(2.0, model.getInputDimension());
                ++sampleIndex) {
            List<Double> actualPrediction = model.predict(testInputs[sampleIndex]);
            List<Double> expectedPrediction = expectedPredictions.get(sampleIndex);
            for (int klassIndex = 0; klassIndex < model.getNumberOfClasses(); ++klassIndex) {
                Assert.assertEquals("Mismatch(sample=" + sampleIndex + ",klass=" + klassIndex + ")",
                        1.0, expectedPrediction.get(klassIndex) / actualPrediction.get(klassIndex),
                        Math.ulp(1.0));
            }
        }
    }
    
}
