package amck.mlmodels;

import java.util.ArrayList;

import org.junit.Assert;
import org.junit.Test;

public class NaiveBayesModelTest {
    
    @Test
    public void testUntrainedModel() {
        int numberOfClasses = 2;
        int inputDimension = 2;
        
        NaiveBayesModel model = new NaiveBayesModel.Builder().inputDimension(2).numberOfClasses(2)
                .build();
        Double uniformLogProbability = new Double(Math.log(1.0/numberOfClasses)
                + inputDimension*Math.log(0.5));
        ArrayList<Double> expected = new ArrayList<Double>(numberOfClasses);
        expected.add(uniformLogProbability);
        expected.add(uniformLogProbability);
        Assert.assertEquals(expected, model.predict(new boolean[] {true, true}));
    }
    
}
