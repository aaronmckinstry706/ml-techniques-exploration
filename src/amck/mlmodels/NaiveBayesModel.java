package amck.mlmodels;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

//TODO: Add parameter checking to methods.
//TODO: Rename loop variables to "___Index" for clarity. 
//TODO: Enforce consistent use of "klass"/"Klass" everywhere. 
//TODO: USE LIST<>, NOT ARRAYLIST<>, for variable types. Geez, like, what are even you doing?

/**
 * A class for training a Naive Bayes model. The model is initialized so that all classes are given
 * equal probability, regardless of inputs. Only data passed to the most recent call to train() will
 * affect the model's output.
 */
public class NaiveBayesModel {
    
    private static final Logger LOGGER = Logger.getLogger(NaiveBayesModel.class.getName());

    private ArrayList<Double> logPriors;
    private ArrayList<ArrayList<HashMap<Boolean, Double>>> logLikelihoods;
    private int inputDimension;
    
    /**
     * A class for constructing NaiveBayesModel objects. 
     */
    public static class Builder {
        
        private int inputDim;
        private boolean inputDimSet;
        private int numOfClasses;
        private boolean numOfClassesSet;
        
        /**
         * Constructs a Builder object. 
         */
        public Builder() {
            inputDim = 0;
            numOfClasses = 0;
        }
        
        /**
         * @param inputDim - A positive integer. 
         * @return The builder object. 
         */
        public Builder inputDimension(int inputDim) {
            this.inputDim = inputDim;
            inputDimSet = true;
            return this;
        }

        
        /**
         * @param numOfClasses - A positive integer. 
         * @return The builder object. 
         */
        public Builder numberOfClasses(int numOfClasses) {
            this.numOfClasses = numOfClasses;
            numOfClassesSet = true;
            return this;
        }
        
        /**
         * Can only be called after both numberOfClasses() and inputDimension() have been called. 
         * 
         * @return A new NaiveBayesModel object. 
         */
        public NaiveBayesModel build() {
            if (!numOfClassesSet)
                throw new IllegalStateException("numberOfClasses was not set.");
            if (!inputDimSet)
                throw new IllegalStateException("inputDimension was not set.");
            return new NaiveBayesModel(inputDim, numOfClasses);
        }
    }

    /**
     * Constructs a Naive Bayes model which assigns a uniform likelihood to any input.
     * 
     * @param inputDimension
     *            - A positive integer.
     * @param numberOfClasses
     *            - A positive integer.
     */
    public NaiveBayesModel(int inputDimension, int numberOfClasses) {
        logPriors = new ArrayList<Double>(numberOfClasses);
        Double uniformLogPrior = new Double(Math.log(1.0 / numberOfClasses));
        for (int i = 0; i < numberOfClasses; ++i) {
            logPriors.add(uniformLogPrior);
        }

        logLikelihoods = new ArrayList<ArrayList<HashMap<Boolean, Double>>>(numberOfClasses);
        Double uniformLogLikelihood = new Double(Math.log(0.5));
        for (int klass = 0; klass < numberOfClasses; ++klass) {
            logLikelihoods.add(new ArrayList<HashMap<Boolean, Double>>(inputDimension));
            for (int feature = 0; feature < inputDimension; ++feature) {
                logLikelihoods.get(klass).add(new HashMap<Boolean, Double>(2));
                logLikelihoods.get(klass).get(feature).put(Boolean.TRUE, uniformLogLikelihood);
                logLikelihoods.get(klass).get(feature).put(Boolean.FALSE, uniformLogLikelihood);
            }
        }

        this.inputDimension = inputDimension;
    }

    /**
     * Resets the predicted probabilities to be uniform across classes, for all inputs--i.e., so
     * that <br>
     * <code>predict(x).get(i) == Math.log(1.0/getNumberOfClasses()) + getInputDimension()*Math.log(0.5)</code><br>
     * for all <code>i</code> and <code>x</code>.
     */
    private void reset() {
        Collections.fill(logPriors, new Double(Math.log(1.0 / logPriors.size())));
        for (int klass = 0; klass < logLikelihoods.size(); ++klass) {
            for (int feature = 0; feature < logLikelihoods.get(klass).size(); ++feature) {
                logLikelihoods.get(klass).get(feature).put(Boolean.TRUE, new Double(Math.log(0.5)));
                logLikelihoods.get(klass).get(feature).put(Boolean.FALSE,
                        new Double(Math.log(0.5)));
            }
        }
    }

    /**
     * Returns <code>log P(class==i | input)</code> for all classes <code>i</code>, according to the
     * model.
     * 
     * @param input
     *            - A boolean array of length <code>getInputDimension()</code>.
     * @return logProbabilities - An ArrayList&lt;Double&gt;. <code>logProbabilities.get(i)</code>
     *         is the predicted value of <code>log P(class==i | input)</code>.
     */
    public ArrayList<Double> predict(boolean[] input) {
        ArrayList<Double> logProbabilities = new ArrayList<Double>();
        for (int klass = 0; klass < getNumberOfClasses(); ++klass) {
            double logProbability = logPriors.get(klass);
            for (int inputIndex = 0; inputIndex < getInputDimension(); ++inputIndex) {
                logProbability += logLikelihoods.get(klass).get(inputIndex).get(input[inputIndex]);
            }
            logProbabilities.add(new Double(logProbability));
        }

        return logProbabilities;
    }

    /**
     * Takes a set of inputs and labels, and trains the model. It must be true that
     * <code>inputs.length == labels.length</code>; in addition, for all valid <code>i</code>, it
     * must be true that <code>inputs[i].length == getInputDimension()</code>.
     * 
     * @param inputs
     *            Matrix of sample inputs.
     * @param labels
     *            Vector of sample labels. Each value is an integer in the range
     *            <code>[0, getNumberOfClasses() - 1]</code> (inclusive).
     */
    public void train(boolean[][] inputs, int[] labels) {
        reset();
        
        // Initialize the counting data structures.
        
        ArrayList<ArrayList<HashMap<Boolean, Double>>> sampleCounts = 
                new ArrayList<ArrayList<HashMap<Boolean, Double>>>();
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            sampleCounts.add(new ArrayList<HashMap<Boolean, Double>>());
            for (int featureIndex = 0; featureIndex < getInputDimension(); ++featureIndex) {
                sampleCounts.get(klassIndex).add(new HashMap<Boolean, Double>());
                sampleCounts.get(klassIndex).get(featureIndex).put(Boolean.TRUE, 0.0);
                sampleCounts.get(klassIndex).get(featureIndex).put(Boolean.FALSE, 0.0);
            }
        }
        
        ArrayList<Double> classCounts = new ArrayList<Double>();
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            classCounts.add(0.0);
        }
        
        // Count. 
        
        for (int sampleIndex = 0; sampleIndex < inputs.length; ++sampleIndex) {
            int klassIndex = labels[sampleIndex];
            classCounts.set(klassIndex, classCounts.get(klassIndex) + 1);
            for (int featureIndex = 0; featureIndex < getInputDimension(); ++featureIndex) {
                HashMap<Boolean, Double> oldCounts = sampleCounts.get(klassIndex).get(featureIndex);
                Boolean feature = inputs[sampleIndex][featureIndex];
                oldCounts.put(feature, oldCounts.get(feature) + 1);
            }
        }
        
        // Calculate log probabilities. 
        
        double sum = 0.0;
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            sum = sum + classCounts.get(klassIndex);
        }
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            logPriors.set(klassIndex, Math.log(classCounts.get(klassIndex)/sum));
        }
        
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            for (int featureIndex = 0; featureIndex < getInputDimension(); ++featureIndex) {
                //TODO: Think of better prefix here than "local". 
                HashMap<Boolean, Double> localFeatureCounts = sampleCounts.get(klassIndex)
                        .get(featureIndex);
                double totalCount = localFeatureCounts.get(Boolean.TRUE)
                        + localFeatureCounts.get(Boolean.FALSE);
                HashMap<Boolean, Double> localLogLikelihoods = logLikelihoods.get(klassIndex)
                        .get(featureIndex);
                
                localLogLikelihoods.put(Boolean.TRUE, Math.log(
                        localFeatureCounts.get(Boolean.TRUE)/totalCount));
                localLogLikelihoods.put(Boolean.FALSE, Math.log(
                        localFeatureCounts.get(Boolean.FALSE)/totalCount));
            }
        }
        
        LOGGER.log(Level.INFO, "logPriors={0}, logLikelihoods={1}", new Object[] {logPriors,
                logLikelihoods});
    }

    /**
     * @return The number of classes for which this model predicts.
     */
    public int getNumberOfClasses() {
        return logPriors.size();
    }

    /**
     * @return The number of dimensions in the model's input.
     */
    public int getInputDimension() {
        return inputDimension;
    }

}
