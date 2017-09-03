package amck.mlmodels;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * A class for training a Naive Bayes model. The model is initialized so that all classes are given
 * equal probability, regardless of inputs. Only data passed to the most recent call to train() will
 * affect the model's output.
 */
public class NaiveBayesModel {

    private ArrayList<Double> logPriors;
    private ArrayList<ArrayList<HashMap<Boolean, Double>>> logLikelihoods;
    private int inputDimension;
    
    /**
     * A class for constructing NaiveBayesModel objects. 
     */
    public static class Builder {
        
        private int inputDim;
        private int numOfClasses;
        
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
            if (inputDim <= 0)
                throw new IllegalArgumentException("inputDim " + inputDim
                        + " must be a positive integer.");
            this.inputDim = inputDim;
            return this;
        }

        
        /**
         * @param numOfClasses - A positive integer. 
         * @return The builder object. 
         */
        public Builder numberOfClasses(int numOfClasses) {
            if (numOfClasses <= 0)
                throw new IllegalArgumentException("numOfClasses " + numOfClasses
                        + " must be a positive integer.");
            this.numOfClasses = numOfClasses;
            return this;
        }
        
        /**
         * Can only be called after both numberOfClasses() and inputDimension() have been called. 
         * 
         * @return A new NaiveBayesModel object. 
         */
        public NaiveBayesModel build() {
            if (numOfClasses <= 0)
                throw new IllegalStateException("numberOfClasses was not set.");
            if (inputDim <= 0)
                throw new IllegalStateException("inputDimension was not set.");
            return new NaiveBayesModel(inputDim, numOfClasses);
        }
    }

    /**
     * Constructs a Naive Bayes model which assigns a uniform likelihood to any input.
     * 
     * @param inputDimension
     *            -- A positive integer.
     * @param numberOfClasses
     *            -- A positive integer.
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
        ArrayList<Double> logProbabilities = new ArrayList<Double>(getNumberOfClasses());
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
        //TODO: Train the model. 
    }

    /**
     * @return The number of classes for which this model predicts.
     */
    public int getNumberOfClasses() {
        return this.logPriors.size();
    }

    /**
     * @return The number of dimensions in the model's input.
     */
    public int getInputDimension() {
        return inputDimension;
    }

}
