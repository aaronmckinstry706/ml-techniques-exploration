package amck.mlmodels;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A class for training a Naive Bayes model. The model is initialized so that all classes are given
 * equal probability, regardless of inputs. Only data passed to the most recent call to train() will
 * affect the model's output.
 */
public class NaiveBayesModel {

    private static final Logger LOGGER = Logger.getLogger(NaiveBayesModel.class.getName());

    private List<Double> logPriors;
    private List<List<Map<Boolean, Double>>> logLikelihoods;
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
         * @param inputDim
         *            - A positive integer.
         * @return The builder object.
         */
        public Builder inputDimension(int inputDim) {
            this.inputDim = inputDim;
            inputDimSet = true;
            return this;
        }

        /**
         * @param numOfClasses
         *            - A positive integer.
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
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("inputDimension must be positive.");
        }
        if (numberOfClasses <= 0) {
            throw new IllegalArgumentException("numberOfClasses must be positive.");
        }

        logPriors = new ArrayList<Double>(numberOfClasses);
        Double uniformLogPrior = new Double(Math.log(1.0 / numberOfClasses));
        for (int i = 0; i < numberOfClasses; ++i) {
            logPriors.add(uniformLogPrior);
        }

        logLikelihoods = new ArrayList<List<Map<Boolean, Double>>>(numberOfClasses);
        Double uniformLogLikelihood = new Double(Math.log(0.5));
        for (int klasIndex = 0; klasIndex < numberOfClasses; ++klasIndex) {
            logLikelihoods.add(new ArrayList<Map<Boolean, Double>>(inputDimension));
            for (int featureIndex = 0; featureIndex < inputDimension; ++featureIndex) {
                logLikelihoods.get(klasIndex).add(new HashMap<Boolean, Double>(2));
                logLikelihoods.get(klasIndex).get(featureIndex).put(Boolean.TRUE,
                        uniformLogLikelihood);
                logLikelihoods.get(klasIndex).get(featureIndex).put(Boolean.FALSE,
                        uniformLogLikelihood);
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
        for (int klassIndex = 0; klassIndex < logLikelihoods.size(); ++klassIndex) {
            for (int featureIndex = 0; featureIndex < logLikelihoods.get(klassIndex)
                    .size(); ++featureIndex) {
                logLikelihoods.get(klassIndex).get(featureIndex).put(Boolean.TRUE,
                        new Double(Math.log(0.5)));
                logLikelihoods.get(klassIndex).get(featureIndex).put(Boolean.FALSE,
                        new Double(Math.log(0.5)));
            }
        }
    }

    /**
     * Returns <code>log P(class==i | input)</code> for all classes <code>i</code>, according to the
     * model.
     * 
     * @param input
     *            - A boolean List of size <code>getInputDimension()</code>.
     * @return logProbabilities - A List&lt;Double&gt;. <code>logProbabilities.get(i)</code> is the
     *         predicted value of <code>log P(class==i | input)</code>.
     */
    public List<Double> predict(List<Boolean> input) {
        if (input.size() != getInputDimension()) {
            throw new IllegalArgumentException(
                    "input.length must be equal to getInputDimension().");
        }

        List<Double> logProbabilities = new ArrayList<Double>();
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            double logProbability = logPriors.get(klassIndex);
            for (int featureIndex = 0; featureIndex < getInputDimension(); ++featureIndex) {
                logProbability += logLikelihoods.get(klassIndex).get(featureIndex)
                        .get(input.get(featureIndex));
            }
            logProbabilities.add(new Double(logProbability));
        }

        return logProbabilities;
    }

    /**
     * Takes a set of inputs and labels, and trains the model. It must be true that
     * <code>inputs.size() == labels.size()</code>; in addition, for all valid <code>i</code>, it
     * must be true that <code>inputs.get(i).size() == getInputDimension()</code>.
     * 
     * @param inputs
     *            Matrix of sample inputs.
     * @param labels
     *            List of sample labels. Each value is an integer in the range
     *            <code>[0, getNumberOfClasses() - 1]</code> (inclusive).
     */
    public void train(List<List<Boolean>> inputs, List<Integer> labels) {
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("inputs.length must be equal to labels.length.");
        }

        reset();

        // Initialize the counting data structures.

        List<List<Map<Boolean, Double>>> sampleCounts = new ArrayList<List<Map<Boolean, Double>>>();
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            sampleCounts.add(new ArrayList<Map<Boolean, Double>>());
            for (int featureIndex = 0; featureIndex < getInputDimension(); ++featureIndex) {
                sampleCounts.get(klassIndex).add(new HashMap<Boolean, Double>());
                sampleCounts.get(klassIndex).get(featureIndex).put(Boolean.TRUE, 0.0);
                sampleCounts.get(klassIndex).get(featureIndex).put(Boolean.FALSE, 0.0);
            }
        }

        List<Double> klassCounts = new ArrayList<Double>();
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            klassCounts.add(0.0);
        }

        // Count.

        for (int sampleIndex = 0; sampleIndex < inputs.size(); ++sampleIndex) {
            if (inputs.get(sampleIndex).size() != getInputDimension()) {
                throw new IllegalArgumentException(
                        "inputs[i].length must equal getInputDimension()" + " for all valid i.");
            }

            int klassIndex = labels.get(sampleIndex);
            klassCounts.set(klassIndex, klassCounts.get(klassIndex) + 1);
            for (int featureIndex = 0; featureIndex < getInputDimension(); ++featureIndex) {
                Map<Boolean, Double> oldCounts = sampleCounts.get(klassIndex).get(featureIndex);
                Boolean feature = inputs.get(sampleIndex).get(featureIndex);
                oldCounts.put(feature, oldCounts.get(feature) + 1);
            }
        }

        // Calculate log probabilities.

        double sum = 0.0;
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            sum = sum + klassCounts.get(klassIndex);
        }
        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            logPriors.set(klassIndex, Math.log(klassCounts.get(klassIndex) / sum));
        }

        for (int klassIndex = 0; klassIndex < getNumberOfClasses(); ++klassIndex) {
            for (int featureIndex = 0; featureIndex < getInputDimension(); ++featureIndex) {
                Map<Boolean, Double> conditionalFeatureCounts = sampleCounts.get(klassIndex)
                        .get(featureIndex);
                double totalCount = conditionalFeatureCounts.get(Boolean.TRUE)
                        + conditionalFeatureCounts.get(Boolean.FALSE);
                Map<Boolean, Double> localLogLikelihoods = logLikelihoods.get(klassIndex)
                        .get(featureIndex);

                localLogLikelihoods.put(Boolean.TRUE,
                        Math.log(conditionalFeatureCounts.get(Boolean.TRUE) / totalCount));
                localLogLikelihoods.put(Boolean.FALSE,
                        Math.log(conditionalFeatureCounts.get(Boolean.FALSE) / totalCount));
            }
        }
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
