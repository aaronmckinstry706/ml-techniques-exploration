# ml-techniques-exploration

In an effort to familiarize myself with the various techniques common in machine learning (and in an effort to showcase some Java code), I'll be implementing a few models from scratch. The All models can be trained with, and make predictions on, a single CPU (multi-threading may come later, but for now it'll be single-threaded). As I begin implementing each model, I'll be adding a description of the technique to this document. 

## Naive Bayes

The Naive Bayes technique is a conditional probability model. Given an input vector ``X = x_1, ..., x_n`` of Boolean random variables, the model calculates the conditional distribution ``P(C | X)`` over the set of classes C_1, ..., C_k. Using Bayes' theorem from probability, one can rewrite this as: ``P(C_i | X) = P(X | C_i) * P(C_i) / P(X)``. 

The term ``P(X | C_i)`` is called the "likelihood". The term ``P(C_i)`` is called the "prior". The term ``P(X)`` is called the "evidence". Since ``P(X) = 1`` (``X`` is our input, so it's fixed), we can ignore this value in our calculations. We now need to concern ourselves with two values: the prior, and the likelihood. We can calculate the prior and the likelihood for each class by counting a set of labeled examples&mdash;i.e., our training data. 

However, this would seem to pose a problem! After all, we need a number of labeled examples which is *exponential* (in ``n``, the size of ``X``) in order to accurately calculate ``P(X | C_i)``, right? This is where we make the naive assumption that all the features are conditionally independent given the class; mathematically, we are assuming ``P(X | C_i) = P(x_1 | C_i) * ... * P(x_n | C_i)``. Because of this assumption, we only need a number of labeled examples which is *linear* in the size ``X``. Cool!

There are a couple of notes to make here&mdash;some about the implementation, some about the model:
* I have implemented the probability model exactly&mdash;that is, I simply count the labeled examples to calculate the conditional distribution over ``C_i``. There are various techniques for smoothing probabilities, but I have not used any here. 
* I actually store/return/manipulate *log* probabilities. This turns multiplications into addition, and also helps with accuracy when ``n`` is large (somewhere on the order of 50 or above). 
* If one is implementing a classification model using Naive Bayes, then the chosen class is ``argmax P(C_i | X) over all C_i``. Because the classifier uses ``argmax``, a curious thing happens when implementing the classification model: it does not matter whether ``sum P(C_i) over all i`` is equal to 1! This is because, given some arbitrary values for ``P(C_i)``, normalizing these values is equivalent to multiplying each term in the ``argmax`` by a positive normalizing constant ``1/Z``&mdash;and ``argmax`` is invariant to positive scalar multiplication. 
