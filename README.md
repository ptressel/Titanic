# Titanic

These are scripts for Kaggle Titanic competition attempt, written for the Coursera U Washington
Practical Predictive Analytics course:

https://www.coursera.org/learn/predictive-analytics

This does data cleaning, and shows some initial exploration, then constructs a set of classifiers
that are used to predict an weighted average probability of the binary outcome, where weights are
the relative out-of-sample accuracies of the classifiers, estimated by cross-validation.  This
overfits wildly, so needs work to control that.  Its score on the public leaderboard is not much
better than the simple gender model.

The run_scripts.R script runs the entire process, but please have a look before running it.
This uses a number of R packages, which should be installed before running.
The assess_classifiers.R step may take significant time to complete.


