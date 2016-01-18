# This shows the order in which the scripts should be run.
# It's recommended to look at the scripts before running each, then run them
# one at a time, and examine the results.
# Some of the scripts (do_cv_acc_only.R and settings.R) are sourced from others,
# so do not appear here.

# These are the packages needed -- load them here so this will stop right away
# if any need to be installed.
require(caret, quietly=TRUE)
require(car, quietly=TRUE)
require(rpart, quietly=TRUE)

# Classifier packages
require(gbm, quietly=TRUE)
require(mboost, quietly=TRUE)  # for glmboost and the Binomial function
require(nnet, quietly=TRUE)    # for avNNet
require(randomForest, quietly=TRUE)

# If multiple CPUs / cores can't be used, comment this out here and in both
# assess_classifiers.R and train_classifiers.R
require(doParallel, quietly=TRUE)

# Read in the data.
source("read_data.R")

# Preliminary examination of the data, and data cleaning.  This will display
# several plots, so should be run with the display available.
source ("examine_data.R")

# Make minimal data sets, without all the extra columns added during data
# exploration.  The cross-validation process creates many subsets of the
# data, so having no more columns than needed reduces the runtime and
# memory use.
source("setup_training_data.R")

# Estimate the out-of-sample accuracy for a number of potentially useful
# types of classifiers.
source("assess_classifiers.R")

# @ToDo: Extend the classifier assessment to do explicit parameter tuning
# by either extracting the parameters chosen by caret internally, or providing
# caret with a grid of parameter values to try.

# Now re-train those classifiers on all the training data.
source("train_classifiers.R")

# Because the training set is so small, we can't afford to hold out any data
# to do an out-of-sample accuracy check on the final classifier.  Instead,
# we'll have to rely on Kaggle's public leaderboard score.  That's not the
# real out-of-sample accuracy, since we can re-submit over and over -- the
# real accuracy is the private score, which is not available.  Here, get the
# *in*-sample accuracy, on the training data.  This is an over-estimate of
# the real out-of-sample accuracy, but we can use this to detect overfitting,
# if the in-sample accuracy is quite a bit higher than the public leaderboard
# accuracy.
source("vote_in_sample_accuracy.R")

# Run prediction on the test set, and prepare a CSV file for uploading to
# Kaggle.
# @ToDo: If there is already a CSV file, rename it.
source("vote_on_test.R")

# Adjust the predictions to account for the survival status of cabin-mates
# for passengers sharing cabins.
source("include_cabin_mates.R")