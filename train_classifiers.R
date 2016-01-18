# Train a complete set of classifiers, using caret train.  Allow caret more
# leeway for parameter selection by letting it do more CV.  In particular, do
# repeatedcv.  There are only 891 records in the training set.  With k=10,
# there would be about 800 to train on per fold.  But when we require an Age
# value, there are only 714 records, so about 630 to train on per fold.  The
# smaller k is, the further the quality of the model will be from training on
# all the available training data, which may make the accuracy estimate less
# good.  But for a larger k there are fewer test samples per fold, which
# increases the variablity in the accuracies.  Another drawback to making k
# larger is that it will take longer to run the training.  Adding more repeats
# while holding k fixed reduces the variance in the estimated accuracy but
# multiplies runtime.  An example in the caret docs shows 5 repeats of 10-fold
# CV on a small dataset.
#
# Train two models for each classifier variant, one on all the available
# training data, without Age as a feature, and another on just the rows that
# have Age, including Age as a feature.

# Allow this to run in parallel on two cores.
require(doParallel, quietly=TRUE)
registerDoParallel(cores=2)

# Define caret and model settings.
source("settings.R")
# This expects that the training data has already been set up.
# source("setup_training_data.R")
# @ToDo: Make the setup consistent.  Should not expect one part of setup to
# have been done, but not another.

# Make this reproducible.
set.seed(54321)

# Models with keys ending in .age are trained on only the complete cases, and
# include Age as a feature.  Models with .no.age are trained on all the data,
# but omit Age as a feature.  Since these take a long time, print a message
# between runs, to show that progress is being made.  Note if this is sourced,
# one can also set echo=TRUE.
models <- list()
print("Training rf without Age")
models$rf.no.age <-
    train(formula.all.but.age, train.minimal, method="rf",
          trControl=trctrl_repeatedcv_20_5)
print("Training rf with Age")
models$rf.age <-
    train(formula.all, train.no.na,
          method="rf",
          trControl=trctrl_repeatedcv_20_5)
print("Training glmboost without Age")
models$glmboost.no.age <-
    train(formula.all.but.age, train.minimal,
          method="glmboost", family=Binomial(),
          trControl=trctrl_repeatedcv_20_5)
print("Training glmboost with Age")
models$glmboost.age <-
    train(formula.all, train.no.na,
          method="glmboost", family=Binomial(),
          trControl=trctrl_repeatedcv_20_5)
# avNNet does not have an option to suppress messages, and it prints a lot...
print("Training avNNet without Age")
models$avnnet.no.age <- suppressMessages(
    train(formula.all.but.age, train.minimal,
          method="avNNet",
          trControl=trctrl_repeatedcv_20_5)
    )
print("Training avNNet with Age")
models$avnnet.age <- suppressMessages(
    train(formula.all, train.no.na,
          method="avNNet",
          trControl=trctrl_repeatedcv_20_5)
    )
print("Training gbm bernoulli without Age")
models$gbm_bernoulli.no.age <-
    train(formula.all.but.age, train.minimal,
          method="gbm", distribution="bernoulli", verbose=FALSE,
          trControl=trctrl_repeatedcv_20_5)
print("Training gbm bernoulli with Age")
models$gbm_bernoulli.age <-
    train(formula.all, train.no.na,
          method="gbm", distribution="bernoulli", verbose=FALSE,
          trControl=trctrl_repeatedcv_20_5)
print("Training gbm adaboost without Age")
models$gbm_adaboost.no.age <-
    train(formula.all.but.age, train.minimal,
          method="gbm", distribution="adaboost", verbose=FALSE,
          trControl=trctrl_repeatedcv_20_5)
print("Training gbm adaboost with Age")
models$gbm_adaboost.age <-
    train(formula.all, train.no.na,
          method="gbm", distribution="adaboost", verbose=FALSE,
          trControl=trctrl_repeatedcv_20_5)

# Write the models out to an Rds file.
saveRDS(models, "models.Rds")
# @ToDo: Check for an old version and make a backup."

# Show the resulting accuracies.
print("Average accuracies per classifier type:")
sapply(accuracies, `[[`, "avg_accuracy")