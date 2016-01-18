# Try out various classifiers on the Titanic data, using CV to get
# generalization performance.  The metric is accuracy, since, for the Kaggle
# task, we are asked to predict the actual outcome, not the probability.
# We don't have enough data to hold out a test set, and there is already a
# sizeable test set for the competition.  So we'll do CV for parameter setting
# and to select classifiers and estimate their out-of-sample accuracy, then
# re-train the chosen classifiers with their chosen parameters on the entire
# training set.  Re-use the CV framework I wrote for the previous Kaggle
# competition.  This is a small dataset, so use a fairly high k, so there is
# enough data to train on for each fold.  I'm not using glm here, and most of
# the classifiers I will use are capable of placing non-linear boundaries.
# So for a first cut, don't restrict the features unless the classifier has
# issues with NAs.  For NA-sensitive classifiers, do separate runs with and
# without Age, and where Age is included, only use complete rows.

# Allow this to run in parallel on two cores.
require(doParallel, quietly=TRUE)
registerDoParallel(cores=2)

require(caret, quietly=TRUE)
require(mboost, quietly=TRUE)  # for glmboost and the Binomial function
source("settings.R")
source("do_cv_acc_only.R")
# This expects that the training data has already been set up.

# Make this reproducible.
set.seed(12345)

# None of these classifiers natively handle NAs, so try all the data with no
# Age, and complete cases with Age.
accuracies <- list()
# rf_oob_10_10.no.age <-
#     do_cv(formula.all.but.age, train.minimal, 10, 10, "rf",
#           train_params=list(trControl=trctrl_rf_oob))
# rf_oob_10_10.no.age$avg_accuracy
print("Assessing rf without Age")
#rf_repeatedcv_10_10.no.age <-
accuracies$rf.no.age <-
    do_cv(formula.all.but.age, train.minimal, 10, 10, "rf",
          train_params=list(trControl=trctrl_repeatedcv_10_1))
# rf_oob_10_10.all.no.na <-
#     do_cv(formula.all, train.no.na, 10, 10, "rf",
#           train_params=list(trControl=trctrl_rf_oob))
# rf_oob_10_10.all.no.na$avg_accuracy
print("Assessing rf with Age")
#rf_repeatedcv_10_10.all.no.na <-
accuracies$rf.age <-
    do_cv(formula.all, train.no.na, 10, 10, "rf",
          train_params=list(trControl=trctrl_repeatedcv_10_1))
print("Assessing glmboost without Age")
#glmboost_10_10.no.age <-
accuracies$glmboost.no.age <-
    do_cv(formula.all.but.age, train.minimal, 10, 10, "glmboost",
          train_params=list(trControl=trctrl_repeatedcv_10_1),
          method_params=params_glmboost)
print("Assessing glmboost with Age")
#glmboost_10_10.all.no.na <-
accuracies$glmboost.age <-
    do_cv(formula.all, train.no.na, 10, 10, "glmboost",
          train_params=list(trControl=trctrl_repeatedcv_10_1),
          method_params=params_glmboost)
print("Assessing avNNet without Age")
#avnnet_10_10.no.age <-
accuracies$avnnet.no.age <-
    suppressMessages(
        do_cv(formula.all.but.age, train.minimal, 10, 10, "avNNet",
              train_params=list(trControl=trctrl_repeatedcv_10_1)))
print("Assessing avNNet with Age")
#avnnet_10_10.all.no.na <-
accuracies$avnnet.age <-
    suppressMessages(
        do_cv(formula.all, train.no.na, 10, 10, "avNNet",
              train_params=list(trControl=trctrl_repeatedcv_10_1)))
print("Assessing gbm bernoulli without Age")
#gbm_bernoulli_10_10.no.age <-
accuracies$gbm_bernoulli.no.age <-
    do_cv(formula.all.but.age, train.minimal, 10, 10, "gbm",
          train_params=list(trControl=trctrl_repeatedcv_10_1),
          method_params=params_gbm_bernoulli)
print("Assessing gbm bernoulli with Age")
#gbm_bernoulli_10_10.all.no.na <-
accuracies$gbm_bernoulli.age <-
    do_cv(formula.all, train.no.na, 10, 10, "gbm",
          train_params=list(trControl=trctrl_repeatedcv_10_1),
          method_params=params_gbm_bernoulli)
print("Assessing gbm adaboost without Age")
#gbm_adaboost_10_10.no.age <-
accuracies$gbm_adaboost.no.age <-
    do_cv(formula.all.but.age, train.minimal, 10, 10, "gbm",
          train_params=list(trControl=trctrl_repeatedcv_10_1),
          method_params=params_gbm_adaboost)
print("Assessing gbm adaboost with Age")
#gbm_adaboost_10_10.all.no.na <-
accuracies$gbm_adaboost.age <-
    do_cv(formula.all, train.no.na, 10, 10, "gbm",
          train_params=list(trControl=trctrl_repeatedcv_10_1),
          method_params=params_gbm_adaboost)

# These are the avg_accuracy values for each case:
# > accuracies.only
#            rf.no.age               rf.age      glmboost.no.age
#            0.8086579            0.8127113            0.7855175
#         glmboost.age        avnnet.no.age           avnnet.age
#            0.7874785            0.8002168            0.8159605
# gbm_bernoulli.no.age    gbm_bernoulli.age  gbm_adaboost.no.age
#            0.7983048            0.8183901            0.7952637
#     gbm_adaboost.age
#            0.8140004

# Write the accuracies out to an Rds file.
saveRDS(accuracies, "accuracies.Rds")
# @ToDo: Check for an old version and make a backup."