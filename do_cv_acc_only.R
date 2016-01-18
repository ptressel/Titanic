# Do cross-validation external to caret, for the purpose *only* of estimating
# out-of-sample error, not for parameter tuning, which is apparently what
# caret's internal CV and bootstrapping training methods are for.  That renders
# them useless for true OOS error estimation, as they have been used to select
# the parameters and the final model.

#require(doParallel, quietly=TRUE)
#registerDoParallel(cores=2)
#require(caret, quietly=TRUE)

# formula: standard outcome ~ predictors formula, to be passed to train.
# d: data frame or table holding the dataset. Must be row-indexable.
# k: number of folds.
# train_method: the classifier method name, for train.
# train_params: other params to be passed to train, e.g. trControl and tuneGrid.
# method_params: other params to be passed to the classifier.
# The two params args should be supplied as lists, with param name as
# the list item name, and value as the item content.
# @ToDo: Add r for number of repeats, maybe params for createFolds?
#
# Can set a random seed before the call to get reproducible results,
# but better to try it first, without setting the seed, several times.
# If accuracy isn't close on repeated random runs, then something is
# wrong, like too little data.
#
# Return value is the average accuracy over the splits, along with the
# accuracy and the returned model from train on each split.

# Process one fold, return accuracy.
do_fold <- function(test_rows, f, d, train_method,
                    train_params, tune_params, method_params) {
    train_rows <- setdiff(1:nrow(d), test_rows)
    model <- do.call(train, c(list(form=f, d=d[train_rows, ]), method=train_method,
                              train_params, method_params))
    pred <- predict(model, newdata=d[test_rows,])
    outcome <- all.vars(f[[2]])
    accuracy <- sum(pred == d[test_rows, outcome]) / length(test_rows)
    accuracy
}

do_rep <- function(rep, f, outcome, d, k, train_method,
                   train_params, tune_params, method_params) {
    # Get the data splits for test set rows.
    test_sets <- createFolds(d[,outcome], k=k, list=TRUE, returnTrain=FALSE)

    # Loop over hold-out splits, call train on the remaining data,
    # predict on the held out split, get accuracy.  This asks for hard
    # class predictions, not probability, so if a threshold for (say)
    # logistic regression is wanted that is not 0.5, caller needs to
    # handle that with method_params.
    fold_accuracies <- sapply(test_sets, do_fold, f=f, d=d,
                              train_method=train_method,
                              train_params=train_params,
                              method_params=method_params)

    # Compute the average accuracy for this repeat.
    rep_accuracy <- sum(fold_accuracies) / k

    list(rep=rep, rep_accuracy=rep_accuracy, fold_accuracies=fold_accuracies)
}

# Process n repeats of k folds, return average accuracy and collected accuracy
# results of each fold.
do_cv <- function(f, d, n, k, train_method,
                  train_params=list(), method_params=list()) {

    # Get the dependent variable / outcome column out of the formula.
    outcome <- all.vars(f[[2]])

    # Do the repeats.
    rep_accuracies <- lapply(1:n, do_rep,
                             f=f, outcome=outcome, d=d, k=k,
                             train_method=train_method,
                             train_params=train_params,
                             method_params=method_params)

    # Compute the average accuracy over the repeats.
    avg_accuracy <- sum(unlist(lapply(rep_accuracies, function(item) item["rep_accuracy"]))) / n

    list(avg_accuracy=avg_accuracy, rep_accuracies=rep_accuracies)
}