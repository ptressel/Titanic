# Attempt to include cabin information as follows:
# When a passenger has a cabin-mate in the training data, include a trivial
# classifier that predicts the same survival value as the cabin-mate.  If there
# are multiple cabin-mates, only use their information if all outcomes match.
# This turns out to be irrelevant, as there are very few such cases.
# With this addition, the age and no-age cases are each split into two.
# This expects to be provided with a function that will predict based on
# cabin information.  No attempt is made to fit this in under the predict
# generic function.  Also, there is no good way to do CV using this prediction
# as when we drop out rows with cabin-mates, we also prevent using them for
# prediction.  The estimated accuracy is thus just the fraction of cabin-mates
# whose outcomes match.  Another way to say this is, there is no training.
# This is instance-based.  If there is no instance to pair with, then there
# is no prediction.  Thus, no predictions are needed on the training data.
# We can compute them once on the test data.  Only 46 test rows (11%) have
# cabin-mates.  For those, we can adjust the predicted probabilities.  Thus
# this has to be run after vote_on_test.R, so that vote.probs.on.test is
# available.  The relevant extraction of cabin-mate info is done here, though
# some was done in examine_data.R.

# The updates on test rows with and without age will differ.  Need to recompute
# the weighted average of the predictions, and the normalization constants are
# different for the age and no-age cases.
if (!exists("p.shared.cabins.match")) {
    p.shared.cabins.match <- readRDS("p_shared_cabins_match.Rds")
}
test.rows.no.age <- is.na(test.minimal$Age)
accuracies.only <- sapply(accuracies, `[[`, "avg_accuracy")
norm.accuracies.no.age <- sum(accuracies.only[models.no.age])
norm.accuracies.age <- sum(accuracies.only[models.age])

# Copy the earlier predictions.
probs.on.test.with.cabin.mates <- vote.probs.on.test

# Get the survival information for passengers in test that have cabin-mates in
# train.  We don't need to prevent passengers from matching themselves, as the
# test rows are distinct from the train rows.  The main purpose of extracting
# this data is to avoid fuss with NAs.
train.cabin.info <- train.set[!is.na(train.set$Cabin), c("Survived", "Cabin")]
# Now match the test rows with train rows.  This provides a logical index for
# the rows that may need their predicted probabilities changed.
test.cabin.in.train <-
    sapply(test.set$Cabin, function (c) {
        !is.na(c) & c %in% train.cabin.info$Cabin
    })

# For each cabin in test (that has any match in train), extract the matching
# train survival information.  Almost no attempt is made to optimize this.
# Some test cabins match more than one train cabin, in which case the result
# might be ambiguous.  Here, don't use any ambiguous cases, where not all
# outcomes are the same.
for (i in seq_along(test.set$Cabin)) {
    if (test.cabin.in.train[i]) {
        outcomes <- train.cabin.info$Survived[train.cabin.info$Cabin ==
                                              test.set$Cabin[i]]
        net.outcome <- sum(outcomes) / length(outcomes)
        if (net.outcome == 0 | net.outcome == 1) {
            # Here, we have an unambiguous cabin-mate outcome.  Update the
            # probability by updating the weighted sum of probabilities,
            # separately for the age and no-age cases.
            old.prob <- vote.probs.on.test[i]
            old.norm <- if (test.rows.no.age[i]) {
                norm.accuracies.no.age
            } else {
                norm.accuracies.age
            }
            new.norm <- old.norm + p.shared.cabins.match
            new.prob <-
                (old.prob * old.norm + net.outcome * p.shared.cabins.match) /
                new.norm
            probs.on.test.with.cabin.mates[i] <- new.prob
        }
    }
}

preds.on.test.with.cabin.mates <-
    as.numeric(probs.on.test.with.cabin.mates > 0.5)

# Did that change any predictions?  Ooo, a whole 2 of them.  And those might
# not be part of the public test set on Kaggle.  On the other hand, the fact
# that most match the old predictions helps confirm the old predictions.
#> sum(preds.on.test.with.cabin.mates != vote.preds.on.test)
#[1] 2

# Submit these anyway...
test.submit.with.cabin.mates <-
    data.frame(PassengerId=test.set$PassengerId,
               Survived=preds.on.test.with.cabin.mates)
write.csv(test.submit.with.cabin.mates,
          "vote_submission_with_cabin_mates.csv", quote=FALSE, row.names=FALSE)