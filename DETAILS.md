# Predicting Survival on the Kaggle Titanic Dataset

## Overview

(Note this writeup is part of an assignment for the Coursera U Washington Practical Predictive Analytics course.)

In the Kaggle competition "Titanic: Machine Learning from Disaster", the goal is to predict whether a passenger survived, given information about that passenger.

We are given this information (features) for each passenger, where available:  name, sex, age, number of siblings / spouse aboard, number of parents / children aboard, ticket number, passenger fare, passenger fare class, cabin, port of embarkation.  The age is unknown for about 20% of the passengers, and cabin is unknown for the majority -- 78%.  There are a very small number of unknown fares and ports.  The ticket "number" is a string of characters including letters, numbers, spaces, and punctuation -- the format and meaning of this text is not described.  Cabins are named by a deck letter followed by a number, but the cabin information given for a passenger may list multiple cabins and includes other text (described later).  There are three fare classes (1st, 2nd, 3rd), and three ports at which passengers embarked.

The data is supplied already split into a training set, which contains the survival outcome for each passenger along with the above information, and a test set, which only contains the above information.

In a Kaggle competition, one submits predictions for the test set.  The predictions are scored for accuracy on *part* of the test data and the fraction correct is reported -- this is the "public" score.  The score on the remaining "private" test data is not reported during the competition.  At the close of the competition, the private score is revealed, and this becomes the participant's score in the competition.  This prevents participants from simply trying different predictions to improve their eventual score.  The Titanic competition is slightly different, in that it is intended for practice, and has been running for several years, so the private score is not revealed.  Thus the accuracy performance that will be quoted later is the public score.

Note I'm selecting this for several reasons:  First, I would rather start with a "practice" competition that doesn't count in standing.  I've done one prior Kaggle competition, but that was a restricted competition for a class.  Second, I tend to err on the side of classifiers that will overfit, and want to work on avoiding that.  Third, I'm involved in emergency management -- I help on an open-source project for emergency management, do mapping for un-mapped disaster areas, etc.  So finding out what affected survival on the Titanic is not entirely an academic exercise.

The analysis was performed in R, using the caret machine learning package, and a number of classification and regression packages.  The code can be seen at:
https://github.com/ptressel/Titanic

## Examining and cleaning the data

Since only a few fares were missing, and since fares are related to fare class, the missing fares were replaced by the median fare for the passenger's fare class.

To infer the missing port of embarkation, the fare class and fare were used.  But this isn't as simple as separating the values into groups and taking the median -- the fare is a continuous numerical value.  Instead, we can ask, given the fare and fare class, which port is most likely?  The fare was histogrammed into 10 bins for each fare class and port -- these provide sample probability distributions of the fare for each fare class and port.  For a passenger needing the port inferred, we know the fare class -- that selects three of the histograms, one for each port.  The passenger's fare tells us the bin in each histogram -- we then choose the port for which the probability of the fare is highest, among the three histograms.

Because there were so few missing fares and ports, the choice of replacement values should not make much difference.  But having no missing values for those features simplifies the rest of the analysis, as most classifiers do not handle missing values.  Since so many age and cabin values were missing, these were not imputed.

A few more pieces of information were extracted from the name and cabin:

The name has the form:
       last name, title. first and middle names
In hope that the title might give some indication of age or status, this was extracted from the name.  There were about 20 different titles.

The cabin text included cabin numbers of the form XN where X is the deck letter and N is the room number, which may have several digits, and also the isolated letters F and T, which are not explained.  The letter T occurred only once, without a cabin number, so was taken to be missing.

The letter F is more interesting.  It is present for a small number of passengers, all but one of whom are 3rd class and have non-British names.  But apart from the F, they have the same cabin as other passengers of higher class.  So one might guess that these are assistants or servants to the other passengers.  The F was removed from the cabin text.  Since only a very small number of passengers were marked with F, it was not used for prediction.

Some cabin text included multiple cabin numbers -- this might indicate that a group of passengers had reserved a suite of cabins.  When cabins appeared in a set, all the cabin numbers were on the same deck.

Since the cabin deck might predict how long it would take for a passenger to reach the boat deck, where the lifeboats were, the deck letter was extracted from the cabin.

## Feature selection

We want to get some indication of which features are relevant to predicting survival.

A few features were dismissed at the outset.  The ticket text was not used as there was no information about the meaning of the text.  The name text, apart from the title, was not expected to be very relevant.  The cabin numbers are irregular -- the layout of each deck is very different -- and the numbers have no mathematical relationship to the distance from stairwells, etc.,so these were not expected to be helpful without getting information from the deck floorplans.

Each numerical feature (age, fare class, fare, # siblings / spouse, # parents / children) was correlated with survival.  None had a high correlation, but fare class and fare were highest.  This is a linear relationship, so there might still be a strong non-linear relationship even if the correlation is low, and it only compares features to survival separately.  Oddly, age had a low correlation, but age is known to be important in who survived.  If so, it may be that age has a non-linear relationship with survival, or depends on another feature as well, such as fare class.

Pairwise correlation between the numerical features was checked, to see if any were so highly correlated that they should not both be used.  None were so strongly correlated to be a concern.

Next, logistic regression was tried with various combinations of features, to predict survival.  This was not expected to be useful for actual prediction, but rather was used to see which features were regarded as significant by the logistic regression model.  That model can also provide variance inflation factor (VIF) values, which measures whether multiple features are linearly dependent -- a high VIF, especially multiple features with high VIF, indicates that one or more of those features aren't needed.

With all remaining features included (fare class, fare, port, sex, # siblings / spouse, # parents / children) only the fare class and # siblings / spouse are significant, and VIF is high for the title and sex.  This makes sense -- the title in a name (e.g. Mr., Mrs.) is usually gender-specific.  Title was removed -- without it, the results were more as expected -- fare class and sex are highly significant.

We want to see if age can be helpful, even if we can't use it for all passengers.  Using only passengers for whom the age is known, logistic regression on the above features plus age finds that age is highly significant, and fare class and sex remain highly significant.  VIF does not show any large dependencies between these features.

To see if deck might be useful, it was used along with the above features without age.  Here, only sex was highly significant.  So this set of features including deck was less useful than the previous two sets, with or without age.

Finally, the lowest-significance features, fare, # parents / children, port were dropped, leaving fare class, sex, # siblings / spouses, and optionally age.  With this reduced set of features, all features were highly significant.

Logistic regression does not capture non-linear relationships.  To see what features are useful to a non-linear classifier, a classification and regression tree (CART) was tried (using the rpart package).  This can handle both numerical and categorical features, and deals with missing data.  It was trained with all the features other than title.  At the root of the tree, CART used sex, then in the next two levels of the tree, age, fare class, # siblings / spouse and fare.

## Classifier selection and assessment

Several classifiers were selected to try out, that are known to perform well on irregular data, and that can handle both numerical and categorical data:  random forest (package randomForest), boosted generalized linear model (glmboost), model averaged neural network (avNNet), stochastic gradient boosting (gbm) with adaboost option, gbm with bernoulli option.  Each was assessed for two cases:  on all the training data, with all features except age (fare class, fare, port, sex, # siblings / spouse, # parents / children), and on the subset of data that contains age, using all features including age.

What we want to know is how well each classifier is expected to perform on data other than what it is trained on.  To do this, 20-fold cross-validation (CV) was run on each classifier, with and without age, and the entire CV was repeated several times.  The accuracies for each run were averaged.  This provides an estimate of how well these will perform on the test data.  Accuracies range from 0.786 (for glmboost without age) to 0.818 (for gbm / bernoulli with age).

## Voting classifier model and training

Each of the above classifiers was then re-trained on the entire training set (for features without age) and on the subset of the training data with age (and including age in the features).  No data was held out as a final out-of-sample check due to the small amount of data available -- we will need to rely on the cross-validation accuracy estimates and on the public test data score provided by Kaggle.

One way to overcome poor performance of individual classifiers is to train multiple different classifiers and combine their predictions.  For instance, one could get the yes / no survival prediction for each, and take the majority vote.

However, this does not take into account how "confident" each classifier is in its prediction, nor how accurate each is expected to be on non-training data.  To deal with both of these things, one can instead have each classifier predict the *probability* that the passenger survived, and then compute the weighted average of these probabilities, using the relative out-of-sample accuracies of each classifier that were found above by cross-validation.  (Caution!  This accuracy-weighted voting isn't known to be a standard procedure -- I made it up -- so caveat emptor...)

In addition, since age does increase the accuracy when it's available, for each passenger with age, the classifier models with age are used, and the classifier models without age are only used if the passenger's age isn't known.

Thus for each passenger, we get a net probability of survival.  Then a threshold of 0.5 is used to decide whether to predict survival (if the probability is at least 0.5) or not (below 0.5).

## Evaluation

A common baseline model for this data is one that uses only the passenger's sex, and predicts survival if the passenger is female, but not if they are male.  This model has an accuracy of 0.787 on the training data -- that's not an out-of-sample error, but that's not really relevant here, as this model has no parameters and does not require any training.  Any classifier that performs no better than this baseline is probably not useful by itself.  This model is also shown in the Kaggle leaderboard, with a score of 0.766 on the public test data.  Of the individual classifiers, only glmboost without age does not beat the simple sex-based model, but given that these classifiers are using a lot more features, they are not performing much better than the baseline.

Since there was too little data to assess the entire ensemble on a held-out part of the training data, its in-sample accuracy was checked.  On the training data, its accuracy is 0.878, which is better than any of the individual classifiers' estimated out-of-sample accuracies.  However, because this is an in-sample accuracy, it is expected to over-estimate the actual out-of-sample accuracy.  And when predictions on the test set were submitted, the ensemble classifier's accuracy dropped to 0.77.

A drop in performance from in-sample to out-of-sample suggests overfitting.  Another possible explanation would be that the training and test data differ in their distribution of feature values -- either they were not very randomly sampled, or there are rare cases in the data that are missing from the training set.  But the fact that the simple sex-based model's performance did not differ as greatly between the training and test data hints otherwise.

One reason for overfitting may be that the classifier models used were too capable of fitting irregular data.  It is common to tune model parameters using CV to find a balance between constraining the model and allowing overfitting.  The caret package does some parameter tuning internally when it trains classifiers, but no explicit parameter tuning was done outside of that.

Another possible reason for overfitting was including all of the lower-importance variables.  If a classifier has sufficient fitting capability, it can be given random noise features, and it will use them to overfit.  So withdrawing the lower-significance features is an appropriate thing to try.

Apart from overfitting, these accuracies just are not as high as those in the Kaggle leaderboard, where accuracies range up to 1.0.  Note that perfect accuracy is not expected -- that would mean the data have *no noise at all*.  To investigate noise, it might be possible to find ambiguous cases -- two passengers whose informative features (i.e. the features other than name, ticket text, or Kaggle identifier) are the same or nearly so, but who have different survival outcomes.  Since the test outcomes are not available, and we're cautioned in the Kaggle instructions not to look for the missing information, this reduces the chance of finding ambiguous examples.  Finding an ambiguous case would prove that an accuracy of 1.0 wasn't really possible.  Note that achieving 1.0 on the current leaderboard does not mean that the accuracy will be 1.0 when the private score is revealed (if it ever is).  Nevertheless, it is expected that getting 1.0 even on the public test data would require tuning to the test results.  This won't tell us much that is meaningful about why people survived or didn't.

However, there are certainly many submissions that are better than 0.77, and even above the 0.878 in-sample accuracy, that are not unreasonably high.  This indicates that overfitting is not the whole problem -- there should still be more information that can be wrung out of the data.

## Revising the solution

The best thing to do at this point would have been to attack the apparent overfitting.  Options for doing this include dropping the less important features, doing more parameter tuning beyond what caret is doing internally, using simpler classifiers.  A more interesting option would be to instead use many more classifiers in the averaging, especially weak classifiers -- this would tend to "smooth" the overfitting.  One way to do this is to resample the training data with replacement (bootstrapping) many times, train classifiers on each sample, and average the results -- this is bootstrap aggregating (bagging).  Another is to place various restrictions on the classifiers, such as giving them different subsets of the features -- this trick is used by random forest models, but could be applied to other types of classifiers.

However, it is more interesting to try to squeeze more information out of the data, especially meaningful information.  In the limited time available, there wasn't time to do both, so I'll have to come back to fixing overfitting later.

The features used in the initial model capture mainly *social* aspects of survival -- did people favor allowing women and children to be rescued? did higher-class people tend to survive?  But what is there that would give us insight into how design of the passenger accomodations affects survival?  Distance of cabins from stairwells, how far people had to go to reach lifeboat embarcation points, obstacles to egress,...  The one item of information we have that speaks to this is the cabin.  Before we start bringing in external information about deck layout, such as distance from each cabin to hallways, stairwells, lifeboats, we can attempt to find out if there is any effect of cabin on survival.

The names of cabins are almost unique to each passenger, and the cabin numbers are not informative in themselves -- there isn't a mathematical relationship between the number and the evacuation difficulty.  But there are two things we can check:

Which deck the cabin is on may relate to how long and how difficult it was to get to the lifeboat deck.  The deck letter can
be extracted from the cabin name.  We've already examined the deck to some extent, and it did not seem very predictive.  This may be because it is related to the fare class -- higher fare classes are on higher decks -- and we have the fare class as a feature.

But there's something that will give us a more more direct check on whether the cabin may be important:  We can look at the passengers who shared a cabin (or suite of cabins) -- did they have the same survival outcome at a higher rate than chance?  If so, that indicates there was an effect of cabin location.

There were 38 pairs of passengers sharing cabins or suites of adjacent cabins.  Of these, 27 had the same survival outcome, and 11 did not, for an accuracy of 0.711.  What should this be compared against?  What we want to know is, what is the probability that a random pair of passengers have the same survival outcome.  That's the same computation as flipping a biased coin twice -- the probability of two heads (HH) or two tails (TT), divided by the probabilty of all options (HH, TT, HT, TH).  Given the individual survival probablity of 0.384 in the training data, the probability of two random passengers' outcomes being the same is 0.527.  The measured rate of 0.711 when the cabin is the same is a lot higher, which indicates that the cabin location may be significant and worth looking into further.

But we can also try to use this information directly, to improve prediction, in the cases where a passenger in the test set shared a cabin with a passenger in the training set.  There are 46 passengers in the test set who share a cabin or cabin suite with one or more passengers in the training set.  For each of these, a prediction was made if the training set cabin-mates all had the same outcome.  That is, if there were two cabin-mates in the training set, and one survived and the other didn't, that gives us no information.  Most of the 46 had only one cabin-mate, so their predicted outcome was unambiguous.  For the unambiguous cases, the additional prediction from cabin-mate survival was treated as another classifier to be included in voting.  Its accuracy was taken to be the 0.711 found above, and the probability was taken to be 1 or 0, depending on cabin-mate survival.  For each of these, the weighted average probability was adjusted, and the threshold applied again.

Out of the 46 cases, only 2 changed predictions.  And when this was submitted to Kaggle, the public score remained the same, meaning those two cases were in the private subset of the test data.

At this point, the decision was made to get this written and submitted, and the code posted on GitHub.  Since I don't like leaving my entry so close to the baseline in the leaderboard, I'll likely plug away at it as time is available.
