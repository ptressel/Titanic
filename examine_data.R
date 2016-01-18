# Basic examination of Titanic data and imputation of missing features.

# Libraries
require(caret, quietly=TRUE)
require(car, quietly=TRUE)
require(rpart, quietly=TRUE)

# Load data
train.colClasses <- c(PassengerId="integer",
                      Survived="integer",
                      Pclass="integer",
                      Name="character",
                      Sex="factor",
                      Age="numeric",
                      SibSp="integer",
                      Parch="integer",
                      Ticket="character",
                      Fare="numeric",
                      Cabin="character",
                      Embarked="factor")
test.colClasses <-  c(PassengerId="integer",
                      Pclass="integer",
                      Name="character",
                      Sex="factor",
                      Age="numeric",
                      SibSp="integer",
                      Parch="integer",
                      Ticket="character",
                      Fare="numeric",
                      Cabin="character",
                      Embarked="factor")
na.strings <- c("", "NA")

# Cannot use "train" as the name of the dataset, as it will collide with the
# caret train() function.  This in spite of R supposedly having separate
# namespaces for functions and variables.  This happens when train is run via
# do.call.  Change the name of test for consistency.  Refuse to change anything
# else.
#train.set <- read.csv("train.csv", stringsAsFactors=FALSE)
#test.set <- read.csv("test.csv", stringsAsFactors=FALSE)
train.set <- read.csv("train.csv", colClasses=train.colClasses, na.strings=na.strings)
test.set <- read.csv("test.csv", colClasses=test.colClasses, na.strings=na.strings)
train.colnames <- colnames(train.set)
test.colnames <- colnames(test.set)

# Extract "hidden" info from some columns.

# Deck -- first letter of the cabin.
# Before we can get this, some cleanup of the Cabin values is needed.

# There are Cabin values that are strings of multiple cabins.  These are
# probably nearby groups of cabins booked by parties.
long.cabin.names <-
    alldata$Cabin[!is.na(alldata$Cabin) & nchar(alldata$Cabin) > 4]
#  [1] "C23 C25 C27"     "F G73"           "C23 C25 C27"     "D10 D12"
#  [5] "B58 B60"         "F E69"           "C22 C26"         "B58 B60"
#  [9] "C22 C26"         "B57 B59 B63 B66" "C23 C25 C27"     "B96 B98"
# [13] "B96 B98"         "C23 C25 C27"     "C22 C26"         "B51 B53 B55"
# [17] "F G63"           "C62 C64"         "F G73"           "B57 B59 B63 B66"
# [21] "B96 B98"         "B82 B84"         "B96 B98"         "B51 B53 B55"
# [25] "B57 B59 B63 B66" "C23 C25 C27"     "F G63"           "B57 B59 B63 B66"
# [29] "C23 C25 C27"     "C55 C57"         "C55 C57"         "B57 B59 B63 B66"
# [33] "B58 B60"         "C62 C64"         "F E46"           "C22 C26"
# [37] "F E57"           "B51 B53 B55"     "D10 D12"         "E39 E41"
# [41] "B52 B54 B56"
# > length(long.cabin.names)
# [1] 41
# > length(unique(long.cabin.names))
# [1] 17
cabins.in.sets <- unique(unlist(sapply(long.cabin.names, function(c) {
    strsplit(c, split=" ", fixed=TRUE)
}, USE.NAMES=FALSE)))

# Are there cabins that appear alone (not in a set) that also appear in sets?
# As it turns out, there are not, but this turns up an odd feature in some
# cabin names.
single.cabins.appearing.in.sets <-
    unique(alldata$Cabin[!is.na(alldata$Cabin) &
                          alldata$Cabin %in% cabins.in.sets])
# > length(single.cabins.appearing.in.sets)
# [1] 2
# > single.cabins.appearing.in.sets
# [1] "E46" "F"

# How many passengers have cabin "F" and what does that mean?
# > alldata[grep("^F$|F | F$", alldata$Cabin, value=FALSE), c("PassengerId", "Pclass", "Cabin", "Name")]
#      PassengerId Pclass Cabin                                       Name
# 76            76      3 F G73                    Moen, Mr. Sigurd Hansen
# 129          129      3 F E69                          Peter, Miss. Anna
# 700          700      3 F G63   Humblen, Mr. Adolf Mathias Nicolai Olsen
# 716          716      3 F G73 Soholt, Mr. Peter Andreas Lauritz Andersen
# 949          949      3 F G63              Abelseth, Mr. Olaus Jorgensen
# 1001        1001      2     F                          Swane, Mr. George
# 1180        1180      3 F E46                    Mardirosian, Mr. Sarkis
# 1213        1213      3 F E57                      Krekorian, Mr. Neshan
# With the exception of the one passenger with F alone, the others have
# Pclass 3.  One might guess that these are assistants to the other passengers
# in the group.
# Is the E46 an error?  Check if Pclass matches, or if there is similarity in
# Name.  There are two records with E46 alone.  Again, note the Pclass 3
# passenger with F -- here paired with two Pclass 1 passengers.
rows.with.E46 <- grep("E46", alldata$Cabin, value=FALSE, fixed=TRUE)
# [1]    7 1038 1180
# > alldata[rows.with.E46, c("PassengerId", "Pclass", "Cabin", "Name")]
#      PassengerId Pclass Cabin                        Name
# 7              7      1   E46     McCarthy, Mr. Timothy J
# 1038        1038      1   E46 Hilliard, Mr. Herbert Henry
# 1180        1180      3 F E46     Mardirosian, Mr. Sarkis

# Go back, remove isolated F from cabin names.
train.set$Cabin[!is.na(train.set$Cabin) & train.set$Cabin=="F"] <- NA
test.set$Cabin[!is.na(test.set$Cabin) & test.set$Cabin=="F"] <- NA
alldata$Cabin[!is.na(alldata$Cabin) & alldata$Cabin=="F"] <- NA
train.set$Cabin <- gsub("^F ", "", train.set$Cabin)
train.set$Cabin <- gsub("F ", " ", train.set$Cabin, fixed=TRUE)
test.set$Cabin <- gsub("^F ", "", test.set$Cabin)
test.set$Cabin <- gsub("F ", " ", test.set$Cabin, fixed=TRUE)
alldata$Cabin <- gsub("^F ", "", alldata$Cabin)
alldata$Cabin <- gsub("F ", " ", alldata$Cabin, fixed=TRUE)
# Next we would replace the single cabins that appear in sets of cabins with
# the set but, as it turns out, there are none -- the only single cabin that
# appeared in a set was E46, and that appeared in "F E46" only.  All the cabin
# sets are consistent -- none overlap, nor are there differing partial sets.

# There is also a single cabin with value "T" in the training set.  There is
# no deck T.  Set that to NA.  Run the same on the test set for consistency
# and in case the train and test sets are modified at some point.
train.set$Cabin[!is.na(train.set$Cabin) & substr(train.set$Cabin,1,1)=="T"] <- NA
test.set$Cabin[!is.na(test.set$Cabin) & substr(test.set$Cabin,1,1)=="T"] <- NA

# Once the F is removed, all cabins appearing in a set have the same deck, and
# the deck is always the first character, whether it's a single cabin or a set.
# So now we can extract the deck.
train.set$Deck <- sapply(train.set$Cabin, function(s) substr(s, 1, 1))
test.set$Deck <- sapply(test.set$Cabin, function(s) substr(s, 1, 1))

# These can be factors, though it might be appropriate to try them as ordered,
# given that they indicate depth within the hull, and thus distance from the
# boat deck where the lifeboats were.
decks <- sort(unique(c(train.set$Deck, test.set$Deck)))
train.set$Deck <- factor(train.set$Deck, levels=decks)
test.set$Deck <- factor(test.set$Deck, levels=decks)

# All names come with a title.  The test set has one title, "Dona", that is
# not present in the train set, though "Don" is.
train.set$Title <- sapply(train.set$Name,
                      function(s) sub("\\. .*$", "", sub("^.*, ", "", s)))
test.set$Title <- sapply(test.set$Name,
                     function(s) sub("\\. .*$", "", sub("^.*, ", "", s)))
titles <- sort(unique(c(train.set$Title, test.set$Title)))
train.set$Title <- factor(train.set$Title, levels=titles)
test.set$Title <- factor(test.set$Title, levels=titles)
# There is one title that appears only in the test data, "Dona".  Since the
# purpose of extracting the titles is to get an indicator of status, assume that
# the male parallel, "Don", is equivalent.  Since gender is also available, we
# don't need to get that from the title, so substituting the male title should
# not, we hope, be an issue.
test.set$Title[test.set$Title == "Dona"] <- "Don"

# Now that we've gotten the "easy" information out of Name and Cabin, exclude
# columns that I don't expect may be useful (without a lot of situation-specific
# investigation, which has little to do with machine learning).  PassengerId is
# only a Kaggle identifier, so is irrelevant.  Ticket contains undescribed
# annotations and an undescribed number.  Assume for now that any relevant
# information is captured in other features, such as Fare and Pclass.  Other
# than the deck, which has already been extracted, Cabin is not likely to be
# useful without mapping cabin identifiers to aspects of the ship's deck layout,
# such as proximity to hallways / stairwells / exits / lifeboats.  Skip this
# for now.
basic.colnames <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")

# Would like to see if the train and test sets are similar in their
# distributions of samples.  Also want to see how different the imputation
# results are if they are based on the train set alone, or all data.  So
# form a data set with the features from both train and test.
alldata <- rbind(train.set[, test.colnames], test.set)

# Capture the summary info per column.  Is it usable?
# The data.frame summary info is not in a convenient form.
#train.summary <- summary(train.set)
#test.summary <- summary(test.set)
# Nor is the summary info for each column -- it is just formatted text.
get.summary <- function(colname, data) {
    summary(data[colname])
}
train.summary <- lapply(train.colnames, get.summary, train.set)
names(train.summary) <- train.colnames
test.summary <- lapply(test.colnames, get.summary, test.set)
names(test.summary) <- test.colnames
alldata.summary <- lapply(test.colnames, get.summary, alldata)
names(alldata.summary) <- test.colnames

# Which features have NAs and how many are there?  The prevalence of NAs
# will help decide whether to impute them (if there are only a few) or have
# separate classifiers for subsets of features excluding the NA features (if
# there are a lot).
number.nas.in.col <- function(colname, d) {
    sum(is.na(d[colname]))
}
train.number.nas <- sapply(train.colnames, number.nas.in.col, train.set)
names(train.number.nas) <- train.colnames
test.number.nas <- sapply(test.colnames, number.nas.in.col, test.set)
names(test.number.nas) <- test.colnames

# Most of the NAs are in Age and Cabin (and hence Deck).  There is a single
# NA in test for Fare, and two NAs in train for Embarked.  Since there are so
# few NAs in Fare and Embarked, they will be imputed.

# For Fare, since it may depend on both the Embarked port and the Pclass, use
# the median within the subset that matches the record with the missing Fare.
test.missing.fare.row <- which(is.na(test.set$Fare))
train.fare.median <- tapply(train.set$Fare,
                            list(train.set$Embarked, train.set$Pclass),
                            median)
test.set$Fare[test.missing.fare.row] <-
    train.fare.median[test.set$Embarked[test.missing.fare.row],
                      test.set$Pclass[test.missing.fare.row]]
# How different is the result if we use only the train data vs. all data?
# Here, we have to avoid the NAs.  Result is the same.
alldata.with.fare <- !is.na(alldata$Fare)
alldata.fare.median <-
    tapply(alldata[alldata.with.fare, "Fare"],
           list(alldata[alldata.with.fare, "Embarked"],
                alldata[alldata.with.fare, "Pclass"]),
           median)
# Fill this in for alldata as well.
alldata$Fare[which(is.na(alldata$Fare))] <- test.set$Fare[test.missing.fare.row]

# For the Embarked port that is missing in two train rows, let's use the Fare
# and Pclass.  Here, it's not obvious how to select the "most probable" value,
# because Fare is continuous not discrete, so we can't just select out the
# Embarked values for the Fare and Pclass of the two rows.
train.missing.embarked.rows <- which(is.na(train.set$Embarked))
# As it turns out, both of these passengers were in the same cabin, paid the
# same fare, 80, and have the same pclass, 1.  So get the fares for that
# pclass, split them by port, and histogram them.  Treat each histogram as a
# distribution, and for each Embarked value, see what probability would be
# assigned to the fare for these rows.  Since there's only one fare we're
# interested in, can simplify that -- choose a width around the target fare
# and get counts within that for the two cases, vs. counts without that.
# Choose whichever port has higher relative counts around the target fare.
get.fares.per.embarked <- function(embarked, data) {
    data[!is.na(data[, "Embarked"]) & data[, "Embarked"] == embarked, "Fare"]
}
train.fare.pclass.1 <- lapply(levels(train.set$Embarked),
                              get.fares.per.embarked,
                              train.set[train.set$Pclass == 1,])
names(train.fare.pclass.1) <- levels(train.set$Embarked)
alldata.fare.pclass.1 <- lapply(levels(alldata$Embarked),
                                get.fares.per.embarked,
                                alldata[alldata$Pclass == 1,])
names(alldata.fare.pclass.1) <- levels(alldata$Embarked)
# There is only a single Fare value for Embarked Q within Pclass 1, and it is
# not the same as that of the rows with missing Embarked.  In addition, only
# three rows with Pclass = 1 even have Embarked = Q.  So that isn't likely to
# be the correct Embarked for these two rows.  For the remaining two options,
# get the counts of fares near the target fare 80.  Fares range from 0 to 513
# and there are enough to have a bin size of 20.  So count fares from 70-90
# for the two cases.  See if the result differs between the train data and
# all data.  No, both yield the same result -- C has the higher probability.
train.fare.near.80.C <-
    sum(train.fare.pclass.1$C >= 70 & train.fare.pclass.1$C <= 90) /
    length(train.fare.pclass.1$C)
train.fare.near.80.S <-
    sum(train.fare.pclass.1$S >= 70 & train.fare.pclass.1$S <= 90) /
    length(train.fare.pclass.1$S)
alldata.fare.near.80.C <-
    sum(alldata.fare.pclass.1$C >= 70 & alldata.fare.pclass.1$C <= 90) /
    length(alldata.fare.pclass.1$C)
alldata.fare.near.80.S <-
    sum(alldata.fare.pclass.1$S >= 70 & alldata.fare.pclass.1$S <= 90) /
    length(alldata.fare.pclass.1$S)
# Assign the one with the higher probability.
train.set[train.missing.embarked.rows, "Embarked"] <-
    if (train.fare.near.80.C >= train.fare.near.80.S) "C" else "S"
# These are the same rows in alldata.
alldata[train.missing.embarked.rows, "Embarked"] <-
    if (train.fare.near.80.C >= train.fare.near.80.S) "C" else "S"

# Note this section isn't needed -- all of the classifiers will convert factors
# to indicators by themselves.
# Convert factors to indicators.  To be sure we get columns for all values,
# do this on alldata, then split that back into train and test.
factor.columns <- c("Sex", "Embarked", "Deck")
alldata.dummies.model <- dummyVars(~ ., alldata[, factor.columns], fullRank=TRUE)
alldata.dummies <- predict(alldata.dummies.model, alldata, na.action=na.pass)
# Names of the new columns.
dummies.colnames <- colnames(alldata.dummies)
# Some columns are effectively factor, but are represented by integers.
# It isn't clear whether Pclass would be better treated as numerical or as a
# factor.  If the effect of Pclass is highly non-linear, then treating it as
# a factor would be appropriate.
numerical.factors <- c("SibSp", "Parch", dummies.colnames)
# Add those columns onto the other data.
alldata <- cbind(alldata, alldata.dummies)
train.set <- cbind(train.set, alldata.dummies[1:nrow(train.set),])
test.set <- cbind(test.set, alldata.dummies[(nrow(train.set)+1):nrow(alldata),])

# Stats on single columns.  For numerical data, get min, max, mean, quartiles.
# For factor data, get a table.  Although summary produces all or most of this,
# it's not in a convenient form -- it's text for display.
column.stats <- function(colname, data) {
    col <- data[colname][[1]]
    # typeof does not return "factor" for factor variables.
    type <- typeof(col)
    if (is.factor(col) | colname %in% numerical.factors) {
        # Get a frequency table.
        tab <- table(data[colname])
        # These should be treated as categorical even if they are not R factors.
        col.stats <- list(type="factor", table=tab)
    } else if (is.numeric(col)) {
        # Get summary statistics.
        col.stats <- list(type=type,
                          mean=mean(col), sd=sd(col),
                          quartiles=quantile(col, na.rm=TRUE))
    } else {
        # For character or other, just report the type.
        col.stats <- list(type=type)
    }
    col.stats
}
train.column.stats <- lapply(c(test.colnames, dummies.colnames),
                             column.stats, train.set)
names(train.column.stats) <- c(test.colnames, dummies.colnames)
alldata.column.stats <- lapply(c(test.colnames, dummies.colnames),
                               column.stats, alldata)
names(alldata.column.stats) <- c(test.colnames, dummies.colnames)

# Near zero variance.  This is a formality -- the features clearly won't fail
# this check.  This is only supported for numerical columns.  As expected, none
# fail.
basic.numerical <- c("Pclass", "Age", "SibSp", "Parch", "Fare")
train.nzv <- nearZeroVar(train.set[, basic.numerical])

# Correlation with outcome.  Again, exclude the one factor variable among the
# basic features.
train.cor.with.survived <- cor(train.set[, "Survived"], train.set[, basic.numerical],
                               use="pairwise.complete.obs")
# > train.cor.with.survived
#         Pclass         Age      SibSp      Parch      Fare
# [1,] -0.338481 -0.07722109 -0.0353225 0.08162941 0.2573065

# Histogram Age separately vs. Survived.
# This is file age_vs_survived.png
age.vs.survived.hist <- ggplot(data=train.set, aes(x=Age)) +
    geom_histogram(binwidth=1) + facet_grid(~ Survived)
print(age.vs.survived.hist)

# Again, with Pclass as well for faceting.
# age_vs_survived_pclass.png
age.vs.survived.pclass.hist <- ggplot(data=train.set, aes(x=Age)) +
    geom_histogram(binwidth=1) + facet_grid(Pclass ~ Survived)
print(age.vs.survived.pclass.hist)

# Had better get histograms vs Pclass overall before jumping to conclusions.
# age_vs_pclass.png
age.vs.pclass.hist <- ggplot(data=train.set, aes(x=Age)) +
    geom_histogram(binwidth=1) + facet_grid(Pclass ~ .)
print(age.vs.pclass.hist)

# Check for correlated features.  No pairs are so highly correlated that they
# should be excluded or merged up front.
train.basic.cor <- cor(train.set[, basic.numerical], use="pairwise.complete.obs")
# > train.basic.cor
#             Pclass         Age       SibSp       Parch        Fare
# Pclass  1.00000000 -0.36922602  0.08308136  0.01844267 -0.54949962
# Age    -0.36922602  1.00000000 -0.30824676 -0.18911926  0.09606669
# SibSp   0.08308136 -0.30824676  1.00000000  0.41483770  0.15965104
# Parch   0.01844267 -0.18911926  0.41483770  1.00000000  0.21622494
# Fare   -0.54949962  0.09606669  0.15965104  0.21622494  1.00000000

# Try logistic regression, mainly to get its take on variable inflation factor
# (vif) and feature significance (p-values).  Try several cases, in order to
# cover all the variables without imputing anything.  To be clear:  The purpose
# of this model is to gather information for feature selection, not (yet) to
# build a classifier.

# Only variables with no NAs:
features.no.nas <-
    c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Title")
formula.no.nas <-
    as.formula(paste("Survived ~ ", paste(features.no.nas, collapse="+")))
glm.no.nas <- glm(formula.no.nas, family=binomial, data=train.set)
print(glm.no.nas)
summary(glm.no.nas)
# Title is not significant.  Only Pclass and SibSp are really significant, with
# minor significance for Parch and Embarked S.  What's strange is that Sex is
# not significant, or rather, Sex male.  But nor is the intercept significant.
# Surely gender was the most important predictor, so something is odd.
vif(glm.no.nas)
# The VIFs for Title and Sex are both high before correction for degrees of
# freedom.  Remove Title.
features.no.nas.no.title <-
    c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked")
formula.no.nas.no.title <-
    as.formula(paste("Survived ~ ", paste(features.no.nas.no.title, collapse="+")))
glm.no.nas.no.title <-
    glm(formula.no.nas.no.title, family=binomial, data=train.set)
print(glm.no.nas.no.title)
summary(glm.no.nas.no.title)
vif(glm.no.nas.no.title)
# Without Title, the results are more as expected.  Intercept, Pclass, Sex are
# highly significant, SibSp and Embarked S are slightly significant.  Nothing
# is up near a p-value of 1, so even the non-significant variables might have
# useful info.  So much for that experiment with Title...exclude it from now on.

# How many rows have Age, how many have Cabin / Deck, how many have both?
sum(!is.na(train.set$Age))
sum(!is.na(train.set$Cabin))
sum(!is.na(train.set$Age) & !is.na(train.set$Cabin))
# 714 / 891 rows have Age.  Only 204 / 891 rows have Cabin.  And only
# 185 / 714 rows have both Age and Cabin.  That's rather a low count to expect
# to get a successful classifier out of it.

# Next add Age, keeping only the rows that have it.
features.age.complete.cases <-
    c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age")
formula.age.complete.cases <-
    as.formula(paste("Survived ~ ", paste(features.age.complete.cases, collapse="+")))
glm.age.complete.cases <-
    glm(formula.age.complete.cases, family=binomial,
        data=train.set[!is.na(train.set$Age), ])
print(glm.age.complete.cases)
summary(glm.age.complete.cases)
vif(glm.age.complete.cases)
# Age is highly significant.  Intercept, Pclass, Sex remain highly significant
# and SibSp moves up in significance.  VIFs are still on a par.

# Since Cabin is only in common for a small number of passengers, it won't have
# much predictive value.  But try Deck -- look only at rows that have it.  Omit
# Age -- want to see what Deck does alone.
features.deck.complete.cases <-
    c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Deck")
formula.deck.complete.cases <-
    as.formula(paste("Survived ~ ", paste(features.deck.complete.cases, collapse="+")))
glm.deck.complete.cases <-
    glm(formula.deck.complete.cases, family=binomial,
        data=train.set[!is.na(train.set$Deck), ])
print(glm.deck.complete.cases)
summary(glm.deck.complete.cases)
vif(glm.deck.complete.cases)
# Only Sex is highly significant now, and the intercept and deck G are
# significant at lower levels.  Deck has the highest VIF.  So the previous
# set, without Deck and with Age, was better.

# Since Fare, Parch, and Embarked are not significant, try without them.
# Try this set with Age.
features.signif.with.age.complete.cases <-
    c("Pclass", "Sex", "SibSp", "Age")
formula.signif.with.age.complete.cases <-
    as.formula(paste("Survived ~ ",
                     paste(features.signif.with.age.complete.cases, collapse="+")))
glm.signif.with.age.complete.cases <-
    glm(formula.signif.with.age.complete.cases, family=binomial,
        data=train.set[!is.na(train.set$Age), ])
print(glm.signif.with.age.complete.cases)
summary(glm.signif.with.age.complete.cases)
vif(glm.signif.with.age.complete.cases)
# Every feature, even SibSp is significant now.
# And without Age.
features.signif.no.age.complete.cases <-
    c("Pclass", "Sex", "SibSp")
formula.signif.no.age.complete.cases <-
    as.formula(paste("Survived ~ ",
                     paste(features.signif.no.age.complete.cases, collapse="+")))
glm.signif.no.age.complete.cases <-
    glm(formula.signif.no.age.complete.cases, family=binomial,
        data=train.set[!is.na(train.set$Age), ])
print(glm.signif.no.age.complete.cases)
summary(glm.signif.no.age.complete.cases)
vif(glm.signif.no.age.complete.cases)
# Without Age, SibSp loses significance.

# At this point, the feature set regarded as most significant by glm is:
# Pclass, Sex, SibSp, Age

# Next see which features CART uses.  Start over at the top, with all features.
# CART is less sensitive to which features are included to begin with.
# Include all the original predictors; don't bother with Title and Deck.
# (This was tried with Deck as well, but rpart did not use it at all.)
# rpart handles NAs, so just let it do what it wants.
features.all <-
    c("Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age")
formula.all <-
    as.formula(paste("Survived ~ ",
                     paste(features.all, collapse="+")))
rpart.all <- rpart(formula.all, data=train.set, method="class")
print(rpart.all)
# node), split, n, loss, yval, (yprob)
#       * denotes terminal node
#   1) root 891 342 0 (0.61616162 0.38383838)
#     2) Sex=male 577 109 0 (0.81109185 0.18890815)
#       4) Age>=6.5 553  93 0 (0.83182640 0.16817360) *
#       5) Age< 6.5 24   8 1 (0.33333333 0.66666667)
#        10) SibSp>=2.5 9   1 0 (0.88888889 0.11111111) *
#        11) SibSp< 2.5 15   0 1 (0.00000000 1.00000000) *
#     3) Sex=female 314  81 1 (0.25796178 0.74203822)
#       6) Pclass>=2.5 144  72 0 (0.50000000 0.50000000)
#        12) Fare>=23.35 27   3 0 (0.88888889 0.11111111) *
#        13) Fare< 23.35 117  48 1 (0.41025641 0.58974359)
#          26) Embarked=S 63  31 0 (0.50793651 0.49206349)
#            52) Fare< 10.825 37  15 0 (0.59459459 0.40540541) *
#            53) Fare>=10.825 26  10 1 (0.38461538 0.61538462)
#             106) Fare>=17.6 10   3 0 (0.70000000 0.30000000) *
#             107) Fare< 17.6 16   3 1 (0.18750000 0.81250000) *
#          27) Embarked=C,Q 54  16 1 (0.29629630 0.70370370) *
#       7) Pclass< 2.5 170   9 1 (0.05294118 0.94705882) *
summary(rpart.all)
# Variable importance
#     Sex     Fare   Pclass    SibSp    Parch      Age Embarked
#      47       18       13        7        6        5        4
varImp(rpart.all)
#            Overall
# Age       19.05955
# Embarked  29.37189
# Fare      77.60257
# Parch     22.34053
# Pclass    88.77362
# Sex      124.42633
# SibSp     23.95477
# rpart uses all the features, but assigns a different importance than glm.
# It prefers Fare to Pclass, where glm preferred Pclass.  This means Fare may
# require non-linear treatment, as rpart can insert splitting points wherever
# it wants into Fare.  But caret's varImp assigns variable importance in a
# different order.  Again, Fare is more useful than it was to glm.

# Note it used both Fare and Pclass.

# Some options for imputing Age:
# Split the data by the other features that are somewhat correlated with Age
# (Pclass, SibSp, Parch) and use the median age within that.
# Use regression to predict Age based on the other features.

# First, get the median age by Pclass, SibSp, Parch.
age.by.pclass.sibsp.parch <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$SibSp, alldata$Parch),
           median, na.rm=TRUE)
age.by.pclass.sibsp <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$SibSp),
           median, na.rm=TRUE)
age.by.pclass.parch <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$Parch),
           median, na.rm=TRUE)
age.by.sibsp.parch <-
    tapply(alldata$Age, list(alldata$SibSp, alldata$Parch),
           median, na.rm=TRUE)
age.by.pclass <-
    tapply(alldata$Age, list(alldata$Pclass),
           median, na.rm=TRUE)
age.by.sibsp <-
    tapply(alldata$Age, list(alldata$SibSp),
           median, na.rm=TRUE)
age.by.parch <-
    tapply(alldata$Age, list(alldata$Parch),
           median, na.rm=TRUE)
age.by.none <- median(alldata$Age, na.rm=TRUE)

# How many samples are each of those based on?
number.non.nas <- function(d) {
    sum(!is.na(d))
}
age.num.by.pclass.sibsp.parch <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$SibSp, alldata$Parch),
           number.non.nas)
age.num.by.pclass.sibsp <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$SibSp),
           number.non.nas)
age.num.by.pclass.parch <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$Parch),
           number.non.nas)
age.num.by.sibsp.parch <-
    tapply(alldata$Age, list(alldata$SibSp, alldata$Parch),
           number.non.nas)
age.num.by.pclass <-
    tapply(alldata$Age, list(alldata$Pclass),
           number.non.nas)
age.num.by.sibsp <-
    tapply(alldata$Age, list(alldata$SibSp),
           number.non.nas)
age.num.by.parch <-
    tapply(alldata$Age, list(alldata$Parch),
           number.non.nas)
age.num.by.none <- number.non.nas(alldata$Age)

# And how much variation is there within each subset?  Use range rather than
# sd, to go along with median.
age.range.by.pclass.sibsp.parch <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$SibSp, alldata$Parch),
           range, na.rm=TRUE)
age.range.by.pclass.sibsp <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$SibSp),
           range, na.rm=TRUE)
age.range.by.pclass.parch <-
    tapply(alldata$Age, list(alldata$Pclass, alldata$Parch),
           range, na.rm=TRUE)
age.range.by.sibsp.parch <-
    tapply(alldata$Age, list(alldata$SibSp, alldata$Parch),
           range, na.rm=TRUE)
age.range.by.pclass <-
    tapply(alldata$Age, list(alldata$Pclass),
           range, na.rm=TRUE)
age.range.by.sibsp <-
    tapply(alldata$Age, list(alldata$SibSp),
           range, na.rm=TRUE)
age.range.by.parch <-
    tapply(alldata$Age, list(alldata$Parch),
           range, na.rm=TRUE)
age.range.by.none <- range(alldata$Age, na.rm=TRUE)

# Redo the above, putting everything into a data frame, with only the sets of
# Pclass, SibSp, Parch that actually occur in the data.  Use this to construct
# the values we'll impute for missing Age values.  First get those unique sets
# of features.
age.by.pclass.sibsp.parch.df <- unique(
    alldata[, c("Pclass", "SibSp", "Parch")]
)
# For each, get the median Age, number of samples that's based on, and the
# range of ages.
# median.by.pclass.sibsp.parch <- function(pclass, sibsp, parch) {
#     median(alldata[alldata$Pclass==pclass &
#                    alldata$SibSp==sibsp &
#                    alldata$Parch==parch, "Age"], na.rm=TRUE)
# }
fun.on.age.by.pclass.sibsp.parch <- function(pclass, sibsp, parch, fname, dname, ...) {
    # fname is which aggregation function to apply
    # dname is what the data argument is called
    # The aggregation function must have just the one data argument and any
    # additional arguments can be passed as ...
    args <- list(alldata[alldata$Pclass==pclass &
                         alldata$SibSp==sibsp &
                         alldata$Parch==parch, "Age"])
    names(args) <- c(dname)
    args <- c(args, ...)
    do.call(fname, args)
}
# age.by.pclass.sibsp.parch.df$Age <-
#     mapply(median.by.pclass.sibsp.parch,
#            age.by.pclass.sibsp.parch.df$Pclass,
#            age.by.pclass.sibsp.parch.df$SibSp,
#            age.by.pclass.sibsp.parch.df$Parch)
age.by.pclass.sibsp.parch.df$Age <-
    mapply(fun.on.age.by.pclass.sibsp.parch,
           age.by.pclass.sibsp.parch.df$Pclass,
           age.by.pclass.sibsp.parch.df$SibSp,
           age.by.pclass.sibsp.parch.df$Parch,
           MoreArgs=list(fname="median", dname="x", na.rm=TRUE))
age.by.pclass.sibsp.parch.df$NumRows <-
    mapply(fun.on.age.by.pclass.sibsp.parch,
           age.by.pclass.sibsp.parch.df$Pclass,
           age.by.pclass.sibsp.parch.df$SibSp,
           age.by.pclass.sibsp.parch.df$Parch,
           MoreArgs=list(fname="number.non.nas", dname="d"))
span <- function(d) {
    # Compute the length of the range of the data d, max - min, or NA if no data.
    if (length(d)==0) return(NA)
    r <- suppressWarnings(range(d, na.rm=TRUE))
    if (r[1]==Inf) return(NA)
    r[2] - r[1]
}
age.by.pclass.sibsp.parch.df$Span <-
    mapply(fun.on.age.by.pclass.sibsp.parch,
           age.by.pclass.sibsp.parch.df$Pclass,
           age.by.pclass.sibsp.parch.df$SibSp,
           age.by.pclass.sibsp.parch.df$Parch,
           MoreArgs=list(fname="span", dname="d"))
# After seeing both how few ages there are for some subsets, and that the
# subsets with enough samples have very broad age ranges, it seems imputing
# the age, by this means at least, won't be meaningful.

# Do passengers in the same cabin have the same outcome, at a higher rate than
# in different cabins?  First, how many share cabins?  Check this in the whole
# data and just in the training data, as the latter will determine whether we
# have enough info to check this hypothesis.
shared.cabins <- table(c(train.set$Cabin, test.set$Cabin))
shared.cabins.breaks <- seq(from=0.5, to=6.5, by=1)
shared.cabins.hist <- hist(shared.cabins, breaks=shared.cabins.breaks)
# Same, just in train.
shared.cabins.train <- table(train.set$Cabin)
shared.cabins.train.hist <- hist(shared.cabins.train, breaks=shared.cabins.breaks)
# There are 38 people sharing cabins by pairs -- look just at those.
shared.cabins.train.groups <- split(train.set[, c("Survived", "PassengerId")],
                                    train.set$Cabin)
# Separate these by how many are in each group.
shared.cabins.train.count <- sapply(shared.cabins.train.groups, nrow)
shared.cabins.train.pairs <- shared.cabins.train.count == 2
# Match the Survived value in those pairs.
shared.cabins.train.pairs.match <-
    sapply(shared.cabins.train.groups[shared.cabins.train.pairs],
           function(d) {
               # Each item we get is a data frame with two rows.
               # Compare the Survived values.
               d$Survived[1] == d$Survived[2]
           })
shared.cabins.train.pairs.match.count <- table(shared.cabins.train.pairs.match)
# > table(shared.cabins.train.pairs.match)
# shared.cabins.train.pairs.match
# FALSE  TRUE
#    11    27
p.shared.cabins.match <-
    shared.cabins.train.pairs.match.count["TRUE"] /
    (shared.cabins.train.pairs.match.count["TRUE"] +
     shared.cabins.train.pairs.match.count["FALSE"])
# > p.shared.cabins.match
#      TRUE
# 0.7105263
# That's more than a 2-1 ratio.  Find out if the skew is more than due to the
# skew in the ratio of survivors to non-survivors.  What's the probability that
# a random pair of passengers survive?
n.survived <- table(train.set$Survived)
p.survived <- n.survived / (sum(n.survived))
# > p.survived
#         0         1
# 0.6161616 0.3838384
# This is like flipping an unfair coin twice -- want the probability of HH or
# TT, versus HT or TH.
p.survived.match <-
    (p.survived["0"] * p.survived["0"] + p.survived["1"] * p.survived["1"]) /
    (p.survived["0"] * p.survived["0"] + p.survived["1"] * p.survived["1"] +
     2 * p.survived["0"] * p.survived["1"])
# > p.survived.match
# 0.526987

# How many passengers in the test set have cabin-mates in the training set?
train.cabins <- train.set$Cabin[!is.na(train.set$Cabin)]
train.cabins.unique <- unique(train.set$Cabin)
test.cabin.in.train <-
    sapply(test.set$Cabin, function (c) {
        !is.na(c) & c %in% train.cabins.unique
    })
# > sum(test.cabin.in.train)
# [1] 46
# So this could potentially affect about 11% of the predictions.
# > sum(test.cabin.in.train) / nrow(test.set)
# [1] 0.1100478
# Check whether any test row matches more than one train rows.  Almost no
# attempt is made to optimize this.  For each cabin in test (that has any match
# in train), count how many of that cabin there are in train.  If any matches
# more than one, that result might be ambiguous.
test.cabin.in.train.values <- test.set$Cabin[test.cabin.in.train]
test.cabin.in.train.count <-
    sapply(test.cabin.in.train.values,
           function (c) {
               sum(train.cabins == c)
           })
# > max(test.cabin.in.train.count)
# [1] 4

# Save the cleaned up data.  Given train.set and test.set we can re-assemble
# alldata.
saveRDS(train.set, "train_set.Rds")
saveRDS(test.set, "test_set.Rds")
# Save the accuracy of using a cabin-mate's survival.  This is for pairs of
# cabin mates only, but there are only a few instances with more than two in
# a cabin.  Yes, this is saving a single variable.
saveRDS(p.shared.cabins.match, "p_shared_cabins_match.Rds")