# By Symon Kimitei
# BANK CUSTOMER CHURN MODELING
# MODELS USED: NAIVE BAYES CLASSIFIER, LOGISTIC REGRESSION FOR BINARY CLASSIFICATION & DECISION TREE CLASSIFIER
# USING APACHE SPARK TO DETECT EMAIL SPAM
#=========================================================================================

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import (VectorAssembler,OneHotEncoder,StringIndexer)
from pyspark.sql import SparkSession
import pyspark.sql as sparksql
spark = SparkSession.builder.appName('Exited').getOrCreate()
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Load the data into the server
X_Rdd = spark.read.load("file:///home/usrmkang/skimitei/project/Churn_Modelling.csv", format="csv", sep=",", inferSchema="true", header="True")

# Get some information about the dataset and its datatypes.
X_Rdd.printSchema()

# Count the number of customers who exited and those who didn't from the Label Column
X_Rdd.groupBy('Exited').count().show()

# Count the number of customers by Geography
X_Rdd.groupBy("Geography").count().orderBy(col("count").desc()).show()

# Count the number of customers by those with Credit card or not
X_Rdd.groupBy(X_Rdd["HasCrCard"]==1).count().orderBy(col("count").desc()).show()


# Use the filter operation to calculate the number of people under 35 who exited the bank. Is age and gender a factor?
X_Rdd.filter(((X_Rdd['Gender']=='Female') | (X_Rdd['Gender']=='Male')) & (X_Rdd['Age']< '35') & (X_Rdd['Exited']== '1')).count()

from pyspark.ml.feature import (VectorAssembler,OneHotEncoder, StringIndexer)
# Most of ML algorithms cannot work directly with categorical data. 
# The encoding allows algorithms which expect continuous features to use categorical features.
# It does not need to know how many categories in a 
# feature beforehand the combination of StringIndexer and OneHotEncoder 

#Encode Gender
gender_indexer=StringIndexer(inputCol='Gender', outputCol='genderIndex')
gender_encoder=OneHotEncoder(inputCol='genderIndex', outputCol='genderVec')

#Encode Geography
geography_indexer=StringIndexer(inputCol='Geography', outputCol='geographyIndex')
geography_encoder=OneHotEncoder(inputCol='geographyIndex', outputCol='geographyVec')

# Drop some columns since they have no effect on our analysis
X_Rdd = X_Rdd.drop("RowNumber", "CustomerId", "Surname")

assembler = VectorAssembler(inputCols=['CreditScore',
 'geographyVec',
 'genderVec',
 'Age',
 'Tenure',
 'Balance',
 'NumOfProducts',
 'HasCrCard',
  'IsActiveMember',
 'EstimatedSalary'],outputCol='features')



# LOGISTIC REGRESSION PREDICTION
#======================================================================================
from pyspark.ml.classification import LogisticRegression
# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="Exited", featuresCol="features", maxIter=10)

# Create a Pipeline, which consists of a sequence of Pipeline stages to be run in a specific order in order to properly preprocess the data ( Workflow )
pipeline = Pipeline(stages=[gender_indexer, geography_indexer, gender_encoder, geography_encoder, assembler, lr])

# split dataset to train and test.

train_data,test_data = X_Rdd.randomSplit([0.8,0.2])

# Next Fit the model. For this we will use the pipeline that was created and the train_data
model = pipeline.fit(train_data)

# Transform Test Data
lr_predictions = model.transform(test_data)
lr_predictions.printSchema()
lr_predictions.show()


# Evaluate a model
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction", metricName="accuracy")
lr_acc = acc_evaluator.evaluate(lr_predictions)
print('A Logistic regression algorithm had an accuracy of: {0:2.2f}%'.format(lr_acc*100))

# Confusion matrix
lr_predictions.groupBy('Exited','prediction').count().show()

# View model's predictions and probabilities of each prediction class
selected = lr_predictions.select("Tenure", "Age","probability","Exited", "prediction")


# ==============5-FOLD CROSS VALIDATION ** LOGISTIC MODEL CLASSIFIER============================
# Create initial Logistic Model object
lr_classifier = LogisticRegression(labelCol="label", featuresCol="features")

# Create ParamGrid for Cross Validation
# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
paramGrid = ParamGridBuilder().addGrid(lr_classifier.regParam, [0.1, 0.01])\
                  .addGrid(lr_classifier.fitIntercept, [False, True])\
                  .addGrid(lr_classifier.elasticNetParam, [0.0, 0.5, 1.0]).build()

# Create 5-fold CrossValidator
lrevaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction')
lrcv = CrossValidator(estimator = lr_classifier, estimatorParamMaps = dtparamGrid, evaluator = lrevaluator, numFolds = 5)

# Transform Train Data
lr_train_data = model.transform(train_data)
lr_train_data.printSchema()
lr_train_data.show()
newTrain=lr_train_data.drop("rawPrediction","probability","prediction")
newLR_Train =newTrain.withColumnRenamed('Exited', 'label')

# Run cross validations
lrcvModel = lrcv.fit(newLR_Train)

# Display the average area under the ROC curve  
# Smoothing values
lrcvModel.avgMetrics

# Use test set here so we can measure the accuracy of our model on new data
# Making Predictions
lr_CV_test_data = model.transform(test_data)
lr_CV_test_data.printSchema()
lr_CV_test_data.show(10)
newTest_lr=nb_CV_test_data.drop("rawPrediction","probability","prediction")
newTest_lr =newTest_lr.withColumnRenamed('Exited', 'label')
lr_predictions1 = lrcvModel .transform(newTest)

# Display the first five 5-fold cross validation predictions
lr_predictions1.select("features","label","prediction").show(5)

# Confusion matrix
lr_predictions1.groupBy('label','prediction').count().show()

# cvModel uses the best model found from the Cross Validation
# Evaluate the best model
lr_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print('Area under the precision-recall curve:', lr_evaluator.evaluate(lr_predictions1))
#============================END================================================


# NAIVE BAYES - CLASSIFIER
#==========================================================================

# FIT the Naïve Bayes classifier
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()
nb = NaiveBayes(labelCol="Exited", featuresCol="features")
pipeline = Pipeline(stages=[gender_indexer, geography_indexer, gender_encoder, geography_encoder, assembler, nb])

# split dataset to train and test.
seed = 0  

#Split the RDD into training set and test set
train_data,test_data = X_Rdd.randomSplit([0.8,0.2], seed)
# Number of records of each dataframe
train_data.count()
test_data.count()

# Next Fit the model. For this we will use the pipeline that was created and train_data
model = pipeline.fit(train_data)

# Transform Test Data
nb_predictions = model.transform(test_data)
nb_predictions.printSchema()
nb_predictions.show()


# Evaluate a model
# Select ( true label, prediction) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction", metricName="accuracy")
nb_acc = acc_evaluator.evaluate(nb_predictions)
print('A Naive Bayes algorithm had an accuracy of: {0:2.2f}%'.format(nb_acc*100))

# Confusion matrix
nb_predictions.groupBy('Exited','prediction').count().show()

# View model's predictions and probabilities of each prediction class
selected = nb_predictions.select("Tenure", "Age","probability","Exited", "prediction")

# ==============5-FOLD CROSS VALIDATION ** NAIVE BAYES============================
# Transform Train Data
nb_CV_train_data = model.transform(train_data)
nb_CV_train_data.printSchema()
nb_CV_train_data.show()
newTrain=nb_CV_train_data.drop("rawPrediction","probability","prediction")
newTrain =newTrain.withColumnRenamed('Exited', 'label')

# FIT the Naïve Bayes classifier
nb = NaiveBayes()
paramGrid_nb = ParamGridBuilder().addGrid(nb.smoothing, np.linspace(0.3, 10, 5)).build()
# Apply five-fold cross validation
cross_val_nb = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid_nb,evaluator=BinaryClassificationEvaluator(),numFolds= 5) 
cv_model_nb = cross_val_nb.fit(newTrain)

# Display the average area under the ROC curve values 
# using the five smoothing values
cv_model_nb.avgMetrics

# Making Predictions
nb_CV_test_data = model.transform(test_data)
nb_CV_test_data.printSchema()
nb_CV_test_data.show(10)
newTest=nb_CV_test_data.drop("rawPrediction","probability","prediction")
newTest =newTest.withColumnRenamed('Exited', 'label')

nb_CV_predictions = cv_model_nb.transform(newTest)
nb_CV_predictions.select('label', 'prediction').show(10)

# Calculate the number of correct and incorrect predictions
nb_CV_predictions.groupBy('label','prediction').count().show()

# Area under the precision-recall curve:
nb_evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction')
print('Area under the precision-recall curve:', nb_evaluator.evaluate(nb_CV_predictions))
#============================END================================================



# DECISION TREE PREDICTION 
#=======================================================================================
# create a DecisionTree object
dtc = DecisionTreeClassifier(labelCol='Exited',featuresCol='features')

# Create a Pipeline, which consists of a sequence of Pipeline stages to be run in a specific order in order to properly preprocess the data ( Workflow )
pipeline = Pipeline(stages=[gender_indexer, geography_indexer, gender_encoder, geography_encoder, assembler, dtc])


# split dataset to train and test.
train_data,test_data = X_Rdd.randomSplit([0.8,0.2])

# Next Fit the model. For this we will use the pipeline that was created and train_data
model = pipeline.fit(train_data)

# Transform Test Data
dtc_predictions = model.transform(test_data)
dtc_predictions.select("features","Exited","prediction").show(5)
dtc_predictions.printSchema()
dtc_predictions.show()

# Evaluate a model
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction", metricName="accuracy")
dtc_acc = acc_evaluator.evaluate(dtc_predictions)
print('A Decision Tree algorithm had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))

# Confusion matrix
dtc_predictions.groupBy('Exited','prediction').count().show()
# View model's predictions and probabilities of each prediction class
selected = dtc_predictions.select("Tenure", "Age","probability","Exited", "prediction")



# ==============5-FOLD CROSS VALIDATION ** DECISION TREE MODEL============================
# Create initial Decision Tree Model
dtc = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=2)

# Create ParamGrid for Cross Validation
dtparamGrid = (ParamGridBuilder().addGrid(dtc.maxDepth, [2, 5, 10, 20, 30])\
                .addGrid(dtc.maxBins, [10, 20, 40, 80, 100]).build())

# Create 5-fold CrossValidator
dtevaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction')
dtcv = CrossValidator(estimator = dtc, estimatorParamMaps = dtparamGrid, evaluator = dtevaluator, numFolds = 5)


# Transform Train Data
dt_CV_train_data = model.transform(train_data)
dt_CV_train_data.printSchema()
dt_CV_train_data.show()
newTrain=dt_CV_train_data.drop("rawPrediction","probability","prediction")
newDT_Train =newTrain.withColumnRenamed('Exited', 'label')

# Run cross validations
dtcvModel = dtcv.fit(newDT_Train)

# Display the average area under the ROC curve values 
# using the five smoothing values
dtcvModel.avgMetrics


# Use test set here so we can measure the accuracy of our model on new data
# Making Predictions
dt_CV_test_data = model.transform(test_data)
dt_CV_test_data.printSchema()
dt_CV_test_data.show(10)
newTest=dt_CV_test_data.drop("rawPrediction","probability","prediction")
newTest =newTest.withColumnRenamed('Exited', 'label')
dtc_predictions1 = dtcvModel.transform(newTest)

# Display the first five 5-fold cross validation predictions
dtc_predictions1.select("features","label","prediction").show(5)

# Confusion matrix
dtc_predictions1.groupBy('label','prediction').count().show()

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
dt_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print('Area under the precision-recall curve:', dt_evaluator.evaluate(dtc_predictions1))
#============================END================================================



















