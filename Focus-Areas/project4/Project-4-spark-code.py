import os

# # Install java
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
! java -version

# Install pyspark
! pip install --ignore-installed pyspark==2.4.4

! java -version

import sys
import time

#Spark ML and SQL
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import array_contains
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import numpy as np
from pyspark.sql.functions import isnan, when, count, col

from pyspark.sql import SparkSession
import pyspark.sql as sparksql
spark = SparkSession.builder.appName('stroke').getOrCreate()
# train = spark.read.csv('https://raw.githubusercontent.com/aman1002/McKinseyOnlineHackathon-Healthcare-/master/train.csv', inferSchema=True,header=True)

import pandas as pd
train = pd.read_csv("https://raw.githubusercontent.com/aman1002/McKinseyOnlineHackathon-Healthcare-/master/train.csv")
train['smoking_status'] = train['smoking_status'].fillna('No Info')
train.head()

train.dtypes

train['bmi'] = train['bmi'].fillna((train['bmi'].mean()))

train = spark.createDataFrame(train)
train.show()

# fill in miss values for bmi 
# as this is numecial data , we will simple fill the missing values with mean
from pyspark.sql.functions import mean
mean = train.select(mean(train['bmi'])).collect()
mean_bmi = mean[0][0]
print(mean_bmi)
train = train.fillna( { 'bmi':mean_bmi } )
#test = test.fillna( { 'bmi':mean_bmi } )

#to print the categories of columns
print(train.select('gender').distinct().rdd.map(lambda r: r[0]).collect())
print(train.select('ever_married').distinct().rdd.map(lambda r: r[0]).collect())
print(train.select('work_type').distinct().rdd.map(lambda r: r[0]).collect())
print(train.select('Residence_type').distinct().rdd.map(lambda r: r[0]).collect())
print(train.select('smoking_status').distinct().rdd.map(lambda r: r[0]).collect())
print(train.select('stroke').distinct().rdd.map(lambda r: r[0]).collect())


train.dtypes



train.select([count(when(isnan(c), c)).alias(c) for c in train.columns]).show()



test = pd.read_csv("https://raw.githubusercontent.com/aman1002/McKinseyOnlineHackathon-Healthcare-/master/test.csv")
test['smoking_status'] = test['smoking_status'].fillna('No Info')
test = spark.createDataFrame(test)
#test.na.fill('No Info', subset=['smoking_status'])
test.show()

from pyspark.sql.functions import *
from pyspark.sql.types import *
a = [IntegerType(),StringType(),IntegerType(),IntegerType(),IntegerType(),
     StringType(),StringType(),StringType(),FloatType(),
     IntegerType(),StringType(),IntegerType()]
print(len(a))
print(len(train.columns))
train = train.withColumn("id", train["id"].cast(IntegerType()))
train = train.withColumn("gender", train["gender"].cast(StringType()))
train = train.withColumn("age", train["age"].cast(IntegerType()))
train = train.withColumn("hypertension", train["hypertension"].cast(IntegerType()))
train = train.withColumn("heart_disease", train["heart_disease"].cast(IntegerType()))
train = train.withColumn("ever_married", train["ever_married"].cast(StringType()))
train = train.withColumn("work_type", train["work_type"].cast(StringType()))
train = train.withColumn("Residence_type", train["Residence_type"].cast(StringType()))
train = train.withColumn("avg_glucose_level", train["avg_glucose_level"].cast(FloatType()))
train = train.withColumn("bmi", train["bmi"].cast(FloatType()))
train = train.withColumn("smoking_status", train["smoking_status"].cast(StringType()))
train = train.withColumn("stroke", train["stroke"].cast(IntegerType()))

  # train = train.withColumn(i, train[i].cast(i))
# train = train.withColumn("Plays", train["Plays"].cast(IntegerType()))

train.dtypes

test = test.withColumn("id", test["id"].cast(IntegerType()))
test = test.withColumn("gender", test["gender"].cast(StringType()))
test = test.withColumn("age", test["age"].cast(IntegerType()))
test = test.withColumn("hypertension", test["hypertension"].cast(IntegerType()))
test = test.withColumn("heart_disease", test["heart_disease"].cast(IntegerType()))
test = test.withColumn("ever_married", test["ever_married"].cast(StringType()))
test = test.withColumn("work_type", test["work_type"].cast(StringType()))
test = test.withColumn("Residence_type", test["Residence_type"].cast(StringType()))
test = test.withColumn("avg_glucose_level", test["avg_glucose_level"].cast(FloatType()))
test = test.withColumn("bmi", test["bmi"].cast(FloatType()))
test = test.withColumn("smoking_status", test["smoking_status"].cast(StringType()))

print("The number of row and columns: ")
print(str(train.count())+","+str(len(train.columns)))
 

train=train.dropDuplicates();
totalRow=train.count()
print("The number of row and columns after removing duplicate rows: ")
print(str(totalRow)+","+str(len(train.columns)))

print("nThe type of each column variable")
train.printSchema()

print("Data Classes :")
train.groupBy('stroke').count().show()

train.stat.crosstab("gender", "stroke").show()

train.stat.crosstab("smoking_status", "stroke").show()

train.describe().show()

#correlation between age and bmi column
print('correlation among age and bmi', train.stat.corr("age","bmi"))
print('correlation among age and hypertension',  train.stat.corr("age","hypertension"))
print('correlation among age and heart_disease',  train.stat.corr("age","heart_disease"))
print('correlation among age and avg_glucose_level',  train.stat.corr("age","avg_glucose_level"))
print('correlation among bmi and heart_disease',  train.stat.corr("bmi","heart_disease"))
print('correlation among bmi and hypertension',  train.stat.corr("bmi","hypertension"))
print('correlation among bmi and avg_glucose_level',  train.stat.corr("bmi","avg_glucose_level"))
print('correlation among hypertension and avg_glucose_level',  train.stat.corr("hypertension","avg_glucose_level"))
print('correlation among   heart_disease and avg_glucose_level',  train.stat.corr("heart_disease","avg_glucose_level"))
print('correlation among   heart_disease and hypertension',  train.stat.corr("heart_disease","hypertension"))

# create DataFrame as a temporary view for SQL queries
train.createOrReplaceTempView('table')

# sql query to find the number of people in specific work_type who have had stroke and not
spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE stroke == 1 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show()
spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE stroke == 0 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show()
spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Male') as percentage FROM table WHERE stroke== 1 AND gender = 'Male' GROUP BY gender").show()
spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Female') as percentage FROM table WHERE stroke== 1 AND gender = 'Female' GROUP BY gender").show()

# fill in missing values for smoking status
# As this is categorical data, we will add one data type "No Info" for the missing one
train_f = train
test_f = test

train_f.printSchema()

# from pyspark.sql.functions import mean
# mean = train_f.select(mean(train_f['bmi'])).collect()
# mean_bmi = mean[0][0]
# print(mean_bmi)
# print(mean)

from pyspark.sql.functions import mean as _mean, stddev as _stddev, col

df_stats = train_f.select(
    _mean(col('bmi')).alias('mean'),
    _stddev(col('bmi')).alias('std')
).collect()

mean = df_stats[0]['mean']
std = df_stats[0]['std']
mean

train_f.describe().show()

test_f.describe().show()





# indexing all categorical columns in the dataset
from pyspark.ml.feature import StringIndexer
indexer1 = StringIndexer(inputCol="gender", outputCol="genderIndex")
indexer2 = StringIndexer(inputCol="ever_married", outputCol="ever_marriedIndex")
indexer3 = StringIndexer(inputCol="work_type", outputCol="work_typeIndex")
indexer4 = StringIndexer(inputCol="Residence_type", outputCol="Residence_typeIndex")
indexer5 = StringIndexer(inputCol="smoking_status", outputCol="smoking_statusIndex")

from pyspark.ml.feature import OneHotEncoderEstimator
encoder = OneHotEncoderEstimator(inputCols=["genderIndex","ever_marriedIndex","work_typeIndex","Residence_typeIndex","smoking_statusIndex"],
                                 outputCols=["genderVec","ever_marriedVec","work_typeVec","Residence_typeVec","smoking_statusVec"])

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['genderVec',
 'age',
 'hypertension',
 'heart_disease',
 'ever_marriedVec',
 'work_typeVec',
 'Residence_typeVec',
 'avg_glucose_level',
 'bmi',
 'smoking_statusVec'],outputCol='features')

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, dtc])

# Doing one hot encoding of indexed data
from pyspark.ml.feature import OneHotEncoderEstimator
encoder = OneHotEncoderEstimator(inputCols=["genderIndex","ever_marriedIndex","work_typeIndex","Residence_typeIndex","smoking_statusIndex"],
                                 outputCols=["genderVec","ever_marriedVec","work_typeVec","Residence_typeVec","smoking_statusVec"])

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['genderVec',
 'age',
 'hypertension',
 'heart_disease',
 'ever_marriedVec',
 'work_typeVec',
 'Residence_typeVec',
 'avg_glucose_level',
 'bmi',
 'smoking_statusVec'],outputCol='features')

# splitting training and validation data
train_data,val_data = train_f.randomSplit([0.7,0.3])




from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import  matplotlib.pyplot as plt
import numpy as np
import itertools
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



rfc = RandomForestClassifier(labelCol='stroke',featuresCol='features')
pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, rfc])
# training model pipeline with data
model = pipeline.fit(train_data)
# making prediction on model with validation data
rfc_predictions = model.transform(val_data)
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
print('A Random Forest algorithm had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="f1")
#evaluator = BinaryClassificationEvaluator()
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
print('A Random Forest algorithm had an F1-Score of: {0:2.2f}%'.format(rfc_acc*100))
#AUC
evaluator = BinaryClassificationEvaluator(labelCol="stroke")
print('Test Area Under ROC', evaluator.evaluate(rfc_predictions))

y_true = rfc_predictions.select(['stroke']).collect()
y_pred = rfc_predictions.select(['prediction']).collect()
print(classification_report(y_true, y_pred))

# argmax returns the index of the max value in a row
cm = confusion_matrix(y_true, y_pred)
cm_plot_labels = ['Not Stroke', 'Stroke']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')



# Select example rows to display.
test_selected = rfc_predictions .select("id", "features", "prediction","probability")
test_selected.limit(5).toPandas()



from pyspark.ml.classification import LogisticRegression
lg = LogisticRegression(labelCol='stroke',featuresCol='features')
pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, lg])
# training model pipeline with data
model = pipeline.fit(train_data)
# making prediction on model with validation data
lg_predictions = model.transform(val_data)
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
lg_acc = acc_evaluator.evaluate(lg_predictions)
print('A Logistic Regression algorithm had an accuracy of: {0:2.2f}%'.format(lg_acc*100))
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="f1")
#evaluator = BinaryClassificationEvaluator()
lg_acc = acc_evaluator.evaluate(lg_predictions)
print('A logistic algorithm had an F1-Score of: {0:2.2f}%'.format(lg_acc*100))
#AUC
evaluator = BinaryClassificationEvaluator(labelCol="stroke")
print('Test Area Under ROC', evaluator.evaluate(lg_predictions))

y_true = lg_predictions.select(['stroke']).collect()
y_pred = lg_predictions.select(['prediction']).collect()
print(classification_report(y_true, y_pred))

# argmax returns the index of the max value in a row
cm = confusion_matrix(y_true, y_pred)
cm_plot_labels = ['Not Stroke', 'Stroke']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# balancing the dataset by duplicating the minority class samples
from pyspark.sql.functions import col, explode, array, lit,ceil
train_data
major_df = train_data.filter(col("stroke") == 0)
print(major_df.count())
minor_df = train_data.filter(col("stroke") == 1)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))

a = range(ratio+1)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
combined_df = major_df.unionAll(oversampled_df)
combined_df.show()
print(combined_df.count())


train_data=combined_df

#random forest
rfc = RandomForestClassifier(labelCol='stroke',featuresCol='features')
pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, rfc])


# training model pipeline with data
model = pipeline.fit(train_data)
# making prediction on model with validation data
rfc_predictions = model.transform(val_data)
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
print('A Random Forest algorithm had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="f1")
#evaluator = BinaryClassificationEvaluator()
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
print('A Random Forest algorithm had an F1-Score of: {0:2.2f}%'.format(rfc_acc*100))
#AUC
evaluator = BinaryClassificationEvaluator(labelCol="stroke")
print('Test Area Under ROC', evaluator.evaluate(rfc_predictions))

y_true = rfc_predictions.select(['stroke']).collect()
y_pred = rfc_predictions.select(['prediction']).collect()
print(classification_report(y_true, y_pred))

# argmax returns the index of the max value in a row
cm = confusion_matrix(y_true, y_pred)
cm_plot_labels = ['Not Stroke', 'Stroke']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

#logistic Regression
lg = LogisticRegression(labelCol='stroke',featuresCol='features')
pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, lg])
# training model pipeline with data
model = pipeline.fit(train_data)
# making prediction on model with validation data
lg_predictions = model.transform(val_data)
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
lg_acc = acc_evaluator.evaluate(lg_predictions)
print('A Logistic Regression algorithm had an accuracy of: {0:2.2f}%'.format(lg_acc*100))
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="f1")
#evaluator = BinaryClassificationEvaluator()
lg_acc = acc_evaluator.evaluate(lg_predictions)
print('A Random Forest algorithm had an F1-Score of: {0:2.2f}%'.format(lg_acc*100))
#AUC
evaluator = BinaryClassificationEvaluator(labelCol="stroke")
print('Test Area Under ROC', evaluator.evaluate(lg_predictions))



y_true = lg_predictions.select(['stroke']).collect()
y_pred = lg_predictions.select(['prediction']).collect()
print(classification_report(y_true, y_pred))

# argmax returns the index of the max value in a row
cm = confusion_matrix(y_true, y_pred)
cm_plot_labels = ['Not Stroke', 'Stroke']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')