from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col,isnan,when,count,countDistinct,lit
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler,VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier,RandomForestClassifier,LinearSVC,NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator

#SparkSession and Dataframe creation:
spark=SparkSession.builder.appName("classification1").getOrCreate()
myFile="/home/spark-3.0.1-bin-hadoop3.2/input/Breast_classification/data.csv"
df=spark.read.format("csv").option("header","true").option("inferSchema","true").load(myFile)

##################################################################################
##################################################################################
##################################################################################
###########################   EXPLORATIVE ANALYSIS  ##############################
##################################################################################
##################################################################################
##################################################################################

#Show some rows and columns of the dataframe to understand it better:
df.select(df.columns[:10]).show(8)

#Dataframe shape
print("               Dataframe's shape: (%s,%d)" %(df.count(), len(df.columns)))
print("")

#Count the NULL values in every column:
nullValues = df.select([count(when(col(c).isNull(),c)).alias(c) for c in df.columns]).show(truncate=False,vertical=True)
#33th column has ONLY NULL values --> you have to delete it:
df2=df.drop(F.col("_c32"))
#df2.show(1)
print("                 Column ""c_32"" deleted -->    Shape of the new dataframe : (%s,%d) " %(df2.count(),len(df2.columns)))
print("")
#Count the number of 'M' and 'B' diagnosis:
print("Number of M  and  B diagnosis:")
df2.groupBy("diagnosis").count().show()


###########################################################################################
#Creation of different graphs to understand better the dataframe:
df3=df2.toPandas()
plot1=sns.countplot(df3['diagnosis'],label='count')
plt.show()

#Pairplot graph creation:
plot2=sns.pairplot(df3.iloc[:,1:6], hue='diagnosis')
plt.show()

#Show the correlation between columns:
df3.corr()

#Show only some of the previous computed correlation values (only for the first 11 columns):
plot3=sns.heatmap(df3.iloc[:,1:12].corr(),annot=True,fmt='.0%')
plt.show()




###################################################################################
###################################################################################
###################################################################################
#########################      PIPELINE        ####################################
###################################################################################
###################################################################################
###################################################################################

#Creation of a function with all the operations to compute always at the beginning of every pipeline
#for every classification method used:

def start():
    global indexer,vector,stdScaler,train,test

    indexer=StringIndexer(inputCol="diagnosis", outputCol="categoryIndex")

    vector=VectorAssembler(inputCols=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'], outputCol="features")

    stdScaler=StandardScaler(inputCol="features",outputCol="scaled")

    train,test =df2.randomSplit([0.7,0.3])


start()


#Creation of different functions to call recursively for every classification algorithm to use:
def LogRegr():
    global pipeline
    lr=LogisticRegression(labelCol="categoryIndex",featuresCol="features",maxIter=10)
    pipeline=Pipeline(stages=[indexer,vector,stdScaler,lr])


def DecisTree():
    global pipeline
    tree=DecisionTreeClassifier(labelCol="categoryIndex",featuresCol="features")
    pipeline=Pipeline(stages=[indexer,vector,stdScaler,tree])


def RandForest():
    global pipeline
    forest=RandomForestClassifier(featuresCol="features",labelCol="categoryIndex")
    pipeline=Pipeline(stages=[indexer,vector,stdScaler,forest])

def SVCLinear():
    global pipeline
    svc_lin=LinearSVC(labelCol="categoryIndex",featuresCol="features",maxIter=5)
    pipeline=Pipeline(stages=[indexer,vector,stdScaler,svc_lin])

def NaiveBay():
    global pipeline
    naive=NaiveBayes(labelCol="categoryIndex",featuresCol="features",modelType="gaussian")
    pipeline=Pipeline(stages=[indexer,vector,stdScaler,naive])


#We create a List containing all the functions created above, to be able to call them one after the other,
#executing the various classification algorithms in sequence one after the other
functionList=[LogRegr,DecisTree,RandForest,SVCLinear,NaiveBay]
algorithmNames=["LOGISTIC REGRESSION","DECISION TREE","RANDOM FOREST","LINEAR SVC","NAIVE BAYES"]

for i in range(len(functionList)):
    functionList[i]()
    algorithmNames[i]

    model=pipeline.fit(train)

    prediction=model.transform(test)

    # We create two lists, (one for the true values and one for the predictions) just for a better display on the terminal
    selection=prediction.select(F.col("categoryIndex").cast("int"),F.col("prediction").cast("int"))
    trueVal_list=list(selection.select("categoryIndex").toPandas()["categoryIndex"])
    predictVal_list=list(selection.select("prediction").toPandas()["prediction"])

    print("//////////////////////////////////////////////---------------------------------------//////////////////////////////////////////////////////")
    print("                                                   -------- %s --------                                 " %algorithmNames[i])
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
    print("TRUE VALUES : ")
    print(trueVal_list)
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
    print("PREVISIONS : ")
    print(predictVal_list)
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
    print("")

    #The sklearn library is used to produce the CONFUSION MATRIX and other metrics:
    from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
    evaluator=MulticlassClassificationEvaluator(labelCol="categoryIndex",predictionCol="prediction")
    y_true=prediction.select(["categoryIndex"]).collect()
    y_predicted=prediction.select(["prediction"]).collect()

    #Creation of the ROC-AUC CURVE graph
    #Calculation of the values fpr, tpr, thresholds and roc_auc useful for the construction of the curve:
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
    roc_auc = auc(fpr, tpr)

    # ROC curve plot:
    plt.plot(fpr, tpr, label='AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (Specifity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    #Other calculated metrics (accuracy, precision, f1 score and recall):
    print(classification_report(y_true,y_predicted))
    accuracy=evaluator.evaluate(prediction,{evaluator.metricName:"accuracy"})
    precision=evaluator.evaluate(prediction,{evaluator.metricName:"precisionByLabel"})
    f1_score=evaluator.evaluate(prediction,{evaluator.metricName:"f1"})
    recall=evaluator.evaluate(prediction,{evaluator.metricName:"recallByLabel"})
    print("Accuracy = %g " % accuracy)
    print("Test error = %g " % (1.0-accuracy))
    print("Precision = %g " % precision )
    print("F1 score = %g " % f1_score)
    print("Recall = %g " % recall)
    print("-------------------------------------------------------------------------------------------------------------------------------------------")
    print("")
    print("")
    print("")


spark.stop()
