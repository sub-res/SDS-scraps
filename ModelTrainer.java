package com.ac.sds.ml;

import org.apache.log4j.Logger;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.ParamPair;
import org.apache.spark.ml.regression.*;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class ModelTrainer
{
    private static final Logger logger = Logger.getLogger(ModelTrainer.class);
    private double trainRatio;
    private String classificationEvalMetric;
    private String regressionEvalMetric;
    private GridSearchParams gsp;

    public ModelTrainer()
    {
        this.gsp = new NormalGridSearchParams();
        trainRatio = 0.8;
        classificationEvalMetric = "accuracy";
        regressionEvalMetric = "rmse";
    }

    public ModelTrainer(GridSearchParams gsp)
    {
        this.gsp = gsp;
        trainRatio = 0.8;
        classificationEvalMetric = "accuracy";
        regressionEvalMetric = "rmse";
    }

    public ModelTrainer setGridSearchParams(GridSearchParams gsp)
    {
        this.gsp = gsp;
        return this;
    }

    public ModelTrainer setTrainRatio(double trainRatio)
    {
        this.trainRatio = trainRatio;
        return this;
    }

    //Available metrics:
    //accuracy, weightedPrecision, weightedRecall, f1
    public ModelTrainer setClassificationEvalMetric(String classificationEvalMetric)
    {
        this.classificationEvalMetric = classificationEvalMetric;
        return this;
    }

    //Available metrics:
    //rmse, mse, m2, mae
    public ModelTrainer setRegressionEvalMetric(String regressionEvalMetric)
    {
        this.regressionEvalMetric = regressionEvalMetric;
        return this;
    }

    /* classifiers */

    public DecisionTreeClassificationModel trainDecisionTreeCM(Dataset<Row> df, int maxCategories)
    {
        df = indexFeatures(df, maxCategories);
        DecisionTreeClassifier dtc = new DecisionTreeClassifier();
        return (DecisionTreeClassificationModel)getTVSClassificationModel(dtc, gsp.getParamGrid(dtc), df).bestModel();
    }

    public RandomForestClassificationModel trainRandomForestCM(Dataset<Row> df, int maxCategories)
    {
        df = indexFeatures(df, maxCategories);
        RandomForestClassifier rfc = new RandomForestClassifier();
        return (RandomForestClassificationModel)getTVSClassificationModel(rfc, gsp.getParamGrid(rfc), df).bestModel();
    }

    public GBTClassificationModel trainGradientBoostedTreesCM(Dataset<Row> df, int maxCategories)
    {
        df = indexFeatures(df, maxCategories);
        GBTClassifier gbtc = new GBTClassifier();
        return (GBTClassificationModel)getTVSClassificationModel(gbtc, gsp.getParamGrid(gbtc), df).bestModel();
    }

    public MultilayerPerceptronClassificationModel trainMultilayerPerceptronCM(Dataset<Row> df)
    {
        DenseVector vec = df.first().getAs("features");
        int features = vec.toArray().length;
        final int classes = 2; //binary classification
        MultilayerPerceptronClassifier mpc = new MultilayerPerceptronClassifier();
        return (MultilayerPerceptronClassificationModel) getTVSClassificationModel(mpc,
            gsp.getParamGrid(mpc, features, classes), df).bestModel();
    }

    public LogisticRegressionModel trainLogisticRegressionCM(Dataset<Row> df)
    {
        LogisticRegression lr = new LogisticRegression();
        return (LogisticRegressionModel)getTVSClassificationModel(lr, gsp.getParamGrid(lr), df).bestModel();
    }

    public NaiveBayesModel trainNaiveBayesCM(Dataset<Row> df)
    {
        NaiveBayes nb = new NaiveBayes();
        return (NaiveBayesModel)getTVSClassificationModel(nb, gsp.getParamGrid(nb), df).bestModel();
    }

    /*regressors*/

    public LinearRegressionModel trainLinearRM(Dataset<Row> df)
    {
        LinearRegression lr = new LinearRegression();
        return (LinearRegressionModel)getTSVRegressionModel(lr, gsp.getParamGrid(lr), df).bestModel();
    }

    public GeneralizedLinearRegressionModel trainGeneralizedLinearRM(Dataset<Row> df)
    {
        GeneralizedLinearRegression glr = new GeneralizedLinearRegression();
        return (GeneralizedLinearRegressionModel)getTSVRegressionModel(glr, gsp.getParamGrid(glr), df).bestModel();
    }

    public DecisionTreeRegressionModel trainDecisionTreeRM(Dataset<Row> df, int maxCategories)
    {
        df = indexFeatures(df, maxCategories);
        DecisionTreeRegressor dtr = new DecisionTreeRegressor();
        return (DecisionTreeRegressionModel)getTSVRegressionModel(dtr, gsp.getParamGrid(dtr), df).bestModel();
    }

    public RandomForestRegressionModel trainRandomForestRM(Dataset<Row> df, int maxCategories)
    {
        df = indexFeatures(df, maxCategories);
        RandomForestRegressor rfr = new RandomForestRegressor();
        return (RandomForestRegressionModel)getTSVRegressionModel(rfr, gsp.getParamGrid(rfr), df).bestModel();
    }

    public GBTRegressionModel trainGradientBoostedTreesRM(Dataset<Row> df, int maxCategories)
    {
        df = indexFeatures(df, maxCategories);
        GBTRegressor gbtr = new GBTRegressor();
        return (GBTRegressionModel)getTSVRegressionModel(gbtr, gsp.getParamGrid(gbtr), df).bestModel();
    }

    public IsotonicRegressionModel trainIsotonicRM(Dataset<Row> df)
    {
        IsotonicRegression ir = new IsotonicRegression();
        return (IsotonicRegressionModel)getTSVRegressionModel(ir, gsp.getParamGrid(ir), df).bestModel();
    }

    private Dataset<Row> indexFeatures(Dataset<Row> df, int maxCategories)
    {
        return new VectorIndexer()
            .setInputCol(ModelHelper.FEATURES)
            .setOutputCol(ModelHelper.FEATURES + "_indexed")
            .setMaxCategories(maxCategories)
            .fit(df)
            .transform(df)
            .drop(ModelHelper.FEATURES)
            .withColumnRenamed(ModelHelper.FEATURES + "_indexed", ModelHelper.FEATURES);
    }

    private TrainValidationSplitModel getTSVModel(Estimator estimator, Evaluator evaluator, ParamMap[] paramGrid,
        Dataset<Row> df)
    {
        long t_start = System.currentTimeMillis();

        TrainValidationSplitModel tvsm = new TrainValidationSplit()
            .setEstimator(estimator)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setTrainRatio(trainRatio)
            .fit(df);

        logger.debug("Time spent training " + tvsm.bestModel().toString() + ": " +
            (System.currentTimeMillis() - t_start) + " ms.");
        logger.debug("Chosen parameters:");
        for (ParamPair pp : tvsm.bestModel().extractParamMap().toList())
        {
            if (!pp.param().name().endsWith("Col"))
            {
                logger.debug(pp.param().name() + ": " + pp.value().toString());
            }
        }

        return tvsm;
    }

    private TrainValidationSplitModel getTVSClassificationModel(Estimator estimator, ParamMap[] paramGrid,
        Dataset<Row> df)
    {
        return getTSVModel(estimator, new MulticlassClassificationEvaluator().setMetricName(classificationEvalMetric),
            paramGrid, df);
    }

    private TrainValidationSplitModel getTSVRegressionModel(Estimator estimator, ParamMap[] paramGrid,
        Dataset<Row> df)
    {
        return getTSVModel(estimator, new RegressionEvaluator().setMetricName(regressionEvalMetric), paramGrid, df);
    }
}
