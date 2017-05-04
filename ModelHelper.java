package com.ac.sds.ml;

import com.ac.sds.data.AcDataSet;
import com.datastax.driver.core.Session;
import com.datastax.driver.core.Statement;
import com.datastax.driver.core.querybuilder.QueryBuilder;
import org.apache.commons.lang.SerializationUtils;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class ModelHelper
{
    public static final String LABEL = "label";
    public static final String FEATURES = "features";

    public static final String MT_MNAME = "name";
    public static final String MT_MODEL = "model";

    private final Logger logger = Logger.getLogger(ModelHelper.class);

    private double nullThreshold;
    private int maxCategories;
    private String keyspace;
    private String table;
    private String modelTable;

    private String reportDirectory;

    private ModelTrainer mt;

    //ctor
    //use default grid search params
    public ModelHelper()
    {
        mt = new ModelTrainer();

        nullThreshold = 0.0;
        maxCategories = 32;
        keyspace = "keyspace";
        table = "table";
        modelTable = "modelTable";

        reportDirectory = "";
    }

    //ctor
    //for setting GridSearchParams manually (e.g. for testing)
    public ModelHelper(GridSearchParams gsp)
    {
        mt = new ModelTrainer(gsp);

        nullThreshold = 0.0;
        maxCategories = 32;
        keyspace = "keyspace";
        table = "table";
        modelTable = "modelTable";
    }

    public ModelHelper setNullThreshold(double nullThreshold)
    {
        this.nullThreshold = nullThreshold;
        return this;
    }

    public ModelHelper setMaxCategories(int maxCategories)
    {
        this.maxCategories = maxCategories;
        return this;
    }

    public ModelHelper setKeyspace(String keyspace)
    {
        this.keyspace = keyspace;
        return this;
    }

    public ModelHelper setTable(String table)
    {
        this.table = table;
        return this;
    }

    public ModelHelper setModelTable(String modelTable)
    {
        this.modelTable = modelTable;
        return this;
    }

    public ModelHelper setGridSearchParams(GridSearchParams gsp)
    {
        mt.setGridSearchParams(gsp);
        return this;
    }

    public ModelHelper setReportDirectory(String dir)
    {
        if (!dir.endsWith("/"))
        {
            dir += "/";
        }
        reportDirectory = dir;
        return this;
    }

    public ModelTrainer getModelTrainer()
    {
        return this.mt;
    }

    //loads dataframe of labeledpoints fit for classification
    public Dataset<Row> getClassificationDataframe(String attrID, JavaSparkContext context)
    {
        return AdoTransformer.loadAndTransformRows(keyspace, table, attrID + AcDataSet.STATUS, nullThreshold, true,
            new SparkSession(context.sc()));
    }

    //loads dataframe of labeledpoints fit for regression
    public Dataset<Row> getRegressionDataframe(String attrID, JavaSparkContext context)
    {
        return AdoTransformer.loadAndTransformRows(keyspace, table, attrID + AcDataSet.VALUE, nullThreshold, true,
            new SparkSession(context.sc()));
    }

    //loads dataframe of labeledpoints fit for time series regression
    public Dataset<Row> getTimeSeriesRegressionDataframe(String attrID, JavaSparkContext context)
    {
        return AdoTransformer.loadAndTransformRows(keyspace, table, attrID + AcDataSet.VALUE, nullThreshold, true,
            new SparkSession(context.sc()));
    }

    //loads dataframe of labeledpoints fit for classification with date threshold
    public Dataset<Row> getClassificationDataframe(String attrID, Date dateThreshold, JavaSparkContext context)
    {
        return AdoTransformer.loadAndTransformRows(keyspace, table, attrID + AcDataSet.STATUS, nullThreshold, false,
            dateThreshold, new SparkSession(context.sc()));
    }

    //loads dataframe of labeledpoints fit for regression with date threshold
    public Dataset<Row> getRegressionDataframe(String attrID, Date dateThreshold, JavaSparkContext context)
    {
        return AdoTransformer.loadAndTransformRows(keyspace, table, attrID + AcDataSet.VALUE, nullThreshold, false,
            dateThreshold, new SparkSession(context.sc()));
    }

    //loads dataframe of labeledpoints fit for time series regression with date threshold
    public Dataset<Row> getTimeSeriesRegressionDataframe(String attrID, Date dateThreshold,
        JavaSparkContext context)
    {
        return AdoTransformer.loadAndTransformRows(keyspace, table, attrID + AcDataSet.VALUE, nullThreshold, true,
            dateThreshold, new SparkSession(context.sc()));
    }

    //returns classification models trained with a Dataframe of labeledpoints
    public Map<String, Model<?>> trainClassificationModels(Dataset<Row> df, boolean forceTrain, Session session)
    {
        Map<String, Model<?>> models = new HashMap<>();
        String modelName;

        logger.info("Retrieving classification model 1 of 6...");
        modelName = "decision tree classification";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainDecisionTreeCM(df, maxCategories);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving classification model 2 of 6...");
        modelName = "random forest classification";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainRandomForestCM(df, maxCategories);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving classification model 3 of 6...");
        modelName = "gradient boosted trees classification";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainGradientBoostedTreesCM(df, maxCategories);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving classification model 4 of 6...");
        modelName = "multilayer perceptron classification";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainMultilayerPerceptronCM(df);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving classification model 5 of 6...");
        modelName = "logistic regression classification";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainLogisticRegressionCM(df);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving classification model 6 of 6...");
        modelName = "naive bayes classification";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainNaiveBayesCM(df);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }


        logger.info("Successfully retrieved " + models.size() + " classification models.");
        return models;
    }

    //returns regression models trained with a Dataframe of labeledpoints
    public Map<String, Model<?>> trainRegressionModels(Dataset<Row> df, boolean forceTrain, Session session)
    {
        Map<String, Model<?>> models = new HashMap<>();
        String modelName;

        logger.info("Retrieving regression model 1 of 6...");
        modelName = "decision tree regression";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainDecisionTreeRM(df, maxCategories);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving regression model 2 of 6...");
        modelName = "random forest regression";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainRandomForestRM(df, maxCategories);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving regression model 3 of 6...");
        modelName = "gradient boosted trees regression";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainGradientBoostedTreesRM(df, maxCategories);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving regression model 4 of 6...");
        modelName = "linear regression";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainLinearRM(df);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving regression model 5 of 6...");
        modelName = "generalized linear regression";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainGeneralizedLinearRM(df);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Retrieving regression model 6 of 6...");
        modelName = "isotonic regression";
        try
        {
            if (forceTrain || !modelExists(modelName, session))
            {
                Model model = mt.trainIsotonicRM(df);
                models.put(modelName, model);
                saveModel(model, modelName, session);
            }
            else
            {
                models.put(modelName, loadModel(modelName, session));
            }
        }
        catch (Exception e)
        {
            logger.warn("Failed to retrieve " + modelName + " model: " + e.getMessage());
        }

        logger.info("Successfully retrieved " + models.size() + " regression models.");
        return models;
    }

    //load data and evaluate classification models
    public void evaluateClassifications(String attrID, JavaSparkContext context, double[] splits, long seed,
        boolean forceTrain, Session session)
    {
        Dataset<Row> df = getClassificationDataframe(attrID, context);
        evaluateClassifications(df, splits, seed, forceTrain, session);
    }

    //evaluate classification models based on acquired dataframe of labeledpoints
    public void evaluateClassifications(Dataset<Row> labeledPoints, double[] splits, long seed, boolean forceTrain,
        Session session)
    {
        Dataset<Row>[] dfs = labeledPoints.randomSplit(splits, seed);
        Dataset<Row> trainingValidationSet = dfs[0];
        trainingValidationSet.cache();
        Dataset<Row> testingSet = dfs[1];
        testingSet.cache();

        Map<String, Model<?>> classificationModels =
            trainClassificationModels(trainingValidationSet, forceTrain, session);

        final int LEGEND_OFFSET = 1;

        String[] legend = new String[]{"Model:", "Accuracy:", "Precision:", "True positive rate:",
            "False positive rate:", "True negative rate:", "False negative rate:", "F1 score:", "Area under ROC curve:",
            "Area under PR curve:"};
        String[][] report = new String[classificationModels.size() + LEGEND_OFFSET][legend.length];
        report[0] = legend;
        int idx = LEGEND_OFFSET;

        for (Map.Entry<String, Model<?>> pair : classificationModels.entrySet())
        {
            Dataset<Row> prediction = pair.getValue().transform(testingSet).select(LABEL, "prediction");

            MulticlassMetrics metrics = new MulticlassMetrics(prediction);
            BinaryClassificationMetrics metrics2 = new BinaryClassificationMetrics(prediction); //for ROC/PR
            double label = 0.0; //0.0 = suspect = positive

            try
            {
                report[idx] = new String[]
                    {
                        pair.getKey(),
                        Double.toString(metrics.accuracy()),
                        Double.toString(metrics.precision(label)),
                        Double.toString(metrics.truePositiveRate(label)),
                        Double.toString(metrics.falsePositiveRate(label)),
                        Double.toString(1.0 - metrics.falsePositiveRate(label)),
                        Double.toString(1.0 - metrics.truePositiveRate(label)),
                        Double.toString(metrics.fMeasure(label)),
                        Double.toString(metrics2.areaUnderROC()),
                        Double.toString(metrics2.areaUnderPR())
                    };
            } catch (Exception e)
            {
                report[idx] = new String[]{
                    pair.getKey(),
                    Double.toString(metrics.accuracy()),
                    "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
                    Double.toString(metrics2.areaUnderROC()),
                    Double.toString(metrics2.areaUnderPR())
                };
                logger.debug(pair.getKey() + ": " + e.getMessage());
            }
            logger.debug(pair.getKey() + ":\n" + metrics.confusionMatrix());
            idx++;
        }

        String formattedReport = formatReport(report, legend);
        logger.info("Classification report:\n" + formattedReport);
        writeReport(formattedReport, reportDirectory + "classificationreport.txt");
    }

    //load data and evaluate regression models
    //dateThreshold for splitting between training and testing set
    public void evaluateRegressions(String attrID, Date dateThreshold, JavaSparkContext context, boolean forceTrain,
        Session session)
    {
        Dataset<Row> df = getRegressionDataframe(attrID, context);
        evaluateRegressions(df, dateThreshold, forceTrain, session);
    }

    //evaluate regression models based on acquired dataframe of labeledpoints
    //dateThreshold for splitting between training and testing set
    public void evaluateRegressions(Dataset<Row> labeledPoints, Date dateThreshold, boolean forceTrain, Session session)
    {
        List<Dataset<Row>> dfs = AdoTransformer.splitByDate(labeledPoints, dateThreshold);
        Dataset<Row> trainingValidationSet = dfs.get(0);
        trainingValidationSet.cache();
        Dataset<Row> testingSet = dfs.get(1);
        testingSet.cache();

        Map<String, Model<?>> regressionModels = trainRegressionModels(trainingValidationSet, forceTrain, session);

        final int LEGEND_OFFSET = 1;

        String[] legend = new String[]{"Model:", "Mean squared error:", "Root mean squared error:", "R^2:",
            "Mean absolute error:"};
        String[][] report = new String[regressionModels.size() + LEGEND_OFFSET][legend.length];
        report[0] = legend;
        int idx = LEGEND_OFFSET;

        for (Map.Entry<String, Model<?>> pair : regressionModels.entrySet())
        {
            Dataset<Row> prediction = pair.getValue().transform(testingSet);

            //for some reason using RegressionMetrics throws an error: haven't figured out exactly why
            //this works too, but it's more cumbersome
            RegressionEvaluator re = new RegressionEvaluator();
            re.setMetricName("mse");
            String mse = Double.toString(re.evaluate(prediction));
            re.setMetricName("rmse");
            String rmse = Double.toString(re.evaluate(prediction));
            re.setMetricName("r2");
            String r2 = Double.toString(re.evaluate(prediction));
            re.setMetricName("mae");
            String mae = Double.toString(re.evaluate(prediction));

            report[idx] = new String[]
                {
                    pair.getKey(),
                    mse,
                    rmse,
                    r2,
                    mae
                };
            idx++;
        }

        String formattedReport = formatReport(report, legend);
        logger.info("Regression report:\n" + formattedReport);
        writeReport(formattedReport, reportDirectory + "regressionreport.txt");
    }

    //load data and evaluate time series regression models
    //dateThreshold for splitting between training and testing set
    public void evaluateTimeSeriesRegressions(String attrID, Date dateThreshold, JavaSparkContext context,
        boolean forceTrain, Session session)
    {
        Dataset<Row> df = getTimeSeriesRegressionDataframe(attrID, context);
        evaluateTimeSeriesRegressions(df, dateThreshold, forceTrain, session);
    }

    //evaluate time series regression models based on acquired dataframe of labeledpoints
    //dateThreshold for splitting between training and testing set
    public void evaluateTimeSeriesRegressions(Dataset<Row> labeledPoints, Date dateThreshold, boolean forceTrain,
        Session session)
    {
        List<Dataset<Row>> dfs = AdoTransformer.splitByDate(labeledPoints, dateThreshold);
        Dataset<Row> trainingValidationSet = dfs.get(0);
        trainingValidationSet.cache();
        Dataset<Row> testingSet = dfs.get(1);
        testingSet.cache();

        Map<String, Model<?>> regressionModels = trainRegressionModels(trainingValidationSet, forceTrain, session);

        final int LEGEND_OFFSET = 1;

        String[] legend = new String[]{"Model:", "Mean squared error:", "Root mean squared error:", "R^2:",
            "Mean absolute error:"};
        String[][] report = new String[regressionModels.size() + LEGEND_OFFSET][legend.length];
        report[0] = legend;
        int idx = LEGEND_OFFSET;

        for (Map.Entry<String, Model<?>> pair : regressionModels.entrySet())
        {
            Dataset<Row> prediction = pair.getValue().transform(testingSet);

            RegressionEvaluator re = new RegressionEvaluator();
            re.setMetricName("mse");
            String mse = Double.toString(re.evaluate(prediction));
            re.setMetricName("rmse");
            String rmse = Double.toString(re.evaluate(prediction));
            re.setMetricName("r2");
            String r2 = Double.toString(re.evaluate(prediction));
            re.setMetricName("mae");
            String mae = Double.toString(re.evaluate(prediction));

            report[idx] = new String[]
                {
                    pair.getKey(),
                    mse,
                    rmse,
                    r2,
                    mae
                };
            idx++;
        }

        String formattedReport = formatReport(report, legend);
        logger.info("Time series regression report:\n" + formattedReport);
        writeReport(formattedReport, reportDirectory + "timeseriesregressionreport.txt");
    }

    //model save method
    public void saveModel(Model model, String name, Session session)
    {
        try
        {
            ByteBuffer buffer = ByteBuffer.wrap(SerializationUtils.serialize(model));
            Statement stm = QueryBuilder.insertInto(modelTable).value(MT_MNAME, name).value(MT_MODEL, buffer);
            session.execute(stm);
        }
        catch (Exception e)
        {
            throw new IllegalArgumentException("Failed to save model \'" + name + "\' to model table \'" + modelTable
                + "\': " + e.getMessage());
        }
    }

    //model load method
    public Model loadModel(String name, Session session)
    {
        Statement readModel = QueryBuilder.select(MT_MODEL).from(modelTable).where(QueryBuilder.eq(MT_MNAME, name));
        com.datastax.driver.core.Row modelRow = session.execute(readModel).one();
        if (modelRow != null)
        {
            return (Model)SerializationUtils.deserialize(modelRow.getBytes(MT_MODEL).array());
        }
        else
        {
            throw new IllegalArgumentException("Failed to load model. Model with name \'" + name
                + "\' does not exist in model table \'" + modelTable + "\'.");
        }
    }

    //clear models
    public void clearModels(Session session)
    {
        session.execute("TRUNCATE " + modelTable + ";");
    }

    public void listModels(Session session)
    {
        List<com.datastax.driver.core.Row> rows = session.execute("SELECT " + MT_MNAME + " FROM " + modelTable).all();
        for (com.datastax.driver.core.Row row : rows)
        {
            logger.info(row.getString(0));
        }
    }

    private boolean modelExists(String name, Session session)
    {
        Statement readModel = QueryBuilder.select(MT_MODEL).from(modelTable).where(QueryBuilder.eq(MT_MNAME, name));
        return !(session.execute(readModel).one() == null);
    }

    private String formatReport(String[][] report, String[] legend)
    {
        final int columnSpacing = 2;

        //column-wise space padding
        for (int i = 0; i < legend.length; i++)
        {
            int maxLen = 0;
            for (int j = 0; j < report.length; j++)
            {
                maxLen = report[j][i].length() > maxLen ? report[j][i].length() : maxLen;
            }

            for (int j = 0; j < report.length; j++)
            {
                int padding = columnSpacing + maxLen - report[j][i].length();
                for (int k = 0; k < padding; k++)
                {
                    report[j][i] = report[j][i] + " ";
                }
            }
        }

        //concatenate into single string
        String formattedReport = "";
        for (int i = 0; i < report.length; i++)
        {
            for (int j = 0; j < report[i].length; j++)
            {
                formattedReport += report[i][j];
            }
            formattedReport += "\n";
        }

        return formattedReport;
    }

    private void writeReport(String report, String path)
    {
        try
        {
            Files.write(Paths.get(path), report.getBytes());
            logger.debug("Wrote report to " + Paths.get(path).toAbsolutePath().toString());
        }
        catch (Exception e)
        {
            logger.error("Unable to write report \'" + path + "\': " + e.getMessage());
        }
    }
}
