package com.ac.sds.ml;

import com.ac.sds.data.AcDataSet;
import com.google.common.collect.Maps;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructField;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class AdoTransformer
{
    private static final Logger logger = Logger.getLogger(AdoTransformer.class);

    public static Dataset<Row> loadAllRows(String keyspace, String table, SparkSession session)
    {
        Map<String, String> options = Maps.newHashMap();
        options.put("keyspace", keyspace);
        options.put("table", table);

        return session
            .read()
            .format("org.apache.spark.sql.cassandra")
            .options(options)
            .load();
    }

    public static Dataset<Row> loadColumn(String keyspace, String table, SparkSession session, String attrID)
    {
        Map<String, String> options = Maps.newHashMap();
        options.put("keyspace", keyspace);
        options.put("table", table);

        return session
            .read()
            .format("org.apache.spark.sql.cassandra")
            .options(options)
            .load()
            .select(AcDataSet.ADO_ID, AcDataSet.TIMESTAMP, attrID);
    }

    public static Dataset<Row> loadRows(String keyspace, String table, SparkSession session, String attrID,
        boolean useSingleColumn)
    {
        long t_start;
        long t_end;

        t_start = System.currentTimeMillis();
        Dataset<Row> df;
        if (useSingleColumn)
        {
            df = loadColumn(keyspace, table, session, attrID);

            if (df.schema()
                .fields()[df.schema().fieldIndex(attrID)]
                .dataType()
                .typeName()
                .toLowerCase()
                .contains("string"))
            {
                throw new IllegalArgumentException("Unable to base regression on indexed string value of feature \'"
                    + attrID + "\'.");
            }
        }
        else
        {
            df = loadAllRows(keyspace, table, session);
        }
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for loading dataframe: " + (t_end - t_start) + "ms");

        return df;
    }

    //input: dataframe where all rows except "ado_id" are not of data type string
    //output: dataframe where all rows except "ado_id" are of data type double
    private static Dataset<Row> doublify(Dataset<Row> df)
    {
        final StructField[] fields = df.schema().fields();
        for (StructField field : fields)
        {
            if (!field.name().equals(AcDataSet.ADO_ID))
            {
                df = df.withColumn(field.name(), df.col(field.name()).cast("double"));
            }
        }
        return df;
    }

    //input: dataframe where all columns ending with "_value" are numerical
    //output: dataframe with ado_id, timestamp, specified feature, and other features merged into column of vectors
    private static Dataset<Row> toLabeledPoints(Dataset<Row> df, String labelFeature, boolean useSingleColumn)
    {
        VectorAssembler assembler = new VectorAssembler();

        if (useSingleColumn)
        {
            assembler.setInputCols(new String[] { AcDataSet.TIMESTAMP });
        }
        else
        {
            final StructField[] fields = df.schema().fields();
            List<String> features = new ArrayList<>();
            for (StructField field : fields)
            {
                if (field.name().endsWith(AcDataSet.VALUE) && !field.name().equals(labelFeature))
                {
                    features.add(field.name());
                }
            }
            assembler.setInputCols(features.toArray(new String[features.size()]));
        }

        //note: vectorassembler can output either dense or sparse vectors depending on which takes less memory
        //find some way to guarantee it's always a dense vector if we want to use WithMean for normalizing
        assembler.setOutputCol(ModelHelper.FEATURES);

        df = assembler.transform(df);
        return df.withColumnRenamed(labelFeature, ModelHelper.LABEL).select(AcDataSet.ADO_ID, AcDataSet.TIMESTAMP,
            ModelHelper.LABEL, ModelHelper.FEATURES);
    }

    private static List<String> getUniqueAdoIDs(Dataset<Row> df)
    {
        return df.javaRDD()
            .map(row -> row.getString(row.fieldIndex(AcDataSet.ADO_ID)))
            .distinct()
            .collect();
    }

    //input: dataframe with columns ado_id, timestamp, sds_label and sds_features
    //output: same as above but with feature vectors normalized and scaled between 0.0 and 1.0
    private static Dataset<Row> normalizeColumnsByAdo(Dataset<Row> df, SparkSession session)
    {
        List<String> adoIDs = getUniqueAdoIDs(df);
        Dataset<Row> normalized = session.createDataFrame(new ArrayList<>(), df.schema());

        for (String adoID : adoIDs)
        {
            Dataset<Row> dfTemp = df.filter(row -> row.getString(row.fieldIndex(AcDataSet.ADO_ID)).equals(adoID));

            logger.debug("Normalizing " + adoID + " (" + dfTemp.count() + " entries)");

            /*
            WithMean is set to false as a hotfix for StandardScaler crashing on sparse vectors which might occur
            depending on the data read. VectorAssembler outputs either dense or sparse vectors depending on which takes
            less memory, see toLabeledPoints()
            TODO: find a way to guarantee all vectors are dense if we want to use WithMean
             */
            StandardScaler ssc = new StandardScaler()
                .setInputCol(ModelHelper.FEATURES)
                .setOutputCol(ModelHelper.FEATURES + "_normalized")
                .setWithMean(false)
                .setWithStd(true);
            StandardScalerModel sscm = ssc.fit(dfTemp);
            dfTemp = sscm.transform(dfTemp);

            MinMaxScaler mmsc = new MinMaxScaler()
                .setInputCol(ModelHelper.FEATURES + "_normalized")
                .setOutputCol(ModelHelper.FEATURES + "_scaled")
                .setMin(0.0)
                .setMax(1.0);
            MinMaxScalerModel mmscm = mmsc.fit(dfTemp);
            dfTemp = mmscm.transform(dfTemp);
            dfTemp = dfTemp.drop(ModelHelper.FEATURES, ModelHelper.FEATURES + "_normalized")
                .withColumnRenamed(ModelHelper.FEATURES + "_scaled", ModelHelper.FEATURES);

            normalized = normalized.union(dfTemp);
        }

        logger.info("Normalized " + adoIDs.size() + " ADOs with a total of " + normalized.count() + " entries.");

        return normalized;
    }

    //input: dataframe with columns ending with "_value"
    //output: same as above, but with every string value in "_value" columns indexed as double
    private static Dataset<Row> indexStrings(Dataset<Row> df)
    {
        if (df.limit(1).javaRDD().isEmpty())
        {
            logger.warn("Dataframe is empty after filtering. AdoTransformer will yield an empty dataframe.");
            return df;
        }

        final StructField[] fields = df.schema().fields();
        for (StructField field : fields)
        {
            if (field.name().endsWith(AcDataSet.VALUE) && field.dataType().typeName().toLowerCase().contains("string"))
            {
                StringIndexer indexer = new StringIndexer()
                    .setInputCol(field.name())
                    .setOutputCol(field.name() + "_indexed");
                df = indexer.setHandleInvalid("skip").fit(df).transform(df)
                    .drop(field.name())
                    .withColumnRenamed(field.name() + "_indexed", field.name());
            }
        }
        return df;
    }

    //input: dataframe with column named ModelHelper.LABEL
    //output: dataframe with column named ModelHelper.LABEL's values binarized between 0 (suspect, positive) and 1 (not suspect, negative)
    private static Dataset<Row> binarizeLabels(Dataset<Row> df, SparkSession session)
    {
        JavaRDD<Row> dfRdd = df.javaRDD();
        StructField[] fields = df.schema().fields();
        dfRdd = dfRdd.map(r ->
        {
            Object[] rowObjects = new Object[fields.length];
            for (int i = 0; i < fields.length; i++)
            {

                if (fields[i].name().equals(ModelHelper.LABEL))
                {
                    rowObjects[i] = ((int)r.getDouble(i) & -65) == 2 || (int)r.getDouble(i) >= 128 ? 0. : 1.;
                }
                else
                {
                    rowObjects[i] = r.get(i);
                }
            }
            return RowFactory.create(rowObjects);
        });

        return session.createDataFrame(dfRdd, df.schema());
    }

    //input: dataframe containing a numerical column "timestamp"
    //output: same as above, ordered by column "timestamp"
    private static Dataset<Row> sortByTimestamp(Dataset<Row> df)
    {
        return df.orderBy(AcDataSet.TIMESTAMP);
    }

    //for splitting data for time series forecasting regression
    public static List<Dataset<Row>> splitByDate(Dataset<Row> df, Date splitDate)
    {
        List<Dataset<Row>> splits = new ArrayList<>();

        splits.add(df.filter(r -> r.getDouble(r.fieldIndex(AcDataSet.TIMESTAMP)) <= splitDate.getTime()));
        splits.add(df.filter(r -> r.getDouble(r.fieldIndex(AcDataSet.TIMESTAMP)) > splitDate.getTime()));

        return splits;
    }

    public static Dataset<Row> loadAndTransformRows(String keyspace, String table, String labelFeature,
        double nullsThreshold, boolean useSingleColumn, SparkSession session)
    {
        Dataset<Row> df = loadRows(keyspace, table, session, labelFeature, useSingleColumn);
        return transformRows(df, labelFeature, nullsThreshold, useSingleColumn, session);
    }

    public static Dataset<Row> loadAndTransformRows(String keyspace, String table, String labelFeature,
        double nullsThreshold, boolean useSingleColumn, Date date, SparkSession session)
    {
        Dataset<Row> df = loadRows(keyspace, table, session, labelFeature, useSingleColumn);
        return transformRows(df, labelFeature, nullsThreshold, useSingleColumn, date, session);
    }

    public static Dataset<Row> transformRows(Dataset<Row> df, String labelFeature,
        double nullsThreshold, boolean useSingleColumn, SparkSession session)
    {
        return transformRows(df, labelFeature, nullsThreshold, useSingleColumn, new Date(0), session);
    }

    public static Dataset<Row> transformRows(Dataset<Row> df, String labelFeature,
        double nullsThreshold, boolean useSingleColumn, Date dateThreshold, SparkSession session)
    {
        //function execution time bookkeeping
        long t_start;
        long t_end;

        //quick way to force exception if label feature is invalid to begin with, does not modify df
        df.select(labelFeature);

        //cleanse null columns
        t_start = System.currentTimeMillis();
        df = FilterColumns.filterIncompleteColumns(df, nullsThreshold);
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for filtering incomplete columns: " + (t_end - t_start) + "ms");

        //check if label column wasn't lost in filtering
        StructField[] fields_filtered = df.schema().fields();
        boolean labelSurvived = false;
        for (StructField field : fields_filtered)
        {
            if (field.name().equals(labelFeature))
            {
                labelSurvived = true;
            }
        }

        if (!labelSurvived)
        {
            throw new IllegalArgumentException("Label feature \'" + labelFeature + "\' was not in dataframe after " +
                "filtering null values. Try increasing the nullsThreshold parameter (current: " + nullsThreshold +
                "), or choosing a different label feature.");
        }

        //automatic check for multiple ADOs
        boolean multipleAdos = df.select(AcDataSet.ADO_ID).distinct().count() > 1;

        /*
        for reasons as of yet unknown, caching the dataframe is only really possible after this point, when
        filterIncompleteColumns has been applied (specifically, the drop operation on the dataframe). Caching any
        earlier seems to replace the values in the dataframe with zeroes and empty strings, causing the ADO
        transformation to fail. So far there doesn't seem to be any rhyme or reason as to why the caching fails like
        this, could possibly be a bug in the datastax cassandra connector.
         */
        t_start = System.currentTimeMillis();
        df.cache();
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for caching dataframe: " + (t_end - t_start) + "ms");

        //cleanse null rows
        t_start = System.currentTimeMillis();
        df = df.filter(new RemoveEmptyRowFilter());
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for filtering incomplete rows: " + (t_end - t_start) + "ms");

        //filter on date threshold
        if (dateThreshold.getTime() > 0)
        {
            df = df.filter(r -> r.getLong(r.fieldIndex(AcDataSet.TIMESTAMP)) <= dateThreshold.getTime());
        }

        //strings to doubles
        t_start = System.currentTimeMillis();
        df = indexStrings(df);
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for indexing string values: " + (t_end - t_start) + "ms");

        //doublify int values
        t_start = System.currentTimeMillis();
        df = doublify(df);
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for casting columns to double: " + (t_end - t_start) + "ms");

        //to labeledPoints
        t_start = System.currentTimeMillis();
        df = toLabeledPoints(df, labelFeature, useSingleColumn);
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for conversion to labeled points: " + (t_end - t_start) + "ms");

        //normalize
        if (multipleAdos)
        {
            if (useSingleColumn || labelFeature.endsWith(AcDataSet.VALUE))
            {
                logger.warn(
                    "Normalization of response on regression data with multiple ADOs is not supported and will" +
                        " yield unusable data unless filtered by individual ADO IDs.");
            }
            t_start = System.currentTimeMillis();
            df = normalizeColumnsByAdo(df, session);
            t_end = System.currentTimeMillis();
            logger.debug("Time elapsed for normalizing features: " + (t_end - t_start) + "ms");
        }

        //label indexing, classifiers don't like labels with value >100
        //even when there are <100 distinct labels
        if (labelFeature.endsWith(AcDataSet.STATUS)) //if not regression
        {
            t_start = System.currentTimeMillis();
            //skip actual indexing step and immediatly binarize instead
            df = binarizeLabels(df, session);
            t_end = System.currentTimeMillis();
            logger.debug("Time elapsed for indexing labels: " + (t_end - t_start) + "ms");
        }

        t_start = System.currentTimeMillis();
        df = sortByTimestamp(df);
        t_end = System.currentTimeMillis();
        logger.debug("Time elapsed for sorting entries by timestamp: " + (t_end - t_start) + "ms");

        return df;
    }
}
