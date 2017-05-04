package com.ac.sds.it.test.ml;

import com.ac.sds.data.AcDataSet;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;

import java.util.*;

public class ReutersSuspectGenerator
{
    private static Logger logger = Logger.getLogger(ReutersSuspectGenerator.class);

    private static final int STATUS_SUSPECT = 2;

    private double suspectRatio;
    private double suspectMagnitude;

    public Dataset<Row> generateSuspectsStdDev(Dataset<Row> df, List<String> attributes, SparkSession session)
    {
        Random rand = new Random();
        return generateSuspectsStdDev(df, attributes, rand.nextInt(), session);
    }

    public Dataset<Row> generateSuspectsStdDev(Dataset<Row> df, List<String> attributes, int seed,
        SparkSession session)
    {
        final double sRatio = suspectRatio;
        final double sMagnitude = suspectMagnitude;

        final Random rand = new Random(seed);
        long count = df.count();

        //gather standard deviations for every ADO ID
        logger.info("Gathering standard deviations...");

        Map<String, Row> stddevPerAdo = new HashMap<>();
        List<String> adoIDs = df
            .javaRDD()
            .map(r -> r.getString(r.fieldIndex(AcDataSet.ADO_ID)))
            .distinct()
            .collect();

        for (String adoID : adoIDs)
        {
            Dataset<Row> descTable = df
                .filter(r -> r.getString(r.fieldIndex(AcDataSet.ADO_ID)).equals(adoID))
                .describe();

            boolean nullStddev = false;
            Row stddevs = descTable.filter(r -> r.getString(r.fieldIndex("summary")).equals("stddev")).first();
            for (String attribute : attributes)
            {
                if (stddevs.isNullAt(stddevs.fieldIndex(attribute + AcDataSet.VALUE)))
                {
                    nullStddev = true;
                    break;
                }
            }

            if(!nullStddev)
            {
                stddevPerAdo.put(adoID, stddevs);
            }

            logger.debug("retrieved stddevs for " + adoID);
        }
        logger.info("Retrieved standard deviations for " + stddevPerAdo.size() + " ADOs.");

        //perform map on RDD because dataset lacks a row encoder
        JavaRDD<Row> dfRdd = df.javaRDD().map(r ->
        {
            if (rand.nextInt((int) count) < count * sRatio)
            {
                String attrID = attributes.get(rand.nextInt(attributes.size()));
                Object[] rowObjects = new Object[r.schema().fields().length];
                for (int i = 0; i < r.schema().fields().length; i++)
                {
                    if (r.schema().fields()[i].name().equals(attrID + AcDataSet.STATUS) &&
                        !r.isNullAt(i)) //status field
                    {
                        rowObjects[i] = STATUS_SUSPECT;
                    }
                    else if (r.schema().fields()[i].name().equals(attrID + AcDataSet.VALUE) &&
                        !r.isNullAt(i)) //value field
                    {
                        //retrieve stddev for attribute value
                        Row stddevRow = stddevPerAdo.get(r.getString(r.fieldIndex(AcDataSet.ADO_ID)));
                        String stddevStr = stddevRow.getString(stddevRow.fieldIndex(attrID + AcDataSet.VALUE));
                        double stddev = Double.parseDouble(stddevStr);

                        //suspect value offset
                        double suspectDev = (1.0 + rand.nextDouble()) * stddev;
                        suspectDev *= rand.nextInt(2) == 0 ? sMagnitude : -sMagnitude;

                        switch (r.schema().fields()[i].dataType().typeName())
                        {
                            case "integer":
                                rowObjects[i] = (int) ((double) r.getInt(i) + suspectDev);
                                break;
                            case "double":
                                rowObjects[i] = r.getDouble(i) + suspectDev;
                                break;
                            default:
                                rowObjects[i] = r.get(i);
                                /*
                                Other type, no implementation for making it suspect (yet?) so we'll leave it unchanged
                                for now. This should really log a warning but apparently the log4j Logger doesn't work
                                from inside a lambda function.
                                */
                                break;
                        }
                    }
                    else
                    {
                        rowObjects[i] = r.get(i); //leave unchanged
                    }

                }

                r = RowFactory.create(rowObjects);
            }

            return r;
        });
        logger.info("Applied suspect generation.");

        //conversion back to dataset
        return session.createDataFrame(dfRdd, df.schema());
    }

    public  Dataset<Row> generateSuspectsZeroes(Dataset<Row> df, List<String> attributes, SparkSession session)
    {
        Random rand = new Random();
        return generateSuspectsZeroes(df, attributes, rand.nextInt(), session);
    }

    public Dataset<Row> generateSuspectsZeroes(Dataset<Row> df, List<String> attributes, int seed,
        SparkSession session)
    {
        final double sRatio = suspectRatio;

        final Random rand = new Random(seed);
        long count = df.count();

        JavaRDD<Row> dfRdd = df.javaRDD().map(r ->
        {
            if (rand.nextInt((int) count) < count * sRatio)
            {
                String attrID = attributes.get(rand.nextInt(attributes.size()));
                Object[] rowObjects = new Object[r.schema().fields().length];
                for (int i = 0; i < r.schema().fields().length; i++)
                {
                    if (r.schema().fields()[i].name().equals(attrID + AcDataSet.STATUS) &&
                        !r.isNullAt(i))
                    {
                        rowObjects[i] = STATUS_SUSPECT;
                    }
                    else if (r.schema().fields()[i].name().equals(attrID + AcDataSet.VALUE) &&
                        !r.isNullAt(i))
                    {
                        switch (r.schema().fields()[i].dataType().typeName())
                        {
                            case "integer":
                                rowObjects[i] = 0;
                                break;
                            case "double":
                                rowObjects[i] = 0.;
                                break;
                            default:
                                rowObjects[i] = r.get(i);
                                break;
                        }
                    }
                    else
                    {
                        rowObjects[i] = r.get(i);
                    }
                }

                r = RowFactory.create(rowObjects);
            }
            return r;
        });

        return session.createDataFrame(dfRdd, df.schema());
    }

    public  Dataset<Row> generateSuspectsNegatives(Dataset<Row> df, List<String> attributes, SparkSession session)
    {
        Random rand = new Random();
        return generateSuspectsNegatives(df, attributes, rand.nextInt(), session);
    }

    public Dataset<Row> generateSuspectsNegatives(Dataset<Row> df, List<String> attributes, int seed,
        SparkSession session)
    {
        final double sRatio = suspectRatio;

        final Random rand = new Random(seed);
        long count = df.count();

        JavaRDD<Row> dfRdd = df.javaRDD().map(r ->
        {
            if (rand.nextInt((int) count) < count * sRatio)
            {
                String attrID = attributes.get(rand.nextInt(attributes.size()));
                Object[] rowObjects = new Object[r.schema().fields().length];
                for (int i = 0; i < r.schema().fields().length; i++)
                {
                    if (r.schema().fields()[i].name().equals(attrID + AcDataSet.STATUS) &&
                        !r.isNullAt(i))
                    {
                        rowObjects[i] = STATUS_SUSPECT;
                    }
                    else if (r.schema().fields()[i].name().equals(attrID + AcDataSet.VALUE) &&
                        !r.isNullAt(i))
                    {
                        switch (r.schema().fields()[i].dataType().typeName())
                        {
                            case "integer":
                                rowObjects[i] = -r.getInt(i);
                                break;
                            case "double":
                                rowObjects[i] = -r.getDouble(i);
                                break;
                            default:
                                rowObjects[i] = r.get(i);
                                break;
                        }
                    }
                    else
                    {
                        rowObjects[i] = r.get(i);
                    }
                }

                r = RowFactory.create(rowObjects);
            }
            return r;
        });

        return session.createDataFrame(dfRdd, df.schema());
    }

    public Dataset<Row> generateSuspectsHiLo(Dataset<Row> df, SparkSession session)
    {
        Random rand = new Random();
        return generateSuspectsHiLo(df, rand.nextInt(), session);
    }

    public Dataset<Row> generateSuspectsHiLo(Dataset<Row> df, int seed, SparkSession session)
    {
        final double sRatio = suspectRatio;
        final double sMagnitude = suspectMagnitude;

        final Random rand = new Random(seed);
        long count = df.count();

        List<String> attributes = new ArrayList<>();
        attributes.add("re_ua_high");
        attributes.add("re_ua_low");
        attributes.add("re_ua_open");
        attributes.add("re_ua_close2");
        attributes.add("re_ua_ask");
        attributes.add("re_ua_bid");
        attributes.add("re_ua_mid");

        JavaRDD<Row> dfRdd = df.javaRDD().map(r ->
        {
            if (rand.nextInt((int) count) < count * sRatio)
            {
                List<Double> attrVals = new ArrayList<>();
                for (String attribute : attributes)
                {
                    if (!r.isNullAt(r.fieldIndex(attribute + AcDataSet.VALUE)))
                    {
                        attrVals.add(r.getDouble(r.fieldIndex(attribute + AcDataSet.VALUE)));
                    }
                }

                if (attrVals.size() == 0)
                {
                    return r;
                }

                Collections.sort(attrVals);
                double newVal;
                double minVal = attrVals.get(0);
                double maxVal = attrVals.get(attrVals.size() - 1);
                double diff = maxVal - minVal;

                String attrID = attributes.get(rand.nextInt(attributes.size()));

                switch(attrID)
                {
                    case "re_ua_high":
                        newVal = (minVal - diff) + (diff * rand.nextDouble() * 0.9 * 2 * sMagnitude);
                        break;
                    case "re_ua_low":
                        newVal = (minVal + diff) + (diff * rand.nextDouble() * 0.9 * 2 * sMagnitude);
                        break;
                    default:
                        if (rand.nextInt(2) == 0)
                        {
                            newVal = minVal - (diff * (0.1 + (rand.nextDouble() * 0.9 * sMagnitude)));
                        }
                        else
                        {
                            newVal = maxVal + (diff * (0.1 + (rand.nextDouble() * 0.9 * sMagnitude)));
                        }
                        break;
                }

                Object[] rowObjects = new Object[r.schema().fields().length];
                for (int i = 0; i < r.schema().fields().length; i++)
                {
                    if (r.schema().fields()[i].name().equals(attrID + AcDataSet.STATUS) &&
                        !r.isNullAt(i))
                    {
                        rowObjects[i] = STATUS_SUSPECT;
                    }
                    else if (r.schema().fields()[i].name().equals(attrID + AcDataSet.VALUE) &&
                        !r.isNullAt(i) &&
                        r.schema().fields()[i].dataType().typeName().equals("double"))
                    {
                        rowObjects[i] = newVal;
                    }
                    else
                    {
                        rowObjects[i] = r.get(i);
                    }
                }

                r = RowFactory.create(rowObjects);
            }
            return r;
        });

        return session.createDataFrame(dfRdd, df.schema());
    }

    public Dataset<Row> generateSuspectsAskBidMid(Dataset<Row> df, SparkSession session)
    {
        Random rand = new Random();
        return generateSuspectsAskBidMid(df, rand.nextInt(), session);
    }

    public Dataset<Row> generateSuspectsAskBidMid(Dataset<Row> df, int seed, SparkSession session)
    {
        final double sRatio = suspectRatio;
        final double sMagnitude = suspectMagnitude;

        final Random rand = new Random(seed);
        long count = df.count();

        List<String> attributes = new ArrayList<>();
        attributes.add("re_ua_ask");
        attributes.add("re_ua_bid");
        attributes.add("re_ua_mid");

        JavaRDD<Row> dfRdd = df.javaRDD().map(r ->
        {
            if (rand.nextInt((int) count) < count * sRatio)
            {
                List<Double> attrVals = new ArrayList<>();
                for (String attribute : attributes)
                {
                    if (!r.isNullAt(r.fieldIndex(attribute + AcDataSet.VALUE)))
                    {
                        attrVals.add(r.getDouble(r.fieldIndex(attribute + AcDataSet.VALUE)));
                    }
                }

                if (attrVals.size() == 0)
                {
                    return r;
                }

                Collections.sort(attrVals);

                String attrID = attributes.get(rand.nextInt(attributes.size()));
                double minVal = attrVals.get(0);
                double maxVal = attrVals.get(attrVals.size() - 1);
                double newVal  = (0.1 + (rand.nextDouble() * 0.9)) * (maxVal - minVal);
                newVal *= rand.nextInt(2) == 0 ? -sMagnitude : sMagnitude;

                Object[] rowObjects = new Object[r.schema().fields().length];
                for (int i = 0; i < r.schema().fields().length; i++)
                {
                    if (r.schema().fields()[i].name().equals(attrID + AcDataSet.STATUS) &&
                        !r.isNullAt(i))
                    {
                        rowObjects[i] = minVal == maxVal ? r.get(i) : STATUS_SUSPECT;
                    }
                    else if (r.schema().fields()[i].name().equals(attrID + AcDataSet.VALUE) &&
                        !r.isNullAt(i) &&
                        r.schema().fields()[i].dataType().typeName().equals("double"))
                    {
                        rowObjects[i] = r.getDouble(i) + newVal;
                    }
                    else
                    {
                        rowObjects[i] = r.get(i);
                    }
                }

                r = RowFactory.create(rowObjects);
            }
            return r;
        });

        return session.createDataFrame(dfRdd, df.schema());
    }

    public ReutersSuspectGenerator()
    {
        suspectMagnitude = 2.0;
        suspectRatio = 0.25;
    }

    public ReutersSuspectGenerator setSuspectMagnitude(double sm)
    {
        suspectMagnitude = sm;
        return this;
    }

    public ReutersSuspectGenerator setSuspectRatio(double sr)
    {
        suspectMagnitude = sr;
        return this;
    }
}
