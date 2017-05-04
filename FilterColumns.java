package com.ac.sds.ml;

import com.ac.sds.data.AcDataSet;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructField;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.col;

public class FilterColumns
{
    private static final Logger logger = Logger.getLogger(FilterColumns.class);

    public static Dataset<Row> filterIncompleteColumns(Dataset<Row> df, double nullPercentageThreshold)
    {
        if (df.limit(1).javaRDD().isEmpty())
        {
            logger.warn("Empty dataframe was loaded from database. No columns can be filtered.");
            return df;
        }

        //stddev of 0:
        List<String> toFilterOnStddev = new ArrayList<>();
        Dataset<Row> descTable = df.describe();
        StructField[] descFields = descTable.schema().fields();
        Row stddevs = descTable.filter(r -> r.getString(r.fieldIndex("summary")).equals("stddev")).first();
        logger.debug("The following columns were dropped because they have a standard deviation of 0.0:");
        for (StructField descField : descFields)
        {

            if (!descField.name().equals("summary")
                && descField.name().endsWith(AcDataSet.VALUE)
                && !stddevs.isNullAt(stddevs.fieldIndex(descField.name()))
                && stddevs.getString(stddevs.fieldIndex(descField.name())).equals("0.0"))
            {
                logger.debug(descField.name());
                toFilterOnStddev.add(descField.name());
            }
        }

        //null threshold:
        List<String> toFilterOnNulls = new ArrayList<>();
        final StructField[] allFields = df.schema().fields();
        Row nullsPerColumn = df
            .javaRDD()
            .map(row ->
            {
                Integer[] nullValues = new Integer[allFields.length];
                for (int i = 0; i < allFields.length; i++)
                {
                    nullValues[i] = row.isNullAt(i) ? 1 : 0;
                }
                return RowFactory.create(nullValues);
            }
        ).reduce((row1, row2) ->
            {
                Integer[] totalNullValues = new Integer[allFields.length];
                for (int i = 0; i < allFields.length; i++)
                {
                    totalNullValues[i] = row1.getInt(i) + row2.getInt(i);
                }
                return RowFactory.create(totalNullValues);
            }
        );

        long numEntries = df.rdd().count();
        logger.debug("The following columns were dropped because they exceed the null value threshold percentage of "
            + (nullPercentageThreshold * 100.0) + "%:");
        for (int i = 0; i < allFields.length; i++)
        {
            if ((double)nullsPerColumn.getInt(i) / (double)numEntries > nullPercentageThreshold)
            {
                logger.debug(allFields[i].name());
                toFilterOnNulls.add(allFields[i].name());
            }
        }

        List<String> toFilter = toFilterOnStddev;
        toFilter.addAll(toFilterOnNulls);

        return dropColumns(df, toFilter.toArray(new String[toFilter.size()]));
    }

    public static Dataset<Row> dropColumns(Dataset<Row> df, String[] fields)
    {
        return df.drop(fields);
    }

    public static Dataset<Row> selectColumns(Dataset<Row> df, String[] fields)
    {
        Column[] cols = new Column[fields.length];
        for (int i = 0; i < fields.length; i++)
        {
            cols[i] = col(fields[i]);
        }

        return df.select(cols);
    }

    public static Dataset<Row> filterOnTypeStrict(Dataset<Row> df, String typeName)
    {
        List<String> toFilter = new ArrayList<>();
        StructField[] fields = df.schema().fields();
        for (StructField field : fields)
        {
            if (field.dataType().typeName().equals(typeName) && !toFilter.contains(field.name()))
            {
                toFilter.add(field.name());
            }
        }

        return selectColumns(df, toFilter.toArray(new String[toFilter.size()]));
    }

    public static Dataset<Row> filterOnType(Dataset<Row> df, String typeName)
    {
        List<String> toFilter = new ArrayList<>();
        StructField[] fields = df.schema().fields();
        for (StructField field : fields)
        {
            if (field.dataType().typeName().equals(typeName) && !toFilter.contains(field.name()))
            {
                toFilter.add(field.name());
            }
        }

        List<String> statuses = new ArrayList<>();
        for (String s : toFilter)
        {
            for (StructField field : fields)
            {
                if (field.name().startsWith(s.substring(0, 8)) && field.name().endsWith("_status"))
                {
                    statuses.add(field.name());
                    break;
                }
            }
        }

        toFilter.addAll(statuses);

        return selectColumns(df, toFilter.toArray(new String[toFilter.size()]));
    }
}
