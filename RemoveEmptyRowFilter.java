package com.ac.sds.ml;

import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.sql.Row;

public class RemoveEmptyRowFilter implements FilterFunction<Row>
{
    private String[] fields;

    public RemoveEmptyRowFilter(String... fields)
    {
        this.fields = fields;
    }

    @Override
    public boolean call(Row row) throws Exception
    {
        if (fields.length > 0)
        {
            for (String field : fields)
            {
                if (row.isNullAt(row.fieldIndex(field)))
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            for (int i = 0; i < row.size(); i++)
            {
                if (row.isNullAt(i))
                {
                    return false;
                }
            }
            return true;
        }
    }
}
