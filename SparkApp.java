package com.ac.sds.spark;

import org.apache.spark.api.java.JavaSparkContext;

import java.io.Serializable;
import java.util.Properties;

public abstract class SparkApp<T> implements Serializable
{

    private final String name;

    private final Properties properties;

    public SparkApp(String name, Properties properties)
    {
        this.name = name;
        this.properties = properties;
    }

    public String getName()
    {
        return name;
    }

    public T submit() throws Exception
    {

        JavaSparkContext context = getContext();
        try
        {
            return task(context);
        }
        finally
        {
            context.stop();
        }
    }

    protected JavaSparkContext getContext()
    {
        return new JavaSparkContext(SparkConfig.create(this, this.properties));
    }

    public abstract T task(JavaSparkContext context) throws Exception;

}
