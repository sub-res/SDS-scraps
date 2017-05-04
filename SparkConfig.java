package com.ac.sds.spark;

import com.google.common.collect.Sets;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.File;
import java.net.URL;
import java.util.Collections;
import java.util.Properties;
import java.util.Set;

import static com.ac.sds.SDSProperties.*;

/**
 * Creates Spark config from different sources.
 */
public class SparkConfig
{

    private static final Logger logger = Logger.getLogger(SparkConfig.class);

    public static SparkConf create(SparkApp task, Properties properties)
    {
        String host = properties.getProperty(PROPERTY_SPARK_MASTER_HOST);
        String port = properties.getProperty(PROPERTY_SPARK_MASTER_PORT);
        SparkConf config = new SparkConf()
            .setMaster("spark://" + host + ":" + port)
            .setAppName("Suspect Detection System: " + task.getName())
            .set("spark.cassandra.connection.host", host)
            .setJars(getJars(task));

        String memory = properties.getProperty(PROPERTY_SPARK_EXECUTOR_MEMORY);
        setConfigProperty(PROPERTY_SPARK_EXECUTOR_MEMORY, memory, config);

        String cores = properties.getProperty(PROPERTY_SPARK_EXECUTOR_CORES);
        setConfigProperty(PROPERTY_SPARK_EXECUTOR_CORES, cores, config);

        String maxcores = properties.getProperty(PROPERTY_SPARK_CORES_MAX);
        setConfigProperty(PROPERTY_SPARK_CORES_MAX, maxcores, config);

        return config;
    }

    private static void setConfigProperty(String property, String value, SparkConf config)
    {
        if (value != null)
        {
            config.set(property, value);
        }
        else
        {
            logger.debug("Property \"" + property + "\" is missing! Default value will be used.");
        }
    }

    private static String[] getJars(Object task)
    {
        String[] mainJars = jarsForClass(SparkApp.class);
        String[] taskJars = jarsForClass(task.getClass());
        Set<String> jars = Sets.newHashSet(mainJars);
        Collections.addAll(jars, taskJars);
        return jars.toArray(new String[jars.size()]);
    }

    private static String[] jarsForClass(Class<?> clazz)
    {
        URL resource = clazz.getResource(getSimpleName(clazz) + ".class");
        return resource.getPath().startsWith("jar:") ? JavaSparkContext.jarOfClass(clazz) : jarOfClass(clazz);
    }

    private static String getSimpleName(Class<?> clazz)
    {
        Class<?> enclosingClass = clazz.getEnclosingClass();
        return enclosingClass != null ? getSimpleName(enclosingClass) : clazz.getSimpleName();
    }

    private static String[] jarOfClass(Class<?> clazz)
    {
        File dir = new File(clazz
            .getProtectionDomain()
            .getCodeSource()
            .getLocation()
            .getPath())
            .getParentFile();

        String[] jars = dir.list((f, n) -> n.endsWith(".jar"));
        if (jars != null)
        {
            for (int i = 0; i < jars.length; i++)
            {
                jars[i] = dir.getAbsolutePath() + "/" + jars[i];
            }
            return jars;
        }
        return new String[0];
    }

}
