package com.ac.sds.ml;

import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.*;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import scala.collection.JavaConversions;

import java.util.Arrays;

public class NormalGridSearchParams implements GridSearchParams
{

    /*classifiers*/

    public ParamMap[] getParamGrid(DecisionTreeClassifier dtc)
    {
        return new ParamGridBuilder()
            .addGrid(dtc.minInfoGain(), new double[]{1.0, 0.5, 0.1, 0.0})
            .addGrid(dtc.impurity(), JavaConversions.asScalaBuffer(Arrays.asList("gini", "entropy")).toList())
            .addGrid(dtc.maxBins(), new int[]{32})
            .addGrid(dtc.maxDepth(), new int[]{3, 5, 7, 9})
            .build();
    }

    public ParamMap[] getParamGrid(RandomForestClassifier rfc)
    {
        return new ParamGridBuilder()
            .addGrid(rfc.minInfoGain(), new double[]{1.0, 0.5, 0.1, 0.0})
            .addGrid(rfc.impurity(), JavaConversions.asScalaBuffer(Arrays.asList("gini", "entropy")).toList())
            .addGrid(rfc.maxBins(), new int[]{32})
            .addGrid(rfc.maxDepth(), new int[]{3, 5, 7, 9})
            .addGrid(rfc.numTrees(), new int[]{5, 10, 15, 20})
            .build();
    }

    public ParamMap[] getParamGrid(GBTClassifier gbtc)
    {
        return new ParamGridBuilder()
            .addGrid(gbtc.minInfoGain(), new double[]{1.0, 0.5, 0.1, 0.0})
            .addGrid(gbtc.impurity(), JavaConversions.asScalaBuffer(Arrays.asList("gini", "entropy")).toList())
            .addGrid(gbtc.maxBins(), new int[]{32})
            .addGrid(gbtc.maxDepth(), new int[]{3, 5, 7, 9})
            .addGrid(gbtc.maxIter(), new int[]{20, 40, 60})
            .build();
    }

    public ParamMap[] getParamGrid(MultilayerPerceptronClassifier mpc, int features, int classes)
    {
        return new ParamGridBuilder()
            .addGrid(mpc.layers(), JavaConversions.asScalaBuffer(Arrays.asList(
                new int[]{features, 16, classes},
                new int[]{features, 32, classes},
                new int[]{features, 32, 16, classes},
                new int[]{features, 16, 32, classes},
                new int[]{features, 16, 32, 16, classes},
                new int[]{features, 32, 16, 32, classes}
            )).toList())
            .addGrid(mpc.maxIter(), new int[]{20, 40, 60})
            .addGrid(mpc.blockSize(), new int[]{128})
            .build();
    }

    public ParamMap[] getParamGrid(LogisticRegression logr)
    {
        return new ParamGridBuilder()
            .addGrid(logr.maxIter(), new int[]{25, 50, 75, 100})
            .addGrid(logr.elasticNetParam(), new double[]{0.0, 0.2, 0.4, 0.6, 0.8, 1.0})
            .addGrid(logr.regParam(), new double[]{0.0, 0.1, 0.01, 0.001})
            .addGrid(logr.fitIntercept())
            .build();
    }

    public ParamMap[] getParamGrid(NaiveBayes nb)
    {
        return new ParamGridBuilder()
            .addGrid(nb.smoothing(), new double[]{0.0, 1.0, 2.0})
            .build();
    }

    /*regressors*/

    public ParamMap[] getParamGrid(DecisionTreeRegressor dtr)
    {
        return new ParamGridBuilder()
            .addGrid(dtr.minInfoGain(), new double[]{1.0, 0.5, 0.1, 0.0})
            .addGrid(dtr.maxBins(), new int[]{32})
            .addGrid(dtr.maxDepth(), new int[]{3, 5, 7, 9})
            .build();
    }

    public ParamMap[] getParamGrid(RandomForestRegressor rfr)
    {
        return new ParamGridBuilder()
            .addGrid(rfr.minInfoGain(), new double[]{1.0, 0.5, 0.1, 0.0})
            .addGrid(rfr.maxBins(), new int[]{32})
            .addGrid(rfr.maxDepth(), new int[]{3, 5, 7, 9})
            .addGrid(rfr.numTrees(), new int[]{5, 10, 15, 20})
            .build();
    }

    public ParamMap[] getParamGrid(GBTRegressor gbtr)
    {
        return new ParamGridBuilder()
            .addGrid(gbtr.minInfoGain(), new double[]{1.0, 0.5, 0.1, 0.0})
            .addGrid(gbtr.maxBins(), new int[]{32})
            .addGrid(gbtr.maxDepth(), new int[]{3, 5, 7, 9})
            .addGrid(gbtr.maxIter(), new int[]{20, 40, 60})
            .build();
    }

    public ParamMap[] getParamGrid(LinearRegression linr)
    {
        return new ParamGridBuilder()
            .addGrid(linr.maxIter(), new int[]{25, 50, 75, 100})
            .addGrid(linr.elasticNetParam(), new double[]{0.0, 0.2, 0.4, 0.6, 0.8, 1.0})
            .addGrid(linr.regParam(), new double[]{0.0, 0.1, 0.01, 0.001})
            .addGrid(linr.fitIntercept())
            .build();
    }

    public ParamMap[] getParamGrid(GeneralizedLinearRegression glinr)
    {
        return new ParamGridBuilder()
            .addGrid(glinr.maxIter(), new int[]{25, 50, 75, 100})
            .addGrid(glinr.regParam(), new double[]{0.0, 0.1, 0.01, 0.001})
            .addGrid(glinr.family(), JavaConversions.asScalaBuffer(Arrays.asList("gaussian", "gamma")).toList())
            .addGrid(glinr.link(), JavaConversions.asScalaBuffer(Arrays.asList("identity", "log", "inverse")).toList())
            .addGrid(glinr.fitIntercept())
            .build();
    }

    public ParamMap[] getParamGrid(IsotonicRegression ir)
    {
        return new ParamGridBuilder()
            .addGrid(ir.isotonic())
            .build();
    }

}
