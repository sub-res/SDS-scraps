package com.ac.sds.ml;

import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.*;

public interface GridSearchParams
{
    ParamMap[] getParamGrid(DecisionTreeClassifier dtc);

    ParamMap[] getParamGrid(RandomForestClassifier rfc);

    ParamMap[] getParamGrid(GBTClassifier gbtc);

    ParamMap[] getParamGrid(MultilayerPerceptronClassifier mpc, int features, int classes);

    ParamMap[] getParamGrid(LogisticRegression logr);

    ParamMap[] getParamGrid(NaiveBayes nb);

    ParamMap[] getParamGrid(DecisionTreeRegressor dtr);

    ParamMap[] getParamGrid(RandomForestRegressor rfc);

    ParamMap[] getParamGrid(GBTRegressor gbtr);

    ParamMap[] getParamGrid(LinearRegression linr);

    ParamMap[] getParamGrid(GeneralizedLinearRegression glinr);

    ParamMap[] getParamGrid(IsotonicRegression ir);
}
