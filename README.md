# Stocks Prediction
> *This project is part of a series worked on as a part of my Udacity Nanodegree program. As this is the capstone project, I have built it from scratch.*

> **Machine Learning Engineer Nanodegree at Udacity** *sponsored by AWS*

[![powered by](https://forthebadge.com/images/badges/powered-by-water.svg)](https://forthebadge.com)

*Update November 1, 2020: I have submitted the project, but will be committing new changes till I have solved the [problem statment](#problem-statement).*

---

# Problem Statement

**The aim of this project is to predict the long-term price trend of one stock with at least 90% accuracy.<sup>1</sup>** Two models' performances will be compared: ARIMA and DeepAR. In the end, the ability of the DeepAR model to predict the next ten months worth of data is tested. More importantly, the model&#39;s ability to predict how the stocks performed in 2020 (the year of the pandemic) will be observed. There are multiple options when it comes to time series forecasting and I have decided to do a univariate prediction. As this is the first time I am independently working on time series data, it is important I keep things as simple as possible. Hence open, high, low, close prices will not be predicted, only adjusted close price will used for the prediction.

In the end, the model should be able to able to predict a general trend of the time series. One particular graph that will give a visual indication of the solution being reached is the quantiles graph. The aim is to use 30-70% quantiles of the predictions to encompass the true time series.

_<sup>1</sup> 90% accuracy can be taken as 10% mean absolute percentage error._

# SETUP
The project has been built using [Amazon SageMaker](http://aws.amazon.com/sagemaker/). Some details of the software and libraries used in the project is provided below.

> If you want to run the project, ensure you have the required [service instances limits](#service-limits-required) for you account.
1. While creating a notebook instance, the current [GitHub Repository was cloned](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-create.html). The version controlled project was then mamaged using Jupyter Lab's Git extension.
2. A SageMaker Notebook instance of type `ml.t2.xlarge` was used for to write and run code in the Jupyter environment. However, instance type `ml.t2.medium` should work perfectly fine to run the code.
3. Most of the libraries required for the project were present in the `conda_amazonei_mxnet_p36` notebook kernel.
4. Only `mplfinance` [library](https://github.com/matplotlib/mplfinance) was installed using `pip` package manager. *Note: the code for the installation is present in the Jupyter notebook and will not have to be installed separately. Just ensure the first two points are met.*

## Service Limits Required
Service limits are present to ensure a user does not use higher compute instance than required and gets a billed for a large amount. You will require the following instance types to run the project on SageMaker:
> You can know more about [service quotas here](https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html). Almost all the instances required usually have a default limit greater than 1. Hence no need to raise limit.
1. Notebook Instances: Either one.
  - `ml.t2.medium`
  - **Optional**: `ml.t2.xlarge` - this can be used for more processing power, not required. Default is `0`. Will require you to put in a request to raise the limit.
2. Training Instances: Either one.
  - `ml.m4.xlarge`
  - `ml.c4.xlarge`
  - **Optional**: `ml.p2.xlarge`: - this can be used for more processing power (CPU + GPU), not required. Default is `0`. Will require you to put in a request to raise the limit.
3. Endpoint Instances: Either one.
  - `ml.t2.medium`
  - `ml.m4.xlarge`
