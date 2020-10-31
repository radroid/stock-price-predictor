# Stocks Prediction
> *This project is part of a series worked on as a part of my Udacity Nanodegree program. As this is the capstone project, I have built it from scratch.*

> **Machine Learning Engineer Nanodegree at Udacity** *sponsored by AWS*

[![powered by](https://forthebadge.com/images/badges/powered-by-water.svg)](https://forthebadge.com)

---

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Table of Figures](#table-of-figures)
- [Capstone Project Report](#capstone-project-report)
  - [Project Overview](#project-overview)
  - [Domain Background](#domain-background)
  - [Problem Statement](#problem-statement)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Data Exploration and Visualisation](#data-exploration-and-visualisation)
  - [Algorithms and Techniques](#algorithms-and-techniques)
  - [Benchmark Model](#benchmark-model)
  - [Data Anomalies and Implementation](#data-anomalies-and-implementation)
  - [Results](#results)
  - [References](#references)


# Table of Figures

[Figure 1: Example Stock Price vs Time graph with predicted values](images/metric-example-graph.png)


# Capstone Project Report

Raj Dholakia

October 31, 2020

## Project Overview
I started with the aim of predicting prices of three stocks and came down using one while I prepared data to be processed. I have finally narrowed it down to training two different models and comparing their performance on `AAPL` stock data (in particular it's daily adjusted close price).

This is my first independent project of this level and as my nature is, I took up a task rather ambituous to be completed on time. However, I have learnt many from the many mistakes I have made during this journey and intend to work on this project post my submission as part of the Machine Learning Nanodegree.

I decided to break down my project into three different parts:
1. Exploratory Data Analysis
2. Data Preparation
3. Model Training and Testing

I focused a lot on ensuring code is clean, graphs are neat and everything is understandable and modular. Everything went smoothly till I reached the Model training part. When I realised what my first mistake was: **ambitious goals**.

It is good to have ambitious goals, it is smarter to ensure they are realistic. I had filled my plate with everything I found interesting and now I did not have the time to finish it all. I had to cut down, which I did not do till the last minute. Now I am going to focus on the Rubric provided. Starting with the origin of this idea of working on stocks data in the next section: **Domain Background**.

> *Following three sections are almost the same as the ones mentioned in the project proposal.*

## Domain Background

An ideal real-world application of machine learning is in the world of trading and investing. It all works on predicting what will have to the price of a stock in the next few minutes or a few years. Hence, the aim is to **predict** and there is a **large amount of historical data** available to assist in making a prediction. Laying out the foundation to apply some machine learning algorithms.

Having said that, the field is yet part of active research. The uncertain nature of stock prices and the time-series aspect of the problem makes it a particularly difficult one to solve. Then again, considering the fact that the field is all about money, a lot research has already been done. Even a decent prediction model, attracts a lot of attention in the field (Asadi _et al._, 2012; Agarwal and Sabitha, 2017)

There are two broad ways one can go about predicting the price of a stock. The first is through **technical analysis**. The historical price data of a stock is used to predict the future price. The second being **fundamental analysis** , which uses unstructured textural information to understand the market sentiment and predict what will happen to the price of a stock. Information sources for the latter approach include news and social media (Subhi Alzazah and Cheng, 2020).

One of the researches cited above worked on &quot;_SENSEX&#39;s Index dataset since 2006 to 2016 for time series forecasting using Rapid miner_&quot; (Agarwal and Sabitha, 2017). This is particularly important because we will be using two indices and one stock in this project.

I decided to take on this problem due to my interest in finance and some internship experience at a start-up that focuses on machine learning applications in intraday trading. I believe working on this project will give me a better understanding of the financial markets (finance in general) and time-series machine learning problems. As I have new to the world of machine learning and finance, I have decided to carry out technical analysis using the historical data of stocks.

## Problem Statement

**The aim of this project is to predict the long-term price trend of one stock with at least 90% accuracy.<sup>1</sup>** Two models' performances will be compared: ARIMA and DeepAR. In the end, the ability of the DeepAR model to predict the next ten months worth of data is tested. More importantly, the model&#39;s ability to predict how the stocks performed in 2020 (the year of the pandemic) will be observed.

_<sup>1</sup> 90% accuracy can be taken as 10% mean absolute percentage error._

## Evaluation Metrics

I will be using two evaluation metrics to understand the model&#39;s performance. 
> One will be quantitative and the other will be visual (can be converted to quantitative):

1. **Mean Absolute Percentage Error (MAPE)**: It is the mean of percentage of absolute errors of the predictions. The following formula explains how it is calculated (&#39;MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)&#39;, 2006; Glen, 2011) :

2. **Percentage Points Correctly Predicted** : This is more of a visual indicator of how the model is doing. It is the percentage of actual points that lie in the 30-70 (shorter inter-quartile) range of the predictions.

![Example Graph](images/metric-example-graph.png)

In the example graph above, it is clear that there are 3 out of 5 points fall in the 30-70 quartile range. Hence,

I came up with this metric as a solution to the problem predicting for larger intervals. I intend to use this to be understand of the model can make accurate predictions on the long-term trends. However, some weakness of the metric would be its inability to give great results for predictions with high variability (standard deviation). As a high standard deviation would be a larger area is covered by the predictions, the probability of the actual value to land within the 30-70 range is higher. However, if the standard deviation is high, the model is not following any specific trend (up or down) but is just spreading in both directions, leading to an inaccurate measure of what is actually happening.

Hence, a combination of MAPE and Percentage Points will give a better understanding of how the model is performing.

> *The following sections report on reasoning and results of the project.*

## Data Exploration and Visualisation
> Notebook 1_Exploratory_Data_Analysis

The data was first loaded and features of the data understood and explained. This included meaning of `OHLC` prices and `Adj Close` price. The data for the three stock and/or indices were then looked visually inspected to determine the length of time series data to be taken. It was noticed that most of the data before `2002` was almost constant, compared to post `2002` for Apple Inc. Furthermore, the decision to use `Adj Close` for project was taken after understanding that multi-variate time series prediction will be beyond the scope of this project.


## Algorithms and Techniques
> Notebook 2_Data_Preparation

Data preparation required a better understanding how the two models: ARIMA and DeepAR. Not only was the working of the algorithm be understood, but also the varying formats in which both accept data were to be understood well. A section is dedicated to the description of the models. Data is then prepared for each of the model.

## Benchmark Model
The results obtained by Nagesh Singh Chauhan in his analysis of Altaba Inc. stock from _1996–04–12_ till _2017–11–10_ (Chauhan, no date). He managed to get a MAPE value of **3.5%**, which can be said to be **96.5% accuracy**, using a well-tuned **ARIMA model**. In this project, the goal will be to get the MAPE value to be less than 90%.

This is a very particular example and it could turn out of that the results obtained are not as expected. I will be identifying the shortcomings of the analysis.

## Data Anomalies and Implementation
One of the major concerns in the data is the missing data for bank holidays and weekends. This is common as stock markets will not be open on that day. As we have daily data, it is necessary that it is taken care of. In notebook 2_Data_Preparation, I had decided to keep missing data and let DeepAR algorithm handle it. However, when training a DeepAR model, an error kept preventing the data from being read. After numerous attempts at ensuring the data is in the correct `JSON Line` format, I decided to remove all the `Nan` values. After removal of all of these, the model trained. I realised the hard way that one of the sources might not have misguided me.

> Flunkert, V. et al. (2018) Amazon SageMaker DeepAR now supports missing values, categorical and time series features, and generalized frequencies | AWS Machine Learning Blog, Amazon SageMaker, Artificial Intelligence. Available at: https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-deepar-now-supports-missing-values-categorical-and-time-series-features-and-generalized-frequencies/ (Accessed: 29 October 2020).

## Results
There is a lot of work needed on this end of the project. From better data handling to improved model tuning, all aspects of the implementation need to be worked on. I have created multiple functions to prepare the data to be fed into the algorithms, to plot graphs to understand performance, evaluate a model's performance with the predictions. Currently, all the results obtained cannot be considered valid. ARIMA model yielded results show a MAPE of almost 100% and DeepAR model shows over 2000%. I believe the problem is in the way the data is being provided to the models: the indexing of the data. The data recieved from a predictor uses integers as an index and the indices in the training data are in DateTimeIndex format. If indices need to be in the same format, it will enable better understanding and smoother integration of real and predicted values.

Second improvement that can be made would be to find a way to manage missing data. One of the ways I have tried to get around it is by linear interpolation of the missing values. I could try to instead remove the missing data and observe how the model performs. As we are looking at a time series which does not have a very seasonal trend, the latter could work better.

Thirdly, the aim should be to get valid results from the models for now. Once valid predictions are being made, hypereparameter tuning can be done. For the ARIMA model, I started by trying to find the optimal value for lag without training and testing the model. All the methods used provide a tentative value of `p` and `q`. The optimal values can only be determined by modelling and testing.

Finally, more examples of time series analysis of stock prices needed to be looked at to understand if the benchmark model provides a realistic goal. This is to be ensure that the benchmark model is robust and similar results can be obtained for other stocks or indices. 

> Another note, all the functions can be neatly packed in a `helper_functions.py` file and accessed in the notebooks to make the notebooks more presentable.

## Final Words
I agree my goal had been ambitious from the beginning. However, I have worked hard to implement everything to the best of my ability for the time I had in my hand. I will continue giving time to this project to improve it and make it something presentable to other Aspiring Machine Learning Engineers. Working on this project has given me a better understanding of what kind of problems can a Machine Learning project face at a small level. It all starts by defining a problem statement and coming back to it now and then to improve it. Some exploratory data analysis can give new ideas and the problem statement can be tuned to take-in the concerns. Setting a methodology will ensure the focus of the project remains the same throughout. Results are the best place to learn about what could be improved in the data processing and implementation process (even if they are incomplete). Machine Learning is all about jumping multiple obstacles, reaching the finish line, coming back and evaluating the obstacles till we understand what is the best way to cross an obstacle.


## References

Agarwal, U. and Sabitha, A. S. (2017) &#39;Time series forecasting of stock market index&#39;, in _India International Conference on Information Processing, IICIP 2016 - Proceedings_. Institute of Electrical and Electronics Engineers Inc. doi: 10.1109/IICIP.2016.7975381.

Asadi, S. _et al._ (2012) &#39;Hybridization of evolutionary Levenberg-Marquardt neural networks and data pre-processing for stock market prediction&#39;, _Knowledge-Based Systems_. Elsevier, 35, pp. 245–258. doi: 10.1016/j.knosys.2012.05.003.

AWS (2019) _Machine Learning with Amazon SageMaker_, _Amazon Web Services, Inc._ Available at: https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-mlconcepts.html (Accessed: 7 October 2020).

Banton, C. (2019) _An Introduction to U.S. Stock Market Indexes_, _Investopedia_. Available at: https://www.investopedia.com/insights/introduction-to-stock-market-indices/ (Accessed: 4 October 2020).

Bourke, D. and Neagoie, A. (2020) _Complete Machine Learning and Data Science: Zero to Mastery | Udemy_, _Udemy_. Available at: https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/ (Accessed: 7 October 2020).

Chauhan, N. S. (no date) _Stock Market Forecasting Using Time Series Analysis_. Available at: https://www.kdnuggets.com/2020/01/stock-market-forecasting-time-series-analysis.html (Accessed: 8 October 2020).

Glen, S. (2011) &#39;MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)&#39;, in _SpringerReference_. doi: 10.1007/springerreference\_6919.

&#39;MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)&#39; (2006) in _Encyclopedia of Production and Manufacturing Management_. Springer US, pp. 462–462. doi: 10.1007/1-4020-0612-8\_580.

_Stock Market Index : Meaning, Importance, NSE &amp; BSE and more_ (2020) _Defmacro Software Pvt. Ltd._ Available at: https://cleartax.in/s/stock-market-index (Accessed: 4 October 2020).

Subhi Alzazah, F. and Cheng, X. (2020) &#39;Recent Advances in Stock Market Prediction Using Text Mining: A Survey&#39;, in _E-Business [Working Title]_. IntechOpen. doi: 10.5772/intechopen.92253.

_Yahoo Finance – stock market live, quotes, business &amp; finance news_ (no date). Available at: https://in.finance.yahoo.com/ (Accessed: 2 October 2020).
