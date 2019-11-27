rm(list = ls())
library(Mcomp)
library(ggplot2)
library(readxl)
library(sarima)
library(tseries)
library(TSPred)
library(dplyr)
library(xts)
library(uroot)
library(dplyr)
library(lubridate)
library(anytime)
library(TSstudio)
Crime_count <- read.csv("C:/Users/sohel/Downloads/Crime_count.csv",col.names= c('year_month','crime_Number'))
head(Crime_count)

Crime_count$year_month=anydate(Crime_count$year_month)
head(Crime_count)

#plotting time series data to see the trend

start_point <- c(year(min(Crime_count$year_month)), month(min(Crime_count$year_month)))
general_time_series <- ts(Crime_count$crime_Number, start = start_point,frequency= 12)
ts_plot(general_time_series,title = 'Plot of Chicago Crime data',Xtitle = 'Time',Ytitle = 'Number of Crime')

ts_seasonal(general_time_series, type = "all") # Seasonal plot

ts_heatmap(general_time_series) # Heatmap plot
ts_cor(general_time_series, lag.max = 60) # ACF and PACF plots
ts_lags(general_time_series, lags = 1:20) # lag plot
# Seasonal lags plot
ts_lags(general_time_series, lags = c(12, 24, 36, 48,60))

# Forecasting model

sample_data <- ts_split(ts.obj = general_time_series, sample.out = 12)
train <- sample_data$train
test <- sample_data$test

# Forecasting with auto.arima
library(forecast)
model <- auto.arima(train) # looking for good arima model
#some other residual/acf/pacf analysis
plot.ts(model$residuals)
acf(ts(model$residuals),main="ACF residuals plot") ###
pacf(ts(model$residuals),main="PACF residuals plot")

##some other test for the validity of the model (Ljung -Box test for different lags)
Box.test(model$residuals,lag=1,type="Ljung-Box")
Box.test(model$residuals,lag=5,type="Ljung-Box")
Box.test(model$residuals,lag=14,type="Ljung-Box")
fc <- forecast(model, h = 12)

# Plotting actual vs. fitted and forecasted
test_forecast(actual = general_time_series, forecast.obj = fc, test = test)
plot_forecast(fc)# Plotting the forecast 

# Run horse race between multiple models
methods <- list(ets1 = list(method = "ets",
                            method_arg = list(opt.crit = "lik"),
                            notes = "ETS model with opt.crit = lik"),
                ets2 = list(method = "ets",
                            method_arg = list(opt.crit = "amse"),
                            notes = "ETS model with opt.crit = amse"),
                arima = list(method = "arima",
                              method_arg = list(order = c(2,1,0)),
                              notes = "ARIMA(2,1,0)"),
                arima2 = list(method = "arima",
                              method_arg = list(order = c(1,1,1),
                                                seasonal = list(order = c(2,1,1))),
                              notes = "SARIMA(1,1,1)(2,1,1)"),
                hw = list(method = "HoltWinters",
                          method_arg = NULL,
                          notes = "HoltWinters Model"),
                tslm = list(method = "tslm",
                            method_arg = list(formula = input ~ trend + season),
                            notes = "tslm model with trend and seasonal components"))

# Training the models with backtesting
modified <- train_model(input = general_time_series,
                  methods = methods,
                  train_method = list(partitions = 6, 
                                      sample.out = 12, 
                                      space = 3),
                  horizon = 12,
                  error = "MAPE") #A Functional Approach for Building the train_model Components...look at r documentation

# Plot the performance of the different models on the testing partitions
plot_model(modified)

##another approach
backtesting=ts_backtesting(general_time_series, models = "abehntw", periods = 6,
                           error = "MAPE", window_size = 3, h = 3, plot = TRUE,
                           a.arg = NULL, b.arg = NULL, e.arg = NULL, h.arg = NULL,
                           n.arg = NULL, t.arg = NULL, w.arg = NULL, xreg.h = NULL,
                           parallel = FALSE)

backtesting$leaderboard# models leaderboard
backtesting$leadForecast$mean # best forecast results

backtesting$Forecast_Final$auto.arima$mean# final forecast of the auto.arima,nnetar model
backtesting$Forecast_Final$nnetar$mean
backtesting$period_1$auto.arima$forecast$mean#nnetar forecast during the first period of testing
backtesting$summary_plot# Get the final plot of the models performance and the selected forecasting model
check_res(backtesting_series$Models_Final$nnetar)#check the residuls of the best model
