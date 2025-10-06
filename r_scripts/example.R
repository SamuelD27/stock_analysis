# Example R Script

# This script performs a simple exponential smoothing forecast using the
# forecast package.  It is intended to be called from Python via rpy2.

library(forecast)

forecast_exponential <- function(series, h = 30) {
  # series: numeric vector of time series data
  # h: forecast horizon
  model <- ets(series)
  f <- forecast(model, h = h)
  return(list(point_forecast = f$mean,
              lower_95 = f$lower[,2],
              upper_95 = f$upper[,2]))
}

"""
Use this function in Python via rpy2:

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
forecast_exponential = ro.r['forecast_exponential']
result = forecast_exponential(series, h=30)

"""