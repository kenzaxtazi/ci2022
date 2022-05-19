import netCDF4 as nc
import xarray as xr
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def data_meanstd(window,ver)->None:
    
    fn_train = "./data/helge_TRAIN_dataset_window{}_sincostime_ver{}.nc".format(window,ver)


    import xarray as xr
    xr_dataset = xr.open_dataset(fn_train)

    coords = np.arange(xr_dataset.dims['input_dim'])

    xr_mean = xr.DataArray(xr_dataset.mean(dim='sample_dim').samples, 
                 coords=[coords], 
                 dims=["features"],
                 name="mean",
                 attrs={
                     "features":'[3*window^2+month_sin+month_cos+lat+lon+y]',
                      }) 

    xr_std = xr.DataArray(xr_dataset.std(dim='sample_dim').samples, 
                 coords=[coords], 
                 dims=["features"],
                 name="stddev",
                 attrs={
                     "features":'[3*window^2+month_sin+month_cos+lat+lon+y]',
                      }) 

    xr_dataset_mean_std = xr.merge( [xr_mean, xr_std], compat='override', combine_attrs='override', )
    print(xr_dataset_mean_std)


    fn = "./data/helge_MEAN_STDDEV_dataset_window{}_sincostime_ver{}.nc".format(window,ver)
    xr_dataset_mean_std.to_netcdf(fn)