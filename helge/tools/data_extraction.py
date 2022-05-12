import netCDF4 as nc
import xarray as xr
import numpy as np
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def data_extraction(start_date, end_date, lead_time, dataset, num_input_time_steps, window=5):
    '''
    Args
    ----
    start_date (str): The start date for extraction. Important, put the trailing 0 at the beginning of year for dates before 1000 (e.g., '0400')
    end_date (str): The end date for extraction
    lead_time (int): The number of months between the predictor/predictand
    dataset (str): Either 'CESM' or 'ECMWF'
    num_input_time_steps (int): The number of time steps to use for each predictor samples
    '''    
    file_name = {'CESM': 'CESM_EA_SPI.nc', 'ECMWF': 'ECMWF_EA_SPI.nc'}[dataset]
    ds = xr.open_dataset(file_name)
    spi = ds['spi'].sel(time=slice(start_date,end_date))
    num_samples=spi.shape[0] 
    #Stack and remove nans
    spi = np.stack([spi.values[n-num_input_time_steps:n] for n in range(num_input_time_steps, num_samples+1)])
    num_samples = spi.shape[0]
    spi[np.isnan(spi)] = 0
    #make sure we have floats in there
    X = spi.astype(np.float32)
    # select Y
    if dataset == 'ECMWF':
        start_date_plus_lead = pd.to_datetime(start_date) + pd.DateOffset(months=lead_time+num_input_time_steps-1)
        end_date_plus_lead = pd.to_datetime(end_date) + pd.DateOffset(months=lead_time)
    elif dataset == 'CESM':
        t_start=datetime.datetime(int(start_date.split('-')[0]),int(start_date.split('-')[1]),int(start_date.split('-')[2]))
        t_end=datetime.datetime(int(end_date.split('-')[0]),int(end_date.split('-')[1]),int(end_date.split('-')[2]))
        start_date_plus_lead = t_start + relativedelta(months=lead_time+num_input_time_steps-1)
        end_date_plus_lead = t_end + relativedelta(months=lead_time)
        if len(str(start_date_plus_lead.year))<4:
            start_date_plus_lead = '0'+start_date_plus_lead.strftime('%Y-%m-%d')
        elif len(str(start_date_plus_lead.year))==4:
            start_date_plus_lead = start_date_plus_lead.strftime('%Y-%m-%d')
        if len(str(end_date_plus_lead.year))<4:
            end_date_plus_lead = '0'+end_date_plus_lead.strftime('%Y-%m-%d')
        elif len(str(end_date_plus_lead.year))==4:
            end_date_plus_lead = end_date_plus_lead.strftime('%Y-%m-%d')
    subsetted_ds = ds['spi'].sel(time=slice(start_date_plus_lead, end_date_plus_lead))
    y = subsetted_ds.values.astype(np.float32)
    y[np.isnan(y)] = 0
    # add month feature
    month = pd.DataFrame(subsetted_ds["time"].to_series().values, columns=['time'])
    month = month['time'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S').month).values
    fract = (month-1)/12*np.pi*2
    month_sin = np.sin(fract)
    month_cos = np.cos(fract)       
    ds.close()
    X = np.moveaxis(X, 1,3)
    orig_shape_X = X.shape
    orig_shape_y = y.shape
    #y = y.reshape(np.shape(y)[0]*np.shape(y)[1]*np.shape(y)[2])
    #X = X.reshape(np.shape(X)[0]*np.shape(X)[1]*np.shape(X)[2], np.shape(X)[3])

    data_set = []
    halfwindow = int(window/2)

    for time_idx in range(num_samples):
        img_y = y[time_idx,:,:]
        img_pad_y = np.pad(img_y, pad_width=window, mode='symmetric')
        
        img_x = X[time_idx,:,:,:]
        img_pad_x = np.ndarray((img_pad_y.shape[0], img_pad_y.shape[1], 3))
        img_pad_x[:,:,0] = np.pad(img_x[:,:,0], pad_width=window, mode='symmetric')
        img_pad_x[:,:,1] = np.pad(img_x[:,:,1], pad_width=window, mode='symmetric')
        img_pad_x[:,:,2] = np.pad(img_x[:,:,2], pad_width=window, mode='symmetric')
        for lat in range(window, 13+window):
            for lon in range(window, 20+window):

                # x
                sample = img_pad_x[lat-halfwindow:lat+halfwindow+1,lon-halfwindow:lon+halfwindow+1,:]
                sample = sample.reshape(np.shape(sample)[0]*np.shape(sample)[1]*np.shape(sample)[2])
                sample = np.append(sample, month_sin[time_idx])
                sample = np.append(sample, month_cos[time_idx])
                sample = np.append(sample, lat)
                sample = np.append(sample, lon)
                
                # y
                sample_y = img_pad_y[lat,lon]
                #sample_y = img_pad_y[lat:lat+window,lon:lon+window]
                #sample_y = sample_y.reshape(np.shape(sample_y)[0]*np.shape(sample_y)[1])
                #data_y.append(sample_y)                
                sample = np.append(sample, sample_y)
                data_set.append(sample)
        #break
    return np.array(data_set)