# Passes the ci2022 given raw data and generates a window-mlp-optimised dataset

def data_creation(data_set, window, ver, all_start_date, all_end_date, lead_time, climate_model, num_input_time_steps):
    def store_data(data_set, fn: str)->None:
        import xarray as xr
        sample_ticks = np.arange(data_set.shape[0])
        input_ticks = np.arange(data_set.shape[1])

        xr_drought = xr.DataArray(data_set, 
                     coords=[sample_ticks, input_ticks], 
                     dims=["sample_dim", "input_dim"],
                     name="samples",
                     attrs={"begin":all_start_date,
                            "end":all_end_date,
                            "climate_model":climate_model, 
                            "num_input_time_steps":num_input_time_steps,
                            "lead_time":lead_time,
                            "unit":"Standard Precipitation Index (SPI)",
                          })

        xr_dataset = xr.merge( [xr_drought], compat='override' )
        print(xr_dataset)

        xr_dataset.to_netcdf(fn)

    # set seed
    seed = 1
    train_fract = 0.8
    import pytorch_lightning as pl
    pl.seed_everything(seed)

    # split in train and test data
    import numpy as np
    rng = np.random.default_rng(seed)
    msk = rng.random(data_set.shape[0]) < 0.8
    data_set_train = data_set[msk,:]
    data_set_test = data_set[~msk,:]

    fn = "helge_TRAIN_dataset_window{}_sincostime_ver{}.nc".format(window,ver)
    store_data(data_set_train, fn)

    fn = "helge_TEST_dataset_window{}_sincostime_ver{}.nc".format(window,ver)
    store_data(data_set_test, fn)