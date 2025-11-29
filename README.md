# resqu
Team Resourceful-Quiver 
<img src="resourceful-quiver-patch.png" alt="Team Resourceful-Quiver
" width="150" height="150">

## See our report here
[TODO]()

### Install
We recommend using a virtual environment to install the package
`pip install -e .`
or 
`uv pip install -e .`

<!-- ### cli
preprocessing
`resqu-preproc --help`

training
TODO -->

### Folder structure
.  
├── data  
├── models  
├── notebooks  
├── pyproject.toml  
├── README.md  
├── resourceful-quiver-patch.png  
├── scripts  
├── src  
└── WhatToDo.MD  

`data` directory should contain all files and folders downloaded from `prc-2025-datasets`  
`scripts` directory has some scripts to run the preprocessing pipeline and creating the model training dataframe

### Prepare dataset
#### Running preprocessing
activate your virtual environmnet, change directory to the project folder, e.g. `cd PRC-2025`

v0 pipeline: `python ./scripts/preproc_v0.py -j <number of parallel processes> -s <train | rank | final>`  
v0 pipeline with wind: `python ./scripts/preproc_v0_wind.py -j 20 -s <train | rank | final>` make sure to change the local-store of presynced fastmeteo data  
v2 pipeline: `python ./scripts/preproc_v2.py -j <number of parallel processes> -s <train | rank | final>`  
v2 pipeline with wind: `python ./scripts/preproc_v2_wind.py -j 20 -s <train | rank | final>` make sure to change the local-store of presynced fastmeteo data  
if there is any issue on how to run the scripts, use the `-h` flag for help

The best ranking score uses `preproc_v2_wind.py`  

#### Create LGBM dataset
After running preprocessing for all dataset (train, rank, final)
run `python ./scripts/create_lgbm_data.py -v <your choice of version>` 
3 parquet file should be saved to `data/lgbm_dataset/`  

### Training
TODO


### Extra stuff
`resqu.data` should have all the directories and metadata loaded  
`resqu.plotter` has some function to visualize the trajectory by flight_id or flight_df
all path to directories are `pathlib.Path`, which should make globing and getting specific file ergonomic  
This I suspects would be the most useful module for other tasks  
Also have methods for getting the raw flight, flight_list, and fuel data by `flight_id`  
Also have a `FlightSet` enum to specify the dataset `TRAIN`, `RANK`, or `FINAL`  
Example  
```
from resqu import data
from resqu import FlightSet

fuel_train_data = data.fuel_train_data # getting the fuel_train.parquet
flight_list_train = data.flight_list_train # getting the flightlist_train.parquet

fid = 'prc770822360'
df = data.get_raw_df(fid) # automatically infer the FlightSet
df = data.get_raw_df(fid, FlightSet.TRAIN) # does the same thing

# you can also pass in the FlightSet to be explicit
flight_info = data.get_flist_id(fid) # returns the row in flight list corresponds to the flight_id
fuel_data = data.get_fuel_id(fid) # grabs the rows in fuel data corresponds to the flight_id

# for plotting
from resqu import plotter
plotter.vis_flight(df, fuel_data) # by df and fuel_data
plotter.vis_train_flight(fid) # by flight_id, shows airport and fuel window
plotter.vis_train_flight(df) # by df, raw or processed, shows airport and fuel window

plotter.plot_pred_result(y_pred, y_true) # plots prediction histogram, with bins on logarithmic scale
```
Tips with `pathlib`
```
preproc_dir = data.data_dir / "flights_train_preproc"

# this should give you all the file names as a list, 
# you can index or iterate this later to grab specific fill for post processing
file_names = [f for f in preproc_dir.glob("*.parquet")] 
fids = [f.stem for f in preproc_dir.glob("*.parquet")] # if you only care about the flight_id

def do_some_stuff(file_names):
    ...
    return df_out

df_out = do_some_stuff(file_names)
out_dir = data.data_dir / "did_some_stuff"
out_dir.mkdir(exist_ok=True) # create the output directory, if it exists no exception will be thrown
df_out.to_parquet(out_dir / "df_out.parquet") # save to "data/did_some_stuff/df_out.parquet"
```