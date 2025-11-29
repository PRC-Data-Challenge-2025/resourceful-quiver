# resqu
Team Resourceful-Quiver

### install
`pip install -e .`
or 
`uv pip install -e .`

### cli
preprocessing
`resqu-preproc --help`

training
TODO

### folder structure
q

### tips
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