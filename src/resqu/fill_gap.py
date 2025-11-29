import utils
import numpy as np
import pandas as pd
from traffic.core import Flight

def fill_gap(gap, df_flight, phase_beginning, phase_end, max_altitude):
    time_gap_minutes = int(gap.dt.iloc[-1] / 60.0)

    to_be_inserted = pd.DataFrame(
        data=np.nan,
        index=[np.nan] * time_gap_minutes,
        columns=df_flight.columns
    )

    if phase_beginning ==  phase_end:
        return None # do nothing, resampling will handle it

    elif phase_beginning == "ascending" and phase_end == "cruising":
        # if gap.index[0] > 0:
        #     point_before = df_flight.iloc[gap.index[0]-1]
        #     get the last VR and extrapolate with damping
        if gap.altitude.iloc[-1] > gap.altitude.iloc[0]:
            alt_diff = gap.altitude.iloc[-1] - gap.altitude.iloc[0]
            time_diff_min = gap.dt.iloc[-1] / 60
            climb_rate = max(850, alt_diff/time_diff_min)
            n_ascend = int(alt_diff / climb_rate)
            if time_gap_minutes < n_ascend:
                n_ascend = time_gap_minutes
                print(f"Warning: {n_ascend} ascend points exceed time gap of {time_gap_minutes} minutes.")
            n_cruise = time_gap_minutes - n_ascend
            ascend_altitudes = np.linspace(
                gap.altitude.iloc[0], gap.altitude.iloc[0] + n_ascend * climb_rate, num=n_ascend, endpoint=False
            )
            ascend_vr = np.ones(n_ascend) * climb_rate
            cruise_altitudes = np.full(n_cruise, gap.altitude.iloc[-1])
            cruise_vr = np.zeros(n_cruise)
            to_be_inserted["altitude"] = np.concatenate([ascend_altitudes, cruise_altitudes])
            to_be_inserted["vertical_rate"] = np.concatenate([ascend_vr, cruise_vr])

            to_be_inserted["timestamp"] = pd.date_range(
                start=gap.timestamp.iloc[0] + pd.Timedelta(minutes=1),
                periods=time_gap_minutes,
                freq='min'
            )
        else:
            to_be_inserted["altitude"] = np.linspace(
                gap.altitude.iloc[0], gap.altitude.iloc[-1], num=time_gap_minutes
            )
            to_be_inserted["vertical_rate"] = np.ones(time_gap_minutes) * (gap.altitude.iloc[-1] - gap.altitude.iloc[0]) / time_gap_minutes
            to_be_inserted["timestamp"] = pd.date_range(
                start=gap.timestamp.iloc[0] + pd.Timedelta(minutes=1),
                periods=time_gap_minutes,
                freq='min'
            )


    elif phase_beginning == "ascending" and phase_end == "descending":
        asce_alt_diff = max_altitude - gap.altitude.iloc[0]
        desc_alt_diff = max_altitude - gap.altitude.iloc[-1]
        n_ascend = int(asce_alt_diff / 850) # 850 ft per minute climb rate
        n_descend = int(desc_alt_diff / 850) # 850 ft per minute descent rate
        total_needed = n_ascend + n_descend
        if total_needed <= time_gap_minutes:
            n_cruise = time_gap_minutes - total_needed
        else:
            scale = time_gap_minutes / total_needed
            n_ascend = int(n_ascend * scale)
            n_descend = time_gap_minutes - n_ascend
            n_cruise = 0 # no cruise phase
        ascend_altitudes = np.linspace(
            gap.altitude.iloc[0], gap.altitude.iloc[0] + n_ascend * 850, num=n_ascend, endpoint=False
        )
        cruise_altitudes = np.full(n_cruise, max_altitude)
        descend_altitudes = np.linspace(
            max_altitude, gap.altitude.iloc[-1], num=n_descend, endpoint=False
        )
        to_be_inserted["altitude"] = np.concatenate([ascend_altitudes, cruise_altitudes, descend_altitudes])
        to_be_inserted["timestamp"] = pd.date_range(
            start=gap.timestamp.iloc[0] + pd.Timedelta(minutes=1),
            periods=time_gap_minutes,
            freq='min'
        )

    elif phase_beginning == "cruising" and phase_end == "descending":
        if gap.altitude.iloc[0] > gap.altitude.iloc[-1]:
            alt_diff = gap.altitude.iloc[0] - gap.altitude.iloc[-1]
            time_diff_min = gap.dt.iloc[-1] / 60
            descend_rate = max(850, alt_diff/time_diff_min)
            n_descend = int(alt_diff / descend_rate) # 850 ft per minute descent rate
            n_descend = min(n_descend, time_gap_minutes)
            n_cruise = time_gap_minutes - n_descend
            cruise_altitudes = np.full(n_cruise, gap.altitude.iloc[0])
            descend_altitudes = np.linspace(
                gap.altitude.iloc[0], gap.altitude.iloc[0] - n_descend * descend_rate, num=n_descend, endpoint=False
            )
            cruise_altitudes = np.full(n_cruise, gap.altitude.iloc[0])
            to_be_inserted["altitude"] = np.concatenate([cruise_altitudes, descend_altitudes])

            to_be_inserted["timestamp"] = pd.date_range(
                start=gap.timestamp.iloc[0] + pd.Timedelta(minutes=1),
                periods=time_gap_minutes,
                freq='min'
            )
        else:
            to_be_inserted["altitude"] = np.linspace(
                gap.altitude.iloc[0], gap.altitude.iloc[-1], num=time_gap_minutes
            )
            to_be_inserted["timestamp"] = pd.date_range(
                start=gap.timestamp.iloc[0] + pd.Timedelta(minutes=1),
                periods=time_gap_minutes,
                freq='min'
            )
    else: # do nothing when phase label is out of order (cruis, ascend), (descend, cruise) or (descend, ascend)
        return None
    return to_be_inserted

def fix_alt(flight):
    """
    Assume the dataframe is already appended with airport data
    """
    df = flight.data
    dt = df.timestamp.diff().dt.total_seconds()
    df = df.assign(dt=dt)
    takeoff_time = df.timestamp.iloc[0]
    landed_time = df.timestamp.iloc[-1]
    segments_processed_tofill = []
    to_fill = df[dt > 1000]
    max_altitude = 35000 if df.altitude.max() < 3000 else df.altitude.max()

    for gaps_index in to_fill.index:
        gap = df.loc[gaps_index-1 : gaps_index]
        beginning = gap.iloc[0]
        end = gap.iloc[1]

        if beginning.altitude < 30000:
            beginning_time_flown = (beginning.timestamp - takeoff_time).total_seconds()
            beginning_time_to_land = (landed_time - beginning.timestamp).total_seconds()
            
            if beginning_time_flown < 50*60 and beginning_time_to_land > beginning_time_flown:
                phase_beginning = "ascending"
            elif beginning_time_to_land < 50*60 and beginning_time_to_land < beginning_time_flown:
                phase_beginning = "descending"
            else:
                phase_beginning = "cruising"
        else:
            phase_beginning = "cruising"

        if end.altitude < 30000: 
            ending_time_flown = (end.timestamp - takeoff_time).total_seconds()
            ending_time_to_land = (landed_time - end.timestamp).total_seconds()

            if ending_time_flown < 50*60 and ending_time_to_land > ending_time_flown:
                phase_end = "ascending"
            elif ending_time_to_land < 50*60 and ending_time_to_land < ending_time_flown:
                phase_end = "descending"
            else:   
                phase_end = "cruising"
        else:
            phase_end = "cruising"
        filled = fill_gap(gap, df, phase_beginning, phase_end, max_altitude)
        if filled is not None:
            segments_processed_tofill.append(fill_gap(gap, df, phase_beginning, phase_end, max_altitude))
    df = pd.concat([df, *segments_processed_tofill], ignore_index=True).sort_values(by='timestamp').reset_index(drop=True).drop(columns=['dt'])
    return Flight(df)