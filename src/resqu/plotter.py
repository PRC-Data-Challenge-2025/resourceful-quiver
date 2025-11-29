import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from metadata import rank_traj_dir, rank_fuel_segs_by_id, flight_list_rank, airport_data, train_traj_dir, train_fuel_segs_by_id, flight_list_train
import resqu.data as data
from resqu.data import FlightSet

from sklearn.metrics import root_mean_squared_error as rmse, r2_score


def vis_flight(f_df, fuel_window=None, plot_source=False):
    fid = f_df.flight_id.iloc[0]
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{fid}")
    fig.subplots_adjust(hspace=0.3)
    axs = fig.subplots(2, 3)

    for ax in axs.flatten():
        ax.tick_params(axis="x", labelrotation=35)
        if fuel_window is not None:
            for start, end in zip(fuel_window.start, fuel_window.end):
                ax.axvspan(
                    start, end, alpha=0.1, facecolor="y", ec="g", lw=0.5, ls="--"
                )

    axs[0, 0].set_title("altitude")
    axs[0, 1].set_title("latitude")
    axs[0, 2].set_title("longitude")
    axs[1, 0].set_title("vertical_rate")
    axs[1, 1].set_title("speed")
    axs[1, 2].set_title("track")

    if plot_source:
        known_source = ["adsb", "acars"]
        adsb_mask = f_df.source == "adsb"
        acars_mask = f_df.source == "acars"
        others_mask = ~f_df.source.isin(known_source)
        axs[0, 0].scatter(
            f_df.timestamp[adsb_mask], f_df.altitude[adsb_mask], s=0.5, alpha=0.5
        )
        axs[0, 0].scatter(
            f_df.timestamp[acars_mask], f_df.altitude[acars_mask], s=5, color="r"
        )
        axs[0, 0].scatter(
            f_df.timestamp[others_mask], f_df.altitude[others_mask], s=1, color="g"
        )

        axs[0, 1].scatter(
            f_df.timestamp[adsb_mask], f_df.latitude[adsb_mask], s=0.5, alpha=0.5
        )
        axs[0, 1].scatter(
            f_df.timestamp[acars_mask], f_df.latitude[acars_mask], s=5, color="r"
        )
        axs[0, 1].scatter(
            f_df.timestamp[others_mask], f_df.latitude[others_mask], s=1, color="g"
        )

        axs[0, 2].scatter(
            f_df.timestamp[adsb_mask], f_df.longitude[adsb_mask], s=0.5, alpha=0.5
        )
        axs[0, 2].scatter(
            f_df.timestamp[acars_mask], f_df.longitude[acars_mask], s=5, color="r"
        )
        axs[0, 2].scatter(
            f_df.timestamp[others_mask], f_df.longitude[others_mask], s=1, color="g"
        )

        axs[1, 0].scatter(
            f_df.timestamp[adsb_mask], f_df.vertical_rate[adsb_mask], s=0.5, alpha=0.5
        )
        axs[1, 0].scatter(
            f_df.timestamp[acars_mask], f_df.vertical_rate[acars_mask], s=5, color="r"
        )
        axs[1, 0].scatter(
            f_df.timestamp[others_mask], f_df.vertical_rate[others_mask], s=1, color="g"
        )

        axs[1, 1].scatter(
            f_df.timestamp[adsb_mask], f_df.groundspeed[adsb_mask], s=0.5, alpha=0.5
        )
        axs[1, 1].scatter(
            f_df.timestamp[acars_mask], f_df.groundspeed[acars_mask], s=5, color="r"
        )
        axs[1, 1].scatter(
            f_df.timestamp[others_mask], f_df.groundspeed[others_mask], s=1, color="g"
        )
        axs[1, 1].scatter(f_df.timestamp, f_df.TAS, s=1, color="orange", alpha=0.5)

        axs[1, 2].scatter(
            f_df.timestamp[adsb_mask], f_df.track[adsb_mask], s=0.5, alpha=0.5
        )
        axs[1, 2].scatter(
            f_df.timestamp[acars_mask], f_df.track[acars_mask], s=5, color="r"
        )
        axs[1, 2].scatter(
            f_df.timestamp[others_mask], f_df.track[others_mask], s=1, color="g"
        )
    else:
        axs[0, 0].scatter(f_df.timestamp, f_df.altitude, s=0.5, alpha=0.5)
        axs[0, 1].scatter(f_df.timestamp, f_df.latitude, s=0.5, alpha=0.5)
        axs[0, 2].scatter(f_df.timestamp, f_df.longitude, s=0.5, alpha=0.5)
        axs[1, 0].scatter(f_df.timestamp, f_df.vertical_rate, s=0.5, alpha=0.5)
        axs[1, 1].scatter(f_df.timestamp, f_df.groundspeed, s=0.5, alpha=0.5)
        axs[1, 2].scatter(f_df.timestamp, f_df.track, s=0.5, alpha=0.5)
    return fig, axs


def vis_rank_flight(f, include_airport=True, plot_source=True):
    if type(f) is str:
        fid = f
        f_df = data.get_raw_df(fid)
    elif type(f) is pd.DataFrame:
        fid = f.flight_id.iloc[0]
        f_df = f

    window = data.get_fuel_id(fid, FlightSet.RANK)
    fig, axs = vis_flight(f_df, window, plot_source=plot_source)

    if include_airport:
        f_info = data.get_flist_id(fid, FlightSet.RANK)
        apt_o = data.airport_data[data.airport_data.icao == f_info.origin_icao]
        apt_d = data.airport_data[
            data.airport_data.icao == f_info.destination_icao
        ]
        t_takeoff = f_info.takeoff
        t_landing = f_info.landed
        axs[0, 0].scatter(t_takeoff, apt_o.elevation, color="limegreen", marker="^")
        axs[0, 0].scatter(t_landing, apt_d.elevation, color="limegreen", marker="v")
        axs[0, 1].scatter(t_takeoff, apt_o.latitude, color="limegreen", marker="^")
        axs[0, 1].scatter(t_landing, apt_d.latitude, color="limegreen", marker="v")
        axs[0, 2].scatter(t_takeoff, apt_o.longitude, color="limegreen", marker="^")
        axs[0, 2].scatter(t_landing, apt_d.longitude, color="limegreen", marker="v")
    return fig, axs


def vis_train_flight(f, include_airport=True, plot_source=True):
    if type(f) is str:
        fid = f
        f_df = data.get_raw_df(fid, FlightSet.TRAIN)
    elif type(f) is pd.DataFrame:
        fid = f.flight_id.iloc[0]
        f_df = f

    window = data.get_fuel_id(fid, FlightSet.TRAIN)
    fig, axs = vis_flight(f_df, window, plot_source=plot_source)

    if include_airport:
        f_info = data.get_flist_id(fid, FlightSet.TRAIN)
        apt_o = data.airport_data[data.airport_data.icao == f_info.origin_icao]
        apt_d = data.airport_data[
            data.airport_data.icao == f_info.destination_icao
        ]
        t_takeoff = f_info.takeoff
        t_landing = f_info.landed
        axs[0, 0].scatter(t_takeoff, apt_o.elevation, color="limegreen", marker="^")
        axs[0, 0].scatter(t_landing, apt_d.elevation, color="limegreen", marker="v")
        axs[0, 1].scatter(t_takeoff, apt_o.latitude, color="limegreen", marker="^")
        axs[0, 1].scatter(t_landing, apt_d.latitude, color="limegreen", marker="v")
        axs[0, 2].scatter(t_takeoff, apt_o.longitude, color="limegreen", marker="^")
        axs[0, 2].scatter(t_landing, apt_d.longitude, color="limegreen", marker="v")
    return fig, axs


def vis_final_flight(f, include_airport=True, plot_source=True):
    if type(f) is str:
        fid = f
        f_df = data.get_raw_df(fid, FlightSet.FINAL)
    elif type(f) is pd.DataFrame:
        fid = f.flight_id.iloc[0]
        f_df = f

    window = data.get_fuel_id(fid, FlightSet.FINAL)
    fig, axs = vis_flight(f_df, window, plot_source=plot_source)

    if include_airport:
        f_info = data.get_flist_id(fid, FlightSet.FINAL)
        apt_o = data.airport_data[data.airport_data.icao == f_info.origin_icao]
        apt_d = data.airport_data[
            data.airport_data.icao == f_info.destination_icao
        ]
        t_takeoff = f_info.takeoff
        t_landing = f_info.landed
        axs[0, 0].scatter(t_takeoff, apt_o.elevation, color="limegreen", marker="^")
        axs[0, 0].scatter(t_landing, apt_d.elevation, color="limegreen", marker="v")
        axs[0, 1].scatter(t_takeoff, apt_o.latitude, color="limegreen", marker="^")
        axs[0, 1].scatter(t_landing, apt_d.latitude, color="limegreen", marker="v")
        axs[0, 2].scatter(t_takeoff, apt_o.longitude, color="limegreen", marker="^")
        axs[0, 2].scatter(t_landing, apt_d.longitude, color="limegreen", marker="v")
    return fig, axs


def vis_flight_widget(fid):
    fset = FlightSet.get_set(fid)
    if fset == FlightSet.TRAIN:
        fig, axs = vis_train_flight(fid)
        fig.canvas.draw_idle()
    elif fset == FlightSet.RANK:
        fig, axs = vis_rank_flight(fid)
        fig.canvas.draw_idle()
    elif fset== FlightSet.FINAL:
        fig, axs = vis_final_flight(fid)
        fig.canvas.draw_idle()
    else:
        pass


def plot_pred_result(pred, true, desc=None, bins=100, ax=None, scatter=False):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
    # ax = plt.gca()
    ax.grid(True)

    if scatter:
        ax.scatter(
            pred,
            true,
            c="m",
            s=5,
            alpha=0.2,
        )
    else:
        h = ax.hist2d(
            pred, true, bins=bins, density=False, alpha=0.99, norm="log", cmap="PiYG"
        )
        ax.figure.colorbar(h[3])
        ax.set_facecolor("gainsboro")

    ax.axline([0, 0], slope=1.0, color="tab:red", label="error = 0")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Real", rotation=0, ha="left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_label_coords(-0.095, 1.02)

    metric_text = f"RMSE: {rmse(true, pred):4f}\nR2: {r2_score(true, pred):4f}"

    title = "Prediction Historgram"
    if desc is not None:
        title = title + f": {desc}"
    ax.set_title(title + f"\n{metric_text}")
    ax.set_aspect("equal")
    return ax
