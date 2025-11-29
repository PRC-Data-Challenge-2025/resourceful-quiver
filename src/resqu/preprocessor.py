import numpy as np
import pandas as pd
from traffic.core import Flight
import openap
from scipy.signal import savgol_filter
import warnings
import joblib
import functools

from tqdm.contrib.concurrent import process_map

from . import data
from .data import FlightSet

from .energy import Energy

import warnings

warnings.filterwarnings(action="ignore")
# from abc import ABC, abstractmethod


class FlightProc(Flight):
    """
    A child class of traffic.core.Flight, stores metadata about fuel window and flight list
    Provides a collection of preprocessing methods
    """

    def __init__(self, df, flight_set: FlightSet | None = None, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        self.flight_set = flight_set

        if flight_set is None:
            # warnings.warn(f"Did not provide a flight set")
            self.flight_set = FlightSet.get_set(self.flight_id)
        self.flight_info = data.get_flist_id(self.flight_id, self.flight_set)
        self.fuel_data = data.get_fuel_id(self.flight_id, self.flight_set)

    def add_airport(self):
        # do before resampling
        fid = self.flight_id
        f_info = self.flight_info
        apt_to = data.airport_data[
            data.airport_data.icao == self.flight_info.origin_icao
        ]
        apt_ld = data.airport_data[
            data.airport_data.icao == self.flight_info.destination_icao
        ]
        t_to = f_info.takeoff
        t_ld = f_info.landed
        row_to = {
            "timestamp": t_to,
            "flight_id": fid,
            "typecode": self.data.typecode[0],
            "latitude": apt_to.latitude.iloc[0],
            "longitude": apt_to.longitude.iloc[0],
            "altitude": apt_to.elevation.iloc[0],
            "groundspeed": np.nan,
            "track": np.nan,
            "vertical_rate": np.nan,
            "mach": np.nan,
            "TAS": np.nan,
            "CAS": np.nan,
            "source": "airport",
        }
        row_ld = {
            "timestamp": t_ld,
            "flight_id": fid,
            "typecode": self.data.typecode[0],
            "latitude": apt_ld.latitude.iloc[0],
            "longitude": apt_ld.longitude.iloc[0],
            "altitude": apt_ld.elevation.iloc[0],
            "groundspeed": np.nan,
            "track": np.nan,
            "vertical_rate": np.nan,
            "mach": np.nan,
            "TAS": np.nan,
            "CAS": np.nan,
            "source": "airport",
        }
        apt_df = pd.DataFrame([row_to, row_ld])
        self.data = (
            pd.concat([self.data, apt_df], ignore_index=True)
            .sort_values(by="timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )

        apt_idx = self.data.index[self.data.source == "airport"]

        self.ts_after_takeoff = self.data.timestamp.iloc[apt_idx[0] + 1]
        self.ts_before_landing = self.data.timestamp.iloc[apt_idx[1] - 1]
        return self

    def interp_lla(self, proj=None):
        # do before resampling
        if proj is None:
            proj = self.projection("lcc")
        self.data = self.compute_xy(proj).data
        xya_col = ["x", "y", "altitude"]
        df_proj_interp = self.data
        df_proj_interp.loc[:, xya_col] = (
            df_proj_interp.set_index("timestamp")[xya_col]
            .interpolate(method="time")
            .reset_index(drop=True)
            .bfill()
            .ffill()
        )
        self.data = (
            Flight(df_proj_interp)
            .compute_latlon_from_xy(proj)
            .drop(columns=["x", "y"])
            .data
        )
        return self

    def get_sparse_cluster(self):
        """
        get consecutive sparse clusters, where there are acars data
        returns the index of these clusters
        """
        cluster_idx = np.where(self.data.source != "adsb")[0]
        split_indices = np.where(np.diff(cluster_idx) != 1)[0] + 1
        clusters = [
            cluster
            for cluster in np.split(cluster_idx, split_indices)
            if len(cluster) > 1
        ]
        return clusters

    def assign_clusters(self):
        # do before resampling
        self.data = self.data.assign(cluster=None)
        max_dt = pd.Timedelta(minutes=5)
        mask = self.data.timestamp.diff() >= max_dt
        gap_idx = np.append(np.append(0, np.flatnonzero(mask)), len(mask))
        for i in range(len(gap_idx) - 1):
            start = gap_idx[i]
            end = gap_idx[i + 1]
            self.data.cluster.iloc[start:end] = i

        sparse_clusters = self.get_sparse_cluster()
        for j, idx in enumerate(sparse_clusters):
            self.data.cluster.iloc[idx] = j - len(sparse_clusters)
        self.data.cluster = self.data.cluster.astype("category")
        return self

    def estimate_missing_sparse_cluster(self):
        # do before resampling
        # estimate vertical rate, groundspeed, and track based on lla for sparse ACARS clusters
        # df = self.data
        subset_col = ["timestamp", "latitude", "longitude", "altitude"]

        clusters = self.get_sparse_cluster()

        # iterate through cluster and use position info to calculate speed and track
        for idx in clusters:
            sub_df = self.data.iloc[idx][subset_col]
            dt = sub_df.timestamp.diff().dt.total_seconds()
            compute_vs = (sub_df.altitude.diff() / (dt / 60.0)).bfill().ffill()
            compute_df = (
                Flight(sub_df)
                .cumulative_distance()
                .data[["compute_gs", "compute_track"]]
            )
            compute_track = compute_df.compute_track
            compute_gs = compute_df.compute_gs
            if len(compute_track) > 10:
                compute_track = savgol_filter(
                    compute_track, window_length=10, polyorder=2
                )
            self.data.loc[idx, "groundspeed"] = compute_gs
            self.data.loc[idx, "track"] = compute_track
            self.data.loc[idx, "vertical_rate"] = compute_vs
        return self

    def filter_by_cluster(self):
        clusters = self.data.cluster.unique()
        dfs = []

        for cluster in clusters:
            df = self.data[self.data.cluster == cluster]
            if cluster >= 0:
                df_filtered = (
                    Flight(df)
                    .filter(
                        altitude=17,
                        latitude=17,
                        longitude=17,
                        vertical_rate=3,
                        groundspeed=5,
                        track=17,
                    )
                    .data
                )
                dfs.append(df_filtered)
            else:
                dfs.append(df)
        self.data = pd.concat(dfs)
        return self

    def get_wind(self, meteo_grid):
        self.data = meteo_grid.interpolate(self.data)
        return self

    def ts(self):
        # self.data = (
        #     self.data.sort_values(by="timestamp")
        #     .assign(ts=(self.data.timestamp - self.data.timestamp.iloc[0]).dt.total_seconds())
        #     .drop_duplicates(subset=["ts"])
        # )
        return self.assign(
            ts=(self.data.timestamp - self.data.timestamp.iloc[0]).dt.total_seconds()
        )

    def tas_from_gs(self):
        tas = self.data.TAS.values
        tas[np.where(np.isnan(tas))] = self.data.groundspeed.values[
            np.where(np.isnan(tas))
        ]
        alt = self.data.altitude.values
        mach = openap.aero.tas2mach(tas * openap.aero.kts, alt * openap.aero.ft)
        cas = openap.aero.tas2cas(tas * openap.aero.kts, alt * openap.aero.ft)
        return self.assign(mach=mach, CAS=cas, TAS=tas)

    def tas_from_mach_cas(self):
        tas_orig = self.data.TAS.values
        cas_orig = self.data.CAS.values
        mach_orig = self.data.mach.values
        alt = self.data.altitude.values
        tas = tas_orig

        # fill tas with cas
        mask = np.isnan(tas_orig) & ~np.isnan(cas_orig) & (cas_orig < 600)
        tas[mask] = (
            openap.aero.cas2tas(
                cas_orig[mask] * openap.aero.kts, alt[mask] * openap.aero.ft
            )
            / openap.aero.kts
        )
        # fill tas with mach
        mask = np.isnan(tas_orig) & ~np.isnan(mach_orig)
        tas[mask] = (
            openap.aero.mach2tas(mach_orig[mask], alt[mask] * openap.aero.ft)
            / openap.aero.kts
        )
        # fill tas with gs
        tas[np.isnan(tas)] = self.data.groundspeed.values[np.isnan(tas)]

        # return self.assign(TAS=tas)
        # calc mach and cas (Maybe we don't care about them?)
        mach = openap.aero.tas2mach(tas * openap.aero.kts, alt * openap.aero.ft)
        cas = (
            openap.aero.tas2cas(tas * openap.aero.kts, alt * openap.aero.ft)
            / openap.aero.kts
        )
        return self.assign(mach=mach, CAS=cas, TAS=tas)

    def estimate_mass_24_chgpt(self):
        # Load model and data
        model_path = data.model_dir / "lgbm_mass_estim_24_feat.pkl"
        model = joblib.load(model_path)

        df = self.data.copy()

        # Compute vertical acceleration safely

        df["vertical_acc"] = df.vertical_rate.diff() / df.ts.diff() / 60

        # flight_id = df.flight_id.iloc[0]
        typecode = df.typecode.iloc[0]

        # Query subsets safely
        def safe_query(data, q):
            try:
                result = data.query(q)
                return (
                    result if not result.empty else pd.DataFrame(columns=data.columns)
                )
            except Exception:
                return pd.DataFrame(columns=data.columns)

        df_climb = safe_query(df, "phase=='CLIMB'")
        df_init = safe_query(df_climb, "altitude<10000")
        df_cruise = safe_query(df, "phase=='CRUISE'")

        # Safe extraction of airport pair and timing

        # fl = self.flight_list.query(f"flight_id=='{flight_id}'")
        fl = self.flight_info
        origin = fl.origin_icao if not fl.empty else "UNK"
        dest = fl.destination_icao if not fl.empty else "UNK"
        apair = origin + dest
        tod = fl.takeoff.hour if not fl.empty else np.nan
        todl = fl.landed.hour if not fl.empty else np.nan
        doy = fl.takeoff.dayofyear if not fl.empty else np.nan

        # Safe aircraft tow lookup
        try:
            tow_row = data.ac_tows.query(f"aircraft_type=='{typecode}'").iloc[0]
            min_tow = tow_row.min_tow_ch_set
            max_tow = tow_row.max_tow_ch_set
            m_tow = tow_row.m_tow
            oew = tow_row.oew
        except Exception:
            min_tow, max_tow, m_tow = np.nan, np.nan, np.nan

        # Helper: safe mean/median/max
        safe_stat = lambda s, func: func(s) if not s.empty else np.nan

        # Build feature dict with robust fallbacks
        X_test = pd.DataFrame(
            [
                {
                    "aircraft_type": typecode,
                    "mean_initroc": safe_stat(df_init.vertical_rate, np.mean),
                    "median_initroc": safe_stat(df_init.vertical_rate, np.median),
                    "max_initroc": safe_stat(df_init.vertical_rate, np.max),
                    "mean_roc": safe_stat(df_climb.vertical_rate, np.mean),
                    "median_roc": safe_stat(df_climb.vertical_rate, np.median),
                    "max_roc": safe_stat(df_climb.vertical_rate, np.max),
                    "mean_pos_climb_vertacc": safe_stat(
                        df_climb.query("vertical_acc>0")["vertical_acc"], np.mean
                    ),
                    "dur_initclimb": (
                        (df_init.timestamp.max() - df_init.timestamp.min()).seconds
                        if not df_init.empty
                        else 0
                    ),
                    "dur_climb": (
                        (df_climb.timestamp.max() - df_climb.timestamp.min()).seconds
                        if not df_climb.empty
                        else 0
                    ),
                    "dur_cruise": (
                        (df_cruise.timestamp.max() - df_cruise.timestamp.min()).seconds
                        if not df_cruise.empty
                        else 0
                    ),
                    "distance": df.distance_km.max() if "distance_km" in df else np.nan,
                    "mean_cruise_altitude": safe_stat(df_cruise.altitude, np.mean),
                    "mean_cruise_tas": safe_stat(df_cruise.TAS, np.mean),
                    "mean_climb_tas": safe_stat(df_climb.TAS, np.mean),
                    "mean_initclimb_tas": safe_stat(df_init.TAS, np.mean),
                    "flight_duration": (
                        (df.timestamp.max() - df.timestamp.min()).seconds / 60
                        if "timestamp" in df
                        else np.nan
                    ),
                    "min_tow_ch": min_tow,
                    "max_tow_ch": max_tow,
                    "m_tow": m_tow,
                    "adepdespair": apair,
                    "time_of_day_arrival": tod,
                    "time_of_day_offblock": todl,
                    "day_of_year": doy,
                }
            ]
        )

        # Handle categorical features
        ct_features = [
            "adepdespair",
            # "countrypair",
            # "time_of_day_offblock",
            # "time_of_day_arrival",
            # "day_of_year",
            "aircraft_type",
        ]
        X_test[ct_features] = X_test[ct_features].astype("category")

        # Predict mass safely
        try:
            mass = model.predict(X_test, categorical_feature=ct_features)[0]
        except Exception as e:
            # print(f"[ERROR] Prediction failed: {e}")
            mass = np.nan
        if m_tow != np.nan:
            if mass > 1.2 * m_tow:
                mass = 0.99 * m_tow
            elif mass < 0.9 * oew:
                mass = 1.15 * oew
            elif mass == np.nan:
                mass = 0.85 * m_tow
        if mass == np.nan:
            mass = 0.85 * m_tow

        return self.assign(tow_est_kg=mass)

    def est_mass_timeflown(self):
        """
        Estimate the mass, based on time flown and maximum take-off and landing weight
        """
        actype = self.data.typecode.iloc[0]
        ac = openap.prop.aircraft(actype.lower(), use_synonym=True)

        # Take-off and landing weight estimation
        tow = 0.8 * ac.get("mtow")
        lw = 0.5 * (ac.get("mlw") + ac.get("oew"))

        # Take-off and landing time
        t_to = self.flight_info.takeoff
        t_ld = self.flight_info.landed

        # Percentage flown of total time
        p_flown = (self.data.timestamp - t_to).dt.total_seconds() / (
            t_ld - t_to
        ).total_seconds()

        # Correct for points before take-off and after landing
        p_flown = np.clip(p_flown, 0, 1)

        # Estimate mass
        mass = tow - p_flown * (tow - lw)
        return self.assign(mass_est_tf=mass)

    def est_dist(self):
        # f = self.cumulative_distance()
        # f = f.assign(distance_km=lambda x: x.cumdist * openap.aero.nm / 1000)
        return self.cumulative_distance(compute_gs=False, compute_track=False).assign(
            distance_km=lambda x: x.cumdist * openap.aero.nm / 1000
        )

    def est_ff(self):
        ac = self.data.typecode.iloc[0]
        mass = self.data.tow_est_kg
        tas = self.data.TAS
        alt = self.data.altitude
        vs = self.data.vertical_rate
        dt = self.data.ts.diff().fillna(0)
        mass_ = mass
        for i in range(5):
            fuel = openap.FuelFlow(ac, use_synonym=True)
            ff = fuel.enroute(mass=mass, tas=tas, alt=alt, vs=vs)

            mass = mass_ - ff * np.cumsum(dt)

        return self.assign(ff_kgs_est=ff, mass_est=mass)

    def est_ff_mass_tf(self):
        ac = self.data.typecode.iloc[0]
        mass = self.data.mass_est_tf
        tas = self.data.TAS
        alt = self.data.altitude
        vs = self.data.vertical_rate
        # dt = self.data.ts.diff().fillna(0)
        # for i in range(5):
        fuel = openap.FuelFlow(ac, use_synonym=True)
        ff = fuel.enroute(mass=mass, tas=tas, alt=alt, vs=vs)

        return self.assign(ff_kgs_est_mass_tf=ff)

    def tas_from_wind(self):
        """
        When ground speed and wind is available, calculate a compute_TAS
        But the compute_TAS should only
        """
        f_df = self.data
        vg = f_df.groundspeed * openap.aero.kts
        dt = f_df.timestamp.diff().dt.total_seconds()
        vgx = vg * np.sin(np.radians(f_df.track))
        vgy = vg * np.cos(np.radians(f_df.track))
        vax = vgx - f_df.u_component_of_wind
        vay = vgy - f_df.v_component_of_wind
        va = np.sqrt((vax**2 + vay**2))
        compute_tas = va / openap.aero.kts
        # airdist = compute_tas * dt * openap.aero.kts
        # f_df.loc[f_df.TAS.isna(), "TAS"] = compute_tas[
        #     f_df.TAS.isna()
        # ]  # this does not work after resampling
        del vg, dt, vgx, vgy, vax, vay, va
        alt = self.data.altitude
        compute_mach = openap.aero.tas2mach(
            compute_tas * openap.aero.kts, alt * openap.aero.ft
        )
        compute_cas = (
            openap.aero.tas2cas(compute_tas * openap.aero.kts, alt * openap.aero.ft)
            / openap.aero.kts
        )
        return self.assign(
            # airdist=airdist.cumsum() / 1000,
            compute_TAS=compute_tas,
            compute_mach=compute_mach,
            compute_CAS=compute_cas,
        )

    def fill_tas_outside_window(self):
        """
        We neet TAS to calculate empirical fuelflow from openap or arcropole
        """
        # speed_col = ['mach', 'CAS', 'TAS']
        mask = self.data.timestamp.notna()
        for _, row in self.fuel_data.iterrows():
            start = row.start
            end = row.end
            seg_mask = (self.data.timestamp >= start) & (self.data.timestamp <= end)
            mask &= ~seg_mask
        self.data.loc[mask, ["TAS"]] = self.data.compute_TAS[mask]
        return self

    def drop_unused_cols(self):
        target_col = ["x", "y", "track_unwrapped"]
        actual_col = [col for col in target_col if col in self.data.columns]
        return self.drop(columns=actual_col)

    def energy(self):
        self.data["wing_surface"], self.data["cd0"], self.data["k"] = (
            Energy.aerodynamic_properties(self.data.typecode.iloc[0])
        )
        temperature, self.data["pressure"], _ = Energy.isa(self.data.altitude)
        if "temperature" not in self.data.columns:
            self.data["temperature"] = temperature
        self.data["density"] = Energy.density_from_temperature(
            self.data.temperature, self.data.pressure
        )
        self.data["drag"] = Energy.drag_estimate(
            self.data.TAS,
            self.data.mass_est_tf,
            self.data.density,
            self.data.wing_surface,
            self.data.cd0,
            self.data.k,
        )
        self.data["acceleration"] = Energy.acceleration(
            self.data.TAS, self.data.timestamp
        )
        self.data["acceleration"] = self.data["acceleration"].fillna(0)
        self.data["thrust"] = Energy.thrust_estimate(
            self.data.vertical_rate,
            self.data.mass_est_tf,
            self.data.TAS,
            self.data.acceleration,
            self.data.drag,
        )
        return self

    def mach_CAS_from_compute_TAS(self):
        tas = self.data.compute_TAS
        alt = self.data.altitude
        mach = openap.aero.tas2mach(tas * openap.aero.kts, alt * openap.aero.ft)
        cas = (
            openap.aero.tas2cas(tas * openap.aero.kts, alt * openap.aero.ft)
            / openap.aero.kts
        )
        return self.assign(compute_mach=mach, compute_CAS=cas)

    def work_done(self):
        dt = self.data.ts.diff().fillna(0)
        thrust = self.data.thrust
        tas_ms = self.data.TAS * openap.extra.aero.kts
        work = np.cumsum(thrust * tas_ms * dt)

        return self.assign(work=work)


class Preprocessor:
    def __init__(
        self, jobs, flight_set, flight_list=None, version: int | str | None = None
    ):
        self.jobs = jobs
        self.flight_set = flight_set
        if self.flight_set is FlightSet.TRAIN:
            preproc_dir = "flights_train_preproc"
            self.flight_list = data.flight_list_train
            self.fuel_data = data.fuel_train_data
        elif self.flight_set is FlightSet.RANK:
            preproc_dir = "flights_rank_preproc"
            self.flight_list = data.flight_list_rank
            self.fuel_data = data.fuel_rank_data
        elif self.flight_set is FlightSet.FINAL:
            preproc_dir = "flights_final_preproc"
            self.flight_list = data.flight_list_final
            self.fuel_data = data.fuel_final_data
        else:
            preproc_dir = data.data_dir / "flights_test_preproc"
            warnings.warn(
                f"Did not provide a flight set, saving processed trajectories in\n{self.preproc_dir}"
            )

        if flight_list is not None:
            self.flight_list = flight_list

        if type(version) is int:
            preproc_dir += f"_v{version}"
        elif type(version) is str:
            preproc_dir += f"_{version}"

        self.preproc_dir = data.data_dir / preproc_dir
        self.preproc_dir.mkdir(exist_ok=True)
        self.meteo_grid = None

    def set_meteo_grid(self, grid):
        self.meteo_grid = grid
        return self

    # @staticmethod
    # def pipe_v2(fid, flight_set, preproc_dir, meteo_grid):
    #     df = data.get_raw_df(fid, flight_set)
    #     flight = FlightProc(df, flight_set)
    #     (
    #         flight.add_airport()
    #         .interp_lla()
    #         .estimate_missing_sparse_cluster()
    #         .filter(
    #             altitude=17,
    #             latitude=17,
    #             longitude=17,
    #             vertical_rate=3,
    #             groundspeed=5,
    #             track=17,
    #         )
    #         .resample("1s", projection="lcc")
    #         .ts()
    #         .phases()
    #         .tas_from_mach_cas()
    #         .get_wind(meteo_grid)
    #         .tas_from_wind()
    #         .fill_tas_outside_window()
    #         .estimate_mass_24_chgpt()
    #         .est_ff()
    #         .to_parquet(preproc_dir / f"{fid}.parquet")
    #     )
    #     return True

    @staticmethod
    def pipe_v1(fid, flight_set, preproc_dir):
        df = data.get_raw_df(fid, flight_set)
        flight = FlightProc(df, flight_set)
        (
            flight.add_airport()
            # .interp_lla()
            # .estimate_missing_sparse_cluster()
            .filter(
                altitude=17,
                latitude=17,
                longitude=17,
                vertical_rate=3,
                groundspeed=5,
                track=17,
            )
            .resample("1s", projection="lcc")
            .ts()
            .phases()
            .est_dist()
            .tas_from_mach_cas()
            .estimate_mass_24_chgpt()
            .est_ff()
            .to_parquet(preproc_dir / f"{fid}.parquet")
        )
        return True

    @staticmethod
    def pipe_v0(
        fid,
        flight_set,
        preproc_dir,
    ):
        df = data.get_rawraw_df(fid, flight_set)
        flight = FlightProc(df, flight_set)
        (
            flight.filter()
            .resample("1s", projection="lcc")
            .phases()
            .ts()
            .est_dist()
            .tas_from_mach_cas()
            .estimate_mass_24_chgpt()
            .est_ff()
            .drop_unused_cols()
            .to_parquet(preproc_dir / f"{fid}.parquet")
        )
        return True

    @staticmethod
    def pipe_best(
        fid,
        flight_set,
        preproc_dir,
    ):
        df = data.get_rawraw_df(fid, flight_set)
        flight = FlightProc(df, flight_set)
        (
            flight.filter()
            .resample("1s", projection="lcc")
            .phases()
            .ts()
            .est_dist()
            .tas_from_gs()
            .estimate_mass_24_chgpt()
            .est_ff()
            .drop_unused_cols()
            .to_parquet(preproc_dir / f"{fid}.parquet")
        )
        return True

    def run_pipe(self, pipe=None, **kwargs):
        """
        run the preprocessing pipeline
        @Input: pipe should be a static function
        """
        fids = self.flight_list.flight_id
        if pipe is None:
            pipe = type(self).pipe_v0
        pipe = functools.partial(
            pipe, flight_set=self.flight_set, preproc_dir=self.preproc_dir, **kwargs
        )
        process_map(pipe, [fid for fid in fids], max_workers=self.jobs)
