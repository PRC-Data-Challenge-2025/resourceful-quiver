"""Energy related features"""

import numpy as np
import pandas as pd
import openap
from openap.drag import Drag
from traffic.core import Flight

# ----- Units -----
kt = 0.51445    # [m/s]
ft = 0.3048     # [m]

# ----- International Standard Atmosphere -----

layers = ['troposphere',
          'tropopause',
          'stratosphere1',
          'stratosphere2',
          'stratopause',
          'mesosphere1',
          'mesosphere2',
          'mesopause']

# Base heights [m]
hb = [0.,
      11000.,
      20000.,
      32000,
      47000,
      51000,
      71000.,
      84852.]

# Temperature lapse rate [K/m]
a = [-0.0065,
     0.,
     0.001,
     0.0028,
     0.,
     -0.0028,
     -0.002]

# Base values
Tb = 288.15     # [K]
pb = 101325.    # [Pa]
rhob = 1.225    # [kg/m3]

R = 287.05287   # [J/kg K]
g = 9.80665     # [m/s2]
gamma = 1.4     # [-]


class Energy:
    @staticmethod
    def aerodynamic_properties(typecode: pd.Series|str) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Get relevant areodynamic properties
        Args:
            typecode (Series|str): aircraft type code
        Returns:
            surface (Series): wing surface area [m2]
            cd0 (Series): zero-lift drag coefficient [-]
            k (Series): lift-induced drag factor [-]
        """
        typecode = typecode.upper()

        # DataFrame with multiple type codes
        if not isinstance(typecode, str) and len(typecode.unique()) > 1:
            # Collect relevant parameters
            surfaces = dict()
            cd0s = dict()
            ks = dict()
            for actype in openap.prop.available_aircraft(use_synonym=True):
                ac = openap.prop.aircraft(actype, use_synonym=True)
                drag = Drag(ac=actype, use_synonym=True)
                surfaces[actype.upper()] = float(ac.get('wing').get('area'))
                cd0s[actype.upper()] = float(drag.polar.get('clean').get('cd0'))
                ks[actype.upper()] = float(drag.polar.get('clean').get('k'))
            
            # Map to DataFrame
            surface = pd.Series(typecode.map(surfaces))
            cd0 = pd.Series(typecode.map(cd0s))
            k = pd.Series(typecode.map(ks))
        # Typecode as string
        elif isinstance(typecode, str):
            ac = openap.prop.aircraft(typecode, use_synonym=True)
            drag = Drag(ac=typecode, use_synonym=True)

            surface = float(ac.get('wing').get('area'))
            cd0 = float(drag.polar.get('clean').get('cd0'))
            k = float(drag.polar.get('clean').get('k'))
        # Typecode as Series
        else:
            ac = openap.prop.aircraft(typecode.iloc[0], use_synonym=True)
            drag = Drag(ac=typecode.iloc[0], use_synonym=True)

            surface = pd.Series([float(ac.get('wing').get('area'))]*len(typecode))
            cd0 = pd.Series([float(drag.polar.get('clean').get('cd0'))]*len(typecode))
            k = pd.Series([float(drag.polar.get('clean').get('k'))]*len(typecode))

        return surface, cd0, k

    @staticmethod
    def isa(alt: pd.Series, deltaT: float = 0.) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        International Standard Atmosphere (ISA)

        Parameters:
            alt (Series): altitude [ft]
            deltaT (float): base temperature shift [deg K or deg C]
        
        Returns:
            tuple: Temperature [K], Pressure [Pa], Density [kg/m3]
        """
        # Convert to [m]
        alt = np.array(alt)*ft  # [m]

        # Start values
        T0 = np.ones(len(alt))*(Tb + deltaT)
        p0 = np.ones(len(alt))*pb
        rho0 = np.ones(len(alt))*rhob

        for i, h0 in enumerate(hb):
            # Reached the mesopause
            if h0 == hb[-1]:
                return pd.Series(T0), pd.Series(p0), pd.Series(rho0)
            
            # Heigth delta
            dh = np.minimum(alt - h0, hb[i + 1] - h0)
            
            # Parameters
            T1 = np.where(dh > 0, T0 + a[i]*dh, T0)
            if np.abs(a[i]) > 0.:
                gaR = g/(a[i]*R)
                p1 = np.where(dh > 0, p0*(np.power(T1/T0, -gaR)), p0)
                rho1 = np.where(dh > 0, rho0*(np.power(T1/T0, -gaR - 1)), rho0)
            else:
                egRT = np.exp((-g/(R*T0))*dh)
                p1 = np.where(dh > 0, p0*egRT, p0)
                rho1 = np.where(dh > 0, rho0*egRT, rho0)
            
            # Update 0 values
            T0 = T1
            p0 = p1
            rho0 = rho1

    @staticmethod
    def density_from_temperature(temperature, pressure) -> float:
        """
        Ideal gas law
        Args:
            temperature: Temperature [K]
            pressure: Pressure [Pa]
        Returns:
            density [kg/m3]
        """
        return pressure/(R*temperature)

    @staticmethod
    def acceleration(tas: pd.Series, timestamp: pd.Series, order: int = 1) -> pd.Series:
        """
        Estimate acceleration
        Args:
            tas (Series): TAS [kts]
            timestamp (Series): Time stamps
            order (int): Order of finite difference
        Returns:
            acceleration (Series): acceleration estimate [m/s2]
        """
        a = 0.
        tas = np.array(tas)*kt   # [m/s]

        # First order backward
        if order == 1:
            t_current = np.array(timestamp)
            tas_prev = np.roll(tas, 1)
            t_prev = np.roll(timestamp, 1)

            a = (tas - tas_prev) / (t_current - t_prev).astype('timedelta64[s]').astype('float64')
            a[0] = np.nan
        # Second order backward
        elif order == 2:
            tas_prev = np.roll(tas, 1)
            tas_2prev = np.roll(tas, 2)
            t_current = np.array(timestamp)
            t_prev = np.roll(timestamp, 1)
            t_2prev = np.roll(timestamp, 2)

            # Probably on works well if dt is constant
            a = (tas - 2*tas_prev + tas_2prev)/np.power(t_current + t_prev + t_2prev, 2)
            a[0] = np.nan
            a[1] = np.nan

        return a

    @staticmethod
    def drag_estimate(tas: pd.Series, mass: pd.Series, density: pd.Series,
                      wing_surface: pd.Series, cd0: pd.Series, k: pd.Series) -> pd.Series:
        """
        Estimate drag
        Args:
            tas (Series): True airspeed [kts]
            mass (Series): Mass [kg]
            density (Series): Density [kg/m3]
            wing_surface (Series): Wing surface area [m2]
            cd0 (Series): Zero-lift drag coefficient [-]
            k (Series): Lift-induced drag factor [-]
        Returns:
            D (Series): drag estimation [N]
        """

        tas = np.array(tas)*kt   # [m/s]

        CL = 2*mass*g/(density * tas*tas * wing_surface)

        CD = cd0 + k * CL*CL

        D = CD * 0.5 * density * tas*tas * wing_surface

        return pd.Series(D)

    @staticmethod
    def thrust_estimate(vertical_rate: pd.Series, mass: pd.Series, tas: pd.Series, acceleration: pd.Series, drag: pd.Series) -> pd.Series:
        """
        Estimate the thrust
        Args:
            vertical_rate (Series): Vertical rate [fpm]
            mass (Series): Mass [kg]
            tas (Series): True airspeed [kts]
            acceleration (Series): Acceleration [m/s2]
            drag (Series): Drag force [N]
        Returns:
            T (Series): thrust force [N]
        """
        vs = np.array(vertical_rate)*ft/60.  # [m/s]

        T = mass*g * vs / tas + mass * acceleration + drag

        return pd.Series(T)
    
    @staticmethod
    def work_done(thrust: pd.Series, tas: pd.Series, timestamp: pd.Series) -> float:
        """
        Compute the work done for a flight segment
        Args:
            thrust (Series): Thrust force [N]
            tas (Series): True airspeed [kts]
            timestamp (Series): Time stamps
        Returns:
            W (float): work done [Nm]
        """
        # Unit conversion
        tas = np.array(tas)*kt   # [m/s]

        # Time delta
        dt = np.roll(timestamp, -1) - np.array(timestamp)
        dt = np.array(dt, dtype=np.float64) / 1e9
        dt[-1] = np.nan
        dt = np.roll(dt, 1)

        # Air distance
        s = tas*dt

        # Work done
        W = s*np.array(thrust)
        W = np.sum(W[:-1])

        return float(W)
