import numpy as np
import xarray as xr
import pysumma as ps

# Stephan-Boltzmann constant (J/s/m^2/K^4)
STEFAN = 5.67e-8


def cloud_correction(emis, cloud_frac):
    return (1 + (0.17 * cloud_frac ** 2)) * emis


def cloud_fraction(shortwave, lat, highlimit=0.6, lowlimit=0.35):
    doy = shortwave.time.dt.dayofyear
    s0 = 1360                     # Solar constant (W/m^2)
    phi = lat * 2 * np.pi / 365   # Convert to radian
    # Declination in radians
    delta = (2 * np.pi / 365) * (23.45 * np.sin(2 * np.pi * (284 + doy) / 365))

    # Top of atmosphere radiation
    hs = np.arccos(-np.tan(phi) * np.tan(delta))
    q0 = s0 * (1/np.pi) * (
        hs * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(hs))

    # Fraction of recieved radiation
    k = shortwave / q0

    # Cloud cover fraction
    cloud_frac = 1 - ((k - lowlimit) / (highlimit - lowlimit))
    cloud_frac = cloud_frac.where(cloud_frac > 0, other=0.0)
    cloud_frac = cloud_frac.where(cloud_frac < 1, other=1.0)
    return cloud_frac


def vapor_pressure(air_pressure, spec_humid):
    """See above for derivation"""
    return -1.607 * air_pressure * spec_humid / (spec_humid - 1)


def longwave_prata(air_temp, vapor_pressure, shortwave, lat, do_cloud_correction):
    """
    Reference:
        Prata, A.J., 1996. A new long-wave formula for estimating
        downward clear-sky radiation at the surface. Q. J. R. Meteor.
        Soc. 122 (533), 1127–1151, doi:10.1002/qj.49712253306.
    """
    z = 46.5 * ((vapor_pressure/100) / air_temp)
    emissivity = 1 - (1 + z) * np.exp(-np.sqrt(1.2 + 3 * z))
    if do_cloud_correction:
        cloud_frac = cloud_fraction(shortwave, lat)
        emissivity = cloud_correction(emissivity, cloud_frac)
    return emissivity * np.power(air_temp, 4) * STEFAN


def longwave_tva(air_temp, vapor_pressure, shortwave, lat, do_cloud_correction):
    """
    Reference:
        Tennessee Valley Authority, 1972. Heat and mass transfer between a
        water surface and the atmosphere. Tennessee Valley Authority, Norris,
        TN. Laboratory report no. 14. Water resources research report no. 0-6803.
    """
    emissivity = 0.74 + 0.0049 * vapor_pressure/1000
    if do_cloud_correction:
        cloud_frac = cloud_fraction(shortwave, lat)
        emissivity = cloud_correction(emissivity, cloud_frac)
    return emissivity * np.power(air_temp, 4) * STEFAN


def longwave_satterlund(air_temp, vapor_pressure, shortwave, lat, do_cloud_correction):
    """
    Reference:
        Satterlund, D.R., 1979. An improved equation for estimating long-wave
        radiation from the atmosphere. Water Resour. Res. 15 (6), 1649–1650,
        doi:10.1029/WR015i006p01649.
    """
    vp = vapor_pressure / 1000
    emissivity = 1.08 * (1 - np.exp(-np.power(vp, air_temp/2016)))
    if do_cloud_correction:
        cloud_frac = cloud_fraction(shortwave, lat)
        emissivity = cloud_correction(emissivity, cloud_frac)
    return emissivity * np.power(air_temp, 4) * STEFAN


def longwave_anderson(air_temp, vapor_pressure, shortwave, lat, do_cloud_correction):
    """
    Referencce:
        Anderson, E.R., 1954. Energy budget studies, water loss
        investigations: lake Hefner studies. U.S. Geol. Surv. Prof. Pap. 269,
        71–119 [Available from U.S. Geological Survey, 807 National Center,
        Reston, VA 20192.].
    """
    emissivity = 0.68 + 0.036 * np.power(vapor_pressure/1000, 0.5)
    if do_cloud_correction:
        cloud_frac = cloud_fraction(shortwave, lat)
        emissivity = cloud_correction(emissivity, cloud_frac)
    return emissivity * np.power(air_temp, 4) * STEFAN


def longwave_dokia(air_temp, vapor_pressure, shortwave, lat, do_cloud_correction):
    """
    References:
      -Clear sky:
        Dilley, A. C., and D. M. O<92>Brien (1998), Estimating downward clear sky
        long-wave irradiance at the surface from screen temperature and precipitable
        water, Q. J. R. Meteorol. Soc., 124, 1391<96> 1401.
      -Cloudy sky:
        Kimball, B. A., S. B. Idso, and J. K. Aase (1982), A model of thermal
        radiation from partly cloudy and overcast skies, Water Resour. Res., 18,
        931<96> 936.
    """
    vp = vapor_pressure / 1000 # Convert to kPa
    w = 4560 * (vp / air_temp) # Prata (1996) approximation for precipitable water

    # Clear sky component of longwave
    lw_clear = 59.38 + 113.7 * np.power(air_temp / 273.16, 6) + 96.96 * np.sqrt(w / 25)

    # Cloud cover corrections
    c = cloud_fraction(shortwave, lat)
    cloud_temp = air_temp - 11
    winter = np.logical_or(vp.time.dt.month <= 2, vp.time.dt.month == 12)
    summer = np.logical_and(vp.time.dt.month <= 8, vp.time.dt.month >=6)
    cloud_temp[winter] -= 2
    cloud_temp[summer] += 2

    # Cloudy sky component of longwave
    eps8z = 0.24 + 2.98e-6 * np.power(vp, 2.0) * np.exp(3000/air_temp)
    tau8 = 1 - eps8z * (1.4 - (0.4 * eps8z))
    f8 = -0.6732 + 0.6240e-2 * cloud_temp - 0.914e-5 * np.power(cloud_temp, 2.0)
    lw_cloud = tau8 * c * f8 * STEFAN * np.power(cloud_temp, 4)
    if do_cloud_correction:
        return lw_clear + lw_cloud
    else:
        return lw_clear


def modify_longwave(parameterization, manager_path, do_cloud_correction=True):
    """
    Creates a new file manager that points to a forcing
    file with new longwave based on an empirical parameterization

    Parameters
    ----------
    parameterization:
        The name of the longwave parameterization to use
    manager_path:
        A the path to a file manager to use as a template
    do_cloud_correction:
        Whether to apply a cloudiness correction factor to the longwave

    Returns
    -------
    The path to the new file manager
    """
    param_funcs = {
        'dokia': longwave_dokia,
        'anderson': longwave_anderson,
        'satterlund': longwave_satterlund,
        'tva': longwave_tva,
        'prata': longwave_prata,
    }
    if do_cloud_correction:
        parameterization_str = parameterization + "_cloud"
    else:
        parameterization_str = parameterization + "_nocloud"
    man_path = '/'.join(manager_path.split('/')[0:-1])
    man_file = manager_path.split('/')[-1]
    filemanager = ps.FileManager(man_path, man_file)
    orig_filepath = filemanager.force_file_list.forcing_paths[0]
    out_filepath = orig_filepath.replace('.nc', f'_{parameterization_str}.nc')
    out_filename = out_filepath.split('/')[-1]
    lat = filemanager.local_attributes['latitude'].values[0]
    forcing = xr.open_dataset(orig_filepath)
    vp = vapor_pressure(forcing['airpres'], forcing['spechum'])
    forcing['LWRadAtm'].values = param_funcs[parameterization](
            forcing['airtemp'], vp, forcing['SWRadAtm'], lat, do_cloud_correction)
    forcing.to_netcdf(out_filepath)

    # Write the new forcing file list and file manager
    new_force_file_list = f'forcing_file_list_{parameterization_str}.txt'
    new_file_manager = f'file_manager_{parameterization_str}.txt'
    with open(f'{filemanager["settingsPath"].value}/{new_force_file_list}', 'w') as f:
        f.write(f"'{out_filename}'")
    filemanager['forcingListFile'].value = new_force_file_list
    filemanager['outFilePrefix'] = (filemanager['outFilePrefix'].value
                                    + f'_{parameterization_str}')
    filemanager.file_name = new_file_manager
    filemanager.write()
    return f'{filemanager.original_path}/{filemanager.file_name}'
