# # #!/usr/bin/env python
# # # -*- coding: utf-8 -*-

# import numpy as np
# from matplotlib import pyplot as plt


# def _wind1d(m_points, parameter_points, distance_units, scale="logx", fig=None, ax=None, i=0, j=0):
#     """Plot a 1D wind.

#     Parameters
#     ----------
#     m_points: np.ndarray
#         The 1st axis points, which are the r bins.
#     parameter_points: np.ndarray
#         The wind parameter to be plotted, in the same shape as the n_points and
#         m_points arrays.
#     distance_units: pypython.wind.WindDistanceUnits
#         The units of the distance axis.
#     scale: str [optional]
#         The scaling of the axes: [logx, logy, loglog, linlin]
#     fig: plt.Figure [optional]
#         A Figure object to update, otherwise a new one will be created.
#     ax: plt.Axes [optional]
#         An axes array to update, otherwise a new one will be created.
#     i: int [optional]
#         The i index for the sub panel to plot onto.
#     j: int [optional]
#         The j index for the sub panel to plot onto.

#     Returns
#     -------
#     fig: plt.Figure
#         The (updated) Figure object for the plot.
#     ax: plt.Axes
#         The (updated) axes array for the plot.
#     """
#     if fig is None or ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6), squeeze=False)

#     ax[i, j].plot(m_points, parameter_points)

#     if distance_units == WindDistanceUnits.cm:
#         ax[i, j].set_xlabel(r"$R$ [cm]")
#     else:
#         ax[i, j].set_xlabel(r"$r / R_{g}$")
#     # ax[i, j].set_xlim(np.min(m_points[m_points > 0]), np.max(m_points))
#     ax[i, j] = set_axes_scales(ax[i, j], scale)
#     fig = finish_figure(fig)

#     return fig, ax


# def _wind2d(
#     m_points,
#     n_points,
#     parameter_points,
#     distance_units,
#     coordinate_system,
#     inclinations_to_plot=None,
#     scale="loglog",
#     vmin=None,
#     vmax=None,
#     fig=None,
#     ax=None,
#     i=0,
#     j=0,
# ):
#     """Plot a 2D wind using a contour plot.

#     Parameters
#     ----------
#     m_points: np.ndarray
#         The 1st axis points, either x or angular (in degrees) bins.
#     n_points: np.ndarray
#         The 2nd axis points, either z or radial bins.
#     parameter_points: np.ndarray
#         The wind parameter to be plotted, in the same shape as the n_points and
#         m_points arrays.
#     distance_units: pypython.wind.WindDistanceUnits
#         The units of the distance axes.
#     coordinate_system: pypython.wind.WindCoordSystem
#         The coordinate system in use, either rectilinear or polar.
#     inclinations_to_plot: List[str] [optional]
#         A list of inclination angles to plot onto the ax[0, 0] sub panel. Must
#         be strings and 0 < inclination < 90.
#     scale: str [optional]
#         The scaling of the axes: [logx, logy, loglog, linlin]
#     vmin: float [optional]
#         The minimum value to plot.
#     vmax: float [optional]
#         The maximum value to plot.
#     fig: plt.Figure [optional]
#         A Figure object to update, otherwise a new one will be created.
#     ax: plt.Axes [optional]
#         An axes array to update, otherwise a new one will be created.
#     i: int [optional]
#         The i index for the sub panel to plot onto.
#     j: int [optional]
#         The j index for the sub panel to plot onto.

#     Returns
#     -------
#     fig: plt.Figure
#         The (updated) Figure object for the plot.
#     ax: plt.Axes
#         The (updated) axes array for the plot.
#     """
#     if fig is None or ax is None:
#         if coordinate_system == WindCoordSystem.cylindrical:
#             fig, ax = plt.subplots(figsize=(8, 6), squeeze=False)
#         elif coordinate_system == WindCoordSystem.polar:
#             fig, ax = plt.subplots(figsize=(8, 6), squeeze=False, subplot_kw={"projection": "polar"})
#         else:
#             raise ValueError(f"Unknown projection, expected {WindCoordSystem.cylindrical} or {WindCoordSystem.polar}")

#     im = ax[i, j].pcolormesh(
#         m_points, n_points, parameter_points, shading="auto", vmin=vmin, vmax=vmax, linewidth=0, rasterized=True
#     )

#     fig.colorbar(im, ax=ax[i, j])

#     # this plots lines representing sight lines for different observers of
#     # different inclinations

#     if inclinations_to_plot:
#         n_coords = np.unique(m_points)
#         for inclination in inclinations_to_plot:
#             if coordinate_system == WindCoordSystem.cylindrical:
#                 m_coords = n_coords * np.tan(0.5 * PI - np.deg2rad(float(inclination)))
#             else:
#                 x_coords = np.logspace(np.log10(0), np.max(n_points))
#                 m_coords = x_coords * np.tan(0.5 * PI - np.deg2rad(90 - float(inclination)))
#                 m_coords = np.sqrt(x_coords**2 + m_coords**2)

#             ax[0, 0].plot(n_coords, m_coords, label=inclination + r"$^{\circ}$")

#         ax[0, 0].legend(loc="lower left")

#     # Clean up the axes with labs and set up scales, limits etc

#     if coordinate_system == WindCoordSystem.cylindrical:
#         if distance_units == WindDistanceUnits.cm:
#             ax[i, j].set_xlabel(r"$x$ [cm]")
#             ax[i, j].set_ylabel(r"$z$ [cm]")
#         else:
#             ax[i, j].set_xlabel(r"$x / R_{g}$")
#             ax[i, j].set_ylabel(r"$z / R_{g}$")
#         ax[i, j].set_xlim(np.min(m_points[m_points > 0]), np.max(m_points))
#         ax[i, j] = set_axes_scales(ax[i, j], scale)
#     else:
#         ax[i, j].set_theta_zero_location("N")
#         ax[i, j].set_theta_direction(-1)
#         ax[i, j].set_thetamin(0)
#         ax[i, j].set_thetamax(90)
#         ax[i, j].set_rlabel_position(90)
#         if distance_units == WindDistanceUnits.cm:
#             ax[i, j].set_ylabel(r"$\log_{10}(r)$ [cm]")
#         else:
#             ax[i, j].set_ylabel(r"$\log_{10}(r / R_{g})$")

#     ax[i, j].set_ylim(np.min(n_points[n_points > 0]), np.max(n_points))
#     fig = finish_figure(fig)

#     return fig, ax


# def wind(
#     wind,
#     parameter,
#     inclinations_to_plot=None,
#     log_parameter=True,
#     scale="loglog",
#     vmin=None,
#     vmax=None,
#     use_coordinates=True,
#     fig=None,
#     ax=None,
#     i=0,
#     j=0,
# ):
#     """Plot a variable from a 1D or 2D wind model.

#     The function is a simple wrapper around _plot_1d_wind and _plot_2d_wind and
#     will use the Wind object to decide if a wind is 1D or 2D, calling the
#     appropriate function to plot the wind.

#     Parameters
#     ----------
#     wind: Wind
#         The Wind object.
#     parameter: np.ndarray or str
#         The wind parameter to be plotted, in the same shape as the coordinate
#         arrays. Can also be the name of the variable.
#     inclinations_to_plot: List[str] [optional]
#         A list of inclination angles to plot onto the ax[0, 0] sub panel. Must
#         be strings and 0 < inclination < 90.
#     log_parameter: bool [optional]
#         If the name of the parameter is provided, rather than an array. This
#         can be used to plot the variable on a log10 or linear color scale.
#     scale: str [optional]
#         The scaling of the axes: [logx, logy, loglog, linlin]
#     vmin: float or None [optional]
#         The minimum value to plot.
#     vmax: float or None [optional]
#         The maximum value to plot.
#     use_coordinates: bool [optional]
#         Plot the wind either using the cell coordinates or the cell indices.
#     fig: plt.Figure [optional]
#         A Figure object to update, otherwise a new one will be created.
#     ax: plt.Axes [optional]
#         An axes array to update, otherwise a new one will be created.
#     i: int [optional]
#         The i index for the sub panel to plot onto.
#     j: int [optional]
#         The j index for the sub panel to plot onto.

#     Returns
#     -------
#     fig: plt.Figure
#         The (updated) Figure object for the plot.
#     ax: plt.Axes
#         The (updated) axes array for the plot.
#     """
#     # If parameter is a string, get the parameter ourself. If it's a numpy
#     # array, then map parameter to parameter_points

#     if type(parameter) is str:
#         parameter_points = wind.get(parameter)
#         if log_parameter:
#             parameter_points = np.log10(parameter_points)
#     elif type(parameter) in [np.ndarray, np.ma.core.MaskedArray]:
#         parameter_points = parameter
#     else:
#         raise ValueError(f"Incompatible type {type(parameter)} for parameter, has to be an numpy array or string")

#     # The stuff can now be passed to the the appropriate plotting function
#     # depending on the coordinate system of the wind

#     if wind.coord_system == WindCoordSystem.spherical:
#         if use_coordinates:
#             r = wind["r"]
#         else:
#             r = wind["i"]
#         fig, ax = _wind1d(r, parameter_points, wind.distance_units, scale, fig, ax, i, j)
#     elif wind.coord_system == WindCoordSystem.cylindrical:
#         if use_coordinates:
#             x, z = wind["x"], wind["z"]
#         else:
#             x, z = wind["i"], wind["j"]
#         fig, ax = _wind2d(
#             x,
#             z,
#             parameter_points,
#             wind.distance_units,
#             wind.coord_system,
#             inclinations_to_plot,
#             scale,
#             vmin,
#             vmax,
#             fig,
#             ax,
#             i,
#             j,
#         )
#     elif wind.coord_system == WindCoordSystem.polar:
#         if not use_coordinates:
#             raise ValueError("cannot use cell indices to plot a polar wind")
#         fig, ax = _wind2d(
#             np.deg2rad(wind["theta"]),
#             np.log10(wind["r"]),
#             parameter_points,
#             wind.distance_units,
#             wind.coord_system,
#             inclinations_to_plot,
#             scale,
#             vmin,
#             vmax,
#             fig,
#             ax,
#             i,
#             j,
#         )
#     else:
#         raise ValueError(f"Unknown coordinate system of type {wind.coord_system}")

#     return fig, ax


# from pypython.constants import PI
# from pypython.plot import finish_figure, set_axes_scales
# from pypython.wind_old import WindCoordSystem, WindDistanceUnits
