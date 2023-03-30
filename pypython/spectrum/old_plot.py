#!/usr/bin/env python
# -*- coding: utf-8 -*-


# def _convert_labels_to_frequency_space(lines, spectral_axis=None, spectrum=None):
#     """Convert the given list of lines/edges from Angstrom to Hz.

#     Parameters
#     ----------
#     lines: List[str, float]
#         The list of labels to convert from wavelength to frequency space.
#     spectrum: pypython.Spectrum
#         A spectrum object, used to find the units of the spectrum.
#     """
#     if spectrum is None and spectral_axis is None:
#         return lines

#     if spectrum:
#         spectral_axis = spectrum["spectral_axis"]

#     if spectral_axis == pypython.spectrum.SpectrumSpectralAxis.frequency:
#         for i in range(len(lines)):
#             lines[i][1] = c.VLIGHT / (lines[i][1] * c.ANGSTROM)

#     return lines


# def _get_inclinations(spectrum, inclinations):
#     """Get the inclination angles which will be used.

#     If all is passed, then the inclinations from the spectrum object will
#     be used. Otherwise, this will simply create a list of strings of the
#     inclinations.

#     Parameters
#     ----------
#     spectrum: pypython.Spectrum
#         The spectrum object.
#     inclinations: list
#         A list of inclination angles wanted to be plotted.
#     """
#     if type(inclinations) != list:
#         inclinations = str(inclinations)

#     if type(inclinations) == str:
#         inclinations = [inclinations]

#     if len(inclinations) > 1:
#         if inclinations[0] == "all":  # ignore "all" if other inclinations are passed
#             inclinations = inclinations[1:]
#     else:
#         if inclinations[0] == "all":
#             inclinations = spectrum.inclinations

#     return [str(inclination) for inclination in inclinations]
# def add_line_ids(
#     ax, lines, linestyle="none", ynorm=0.90, offset=0, rotation="vertical", fontsize=15, whitespace_scale=2
# ):
#     """Add labels for line transitions or other regions of interest onto a
#     matplotlib figure. Labels are placed at the top of the panel and dashed
#     lines, with zorder = 0, are drawn from top to bottom.

#     Parameters
#     ----------
#     ax: plt.Axes
#         The axes (plt.Axes) to add line labels too
#     lines: list
#         A list containing the line name and wavelength in Angstroms
#         (ordered by wavelength)
#     linestyle: str [optional]
#         The type of line to draw to show where the transitions are. Allowed
#         values [none, dashed, top]
#     ynorm: float [optional]
#         The normalized y coordinate to place the label.
#     logx: bool [optional]
#         Use when the x-axis is logarithmic
#     offset: float or int [optional]
#         The amount to offset line labels along the x-axis
#     rotation: str [optional]
#         Vertical or horizontal rotation for text ids
#     fontsize: int [optional]
#         The fontsize of the labels
#     whitespace_scale: float [optional]
#         The amount to scale the upper y limit of the plot by, to add whitespace
#         for the line labels.

#     Returns
#     -------
#     ax: plt.Axes
#         The updated axes object.
#     """
#     nlines = len(lines)
#     xlims = ax.get_xlim()
#     ylims = ax.get_ylim()

#     for i in range(nlines):
#         label = lines[i][0]
#         x = lines[i][1]

#         if x < xlims[0]:
#             continue
#         if x > xlims[1]:
#             continue

#         if linestyle == "dashed":
#             ax.axvline(x, linestyle="--", linewidth=0.5, color="k", zorder=1)
#         elif linestyle == "thick":
#             ax.axvline(x, linestyle="-", linewidth=2, color="k", zorder=1)
#         elif linestyle == "top":
#             raise NotImplementedError()

#         x = x - offset

#         # Calculate the x location of the label in axes coordinates

#         if ax.get_xscale() == "log":
#             xnorm = (np.log10(x) - np.log10(xlims[0])) / (np.log10(xlims[1]) - np.log10(xlims[0]))
#         else:
#             xnorm = (x - xlims[0]) / (xlims[1] - xlims[0])

#         ax.text(
#             xnorm, ynorm, label, ha="center", va="center", rotation=rotation, fontsize=fontsize, transform=ax.transAxes
#         )

#     ax.set_ylim(ylims[0], ylims[1] * whitespace_scale)

#     return ax


# def common_lines(spectrum=None, spectral_axis=None):
#     """Return a list containing the names of line transitions and the
#     wavelength of the transition in Angstroms. Instead of returning the
#     wavelength, the frequency can be returned instead. It is also possible to
#     return in log space.

#     Parameters
#     ----------
#     spectral_axis: pypython.Spectrum.SpectrumSpectralAxis
#         The units of the spectral axis
#     spectrum: pypython.Spectrum
#         The spectrum object. Used to get the spectral axis units.

#     Returns
#     -------
#     line: List[List[str, float]]
#         A list of lists where each element of the list is the name of the
#         transition/edge and the rest wavelength of that transition in
#         Angstroms.
#     """

#     lines = [
#         [r"N \textsc{iii} / O \textsc{iii}", 305],
#         [r"P \textsc{v}", 1118],
#         [r"Ly$\alpha$ / N \textsc{v}", 1216],
#         ["", 1242],
#         [r"O \textsc{v} / Si \textsc{iv}", 1371],
#         ["", 1400],
#         [r"N \textsc{iv}", 1489],
#         [r"C \textsc{iv}", 1548],
#         ["", 1550],
#         [r"He \textsc{ii}", 1640],
#         [r"N \textsc{iii]}", 1750],
#         [r"Al \textsc{iii}", 1854],
#         [r"C \textsc{iii]}", 1908],
#         [r"Mg \textsc{ii}", 2798],
#         [r"Ca \textsc{ii}", 3934],
#         ["", 3969],
#         [r"H$_{\delta}$", 4101],
#         [r"H$_{\gamma}$", 4340],
#         [r"He \textsc{ii}", 4389],
#         [r"He \textsc{ii}", 4686],
#         [r"H$_{\beta}$", 4861],
#         [r"He \textsc{i}", 5877],
#         ["", 5897],
#         [r"H$_{\alpha}$", 6564],
#         [r"He \textsc{i}", 7067],
#     ]

#     return _convert_labels_to_frequency_space(lines, spectral_axis, spectrum)


# def photoionization_edges(spectrum=None, spectral_axis=None):
#     """Return a list containing the names of line transitions and the
#     wavelength of the transition in Angstroms. Instead of returning the
#     wavelength, the frequency can be returned instead. It is also possible to
#     return in log space.

#     Parameters
#     ----------
#     spectral_axis: pypython.Spectrum.SpectrumSpectralAxis
#         The units of the spectral axis
#     spectrum: Spectrum [optional]
#         The spectrum object. Used to get the spectral axis units.

#     Returns
#     -------
#     edges: List[List[str, float]]
#         A list of lists where each element of the list is the name of the
#         transition/edge and the rest wavelength of that transition in
#         Angstroms.
#     """

#     edges = [
#         [r"O \textsc{viii}", 14],
#         [r"O \textsc{vii}", 16],
#         [r"O \textsc{vi} / O \textsc{v}", 98],
#         [r"", 105],
#         [r"O \textsc{iv}", 160],
#         [r"He \textsc{ii}", 227],
#         [r"He \textsc{i}", 504],
#         ["Lyman", 912],
#         ["Balmer", 3646],
#         ["Paschen", 8204],
#     ]

#     return _convert_labels_to_frequency_space(edges, spectral_axis, spectrum)


# # Plotting functions -----------------------------------------------------------


# def components(
#     spectrum, xmin=None, xmax=None, scale="loglog", alpha=0.65, multiply_by_spatial_units=False, display=False
# ):
#     """Plot the different components of the spectrum.

#     The components are the columns labelled with words, rather than inclination
#     angles. These columns are not supposed to add together, i.e. all of them
#     together shouldn't equal the Emitted spectrum, which is the angle
#     averaged escaping flux/luminosity.

#     Parameters
#     ----------
#     spectrum: pypython.Spectrum
#         The spectrum object.
#     xmin: float [optional]
#         The minimum x boundary to plot.
#     xmax: float [optional]
#         The maximum x boundary to plot.
#     scale: str [optional]
#         The scaling of the axes.
#     alpha: float [optional]
#         The line transparency on the plot.
#     multiply_by_spatial_units: bool [optional]
#         Plot in flux or nu Lnu instead of flux density or luminosity.
#     display: bool [optional]
#         Display the object once plotted.

#     Returns
#     -------
#     fig: plt.Figure
#         matplotlib Figure object.
#     ax: plt.Axes
#         matplotlib Axes object.
#     """
#     fig, ax = plt.subplots(2, 1, figsize=(12, 10))

#     ax[0] = plot_spectrum(
#         ax[0], spectrum, ["CenSrc", "Disk", "WCreated"], xmin, xmax, alpha, scale, multiply_by_spatial_units
#     )
#     ax[1] = plot_spectrum(
#         ax[1], spectrum, ["Created", "Emitted", "Wind", "HitSurf"], xmin, xmax, alpha, scale, multiply_by_spatial_units
#     )

#     fig = pyplt.finish_figure(fig)
#     fig.savefig(f"{spectrum.fp}/{spectrum.root}_{spectrum.current}_components.png")

#     if display:
#         plt.show()

#     return fig, ax


# def optical_depth(
#     spectrum,
#     inclinations="all",
#     xmin=None,
#     xmax=None,
#     scale="loglog",
#     label_edges=True,
#     frequency_space=True,
#     display=False,
# ):
#     """Plot the continuum optical depth spectrum.

#     Create a plot of the continuum optical depth against either frequency or
#     wavelength. Frequency space is the default and preferred. This function
#     returns the Figure and Axes object.

#     Parameters
#     ----------
#     spectrum: pypython.Spectrum
#         The spectrum object.
#     inclinations: str or list or tuple [optional]
#         A list of inclination angles to plot, but all is also an acceptable
#         choice if all inclinations are to be plotted.
#     xmin: float [optional]
#         The lower x boundary for the figure
#     xmax: float [optional]
#         The upper x boundary for the figure
#     scale: str [optional]
#         The scale of the axes for the plot.
#     label_edges: bool [optional]
#         Label common absorption edges of interest onto the figure
#     frequency_space: bool [optional]
#         Create the figure in frequency space instead of wavelength space
#     display: bool [optional]
#         Display the final plot if True.

#     Returns
#     -------
#     fig: plt.Figure
#         matplotlib Figure object.
#     ax: plt.Axes
#         matplotlib Axes object.
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(12, 7))
#     current = spectrum.current
#     spectrum.set("spec_tau")

#     # Determine if we're plotting in frequency or wavelength space and then
#     # determine the inclinations we want to plot

#     if frequency_space:
#         xlabel = "Freq."
#         units = pypython.spectrum.SpectrumUnits.f_nu
#     else:
#         xlabel = "Lambda"
#         units = pypython.spectrum.enum.SpectrumUnits.F_LAM

#     if not xmin:
#         xmin = np.min(spectrum[xlabel])
#     if not xmax:
#         xmax = np.max(spectrum[xlabel])

#     inclinations = _get_inclinations(spectrum, inclinations)

#     # Now loop over the inclinations and plot each one, skipping ones which are
#     # completely empty

#     for inclination in inclinations:
#         if inclination != "all":
#             if inclination not in spectrum.inclinations:  # Skip inclinations which don't exist
#                 continue

#         x, y = pypython.get_xy_subset(spectrum[xlabel], spectrum[inclination], xmin, xmax)
#         if np.count_nonzero(y) == 0:  # skip arrays which are all zeros
#             continue

#         ax.plot(x, y, label=f"{inclination}" + r"$^{\circ}$")

#     ax = pyplt.set_axes_scales(ax, scale)
#     ax.set_ylabel(r"Continuum Optical Depth")

#     if frequency_space:
#         ax.set_xlabel(r"Rest-frame Frequency [Hz]")
#     else:
#         ax.set_xlabel(r"Rest-frame Wavelength [$\AA$]")

#     ax.legend(loc="upper left")

#     if label_edges:
#         ax = add_line_ids(ax, photoionization_edges(spectrum), linestyle="none", offset=0)

#     spectrum.set(current)
#     fig = pyplt.finish_figure(fig)
#     fig.savefig(f"{spectrum.fp}/{spectrum.root}_spec_optical_depth.png")

#     if display:
#         plt.show()

#     return fig, ax


# def reprocessing(spectrum, xmin=None, xmax=None, scale="loglog", label_edges=True, alpha=0.75, display=False):
#     """Create a plot to show the amount of reprocessing in the model.

#     Parameters
#     ----------

#     Returns
#     -------
#     fig: plt.Figure
#         matplotlib Figure object.
#     ax: plt.Axes
#         matplotlib Axes object.
#     """
#     if "spec_tau" not in spectrum.available:
#         raise ValueError("There is no spec_tau spectrum so cannot create this plot")
#     if "spec" not in spectrum.available and "log_spec" not in spectrum.available:
#         raise ValueError("There is no observer spectrum so cannot create this plot")

#     fig, ax = plt.subplots(figsize=(12, 7))
#     ax2 = ax.twinx()

#     ax = pyplt.set_axes_scales(ax, scale)
#     ax2 = pyplt.set_axes_scales(ax2, scale)

#     # Plot the optical depth

#     spectrum.set("spec_tau")

#     for n, inclination in enumerate(spectrum.inclinations):
#         y = spectrum[inclination]
#         if np.count_nonzero == 0:
#             continue
#         ax.plot(spectrum["Freq."], y, label=str(inclination) + r"$^{\circ}$", alpha=alpha)

#     ax.legend(loc="upper left")
#     ax.set_xlim(xmin, xmax)
#     ax.set_xlabel("Rest-frame Frequency [Hz]")
#     ax.set_ylabel("Continuum Optical Depth")

#     if label_edges:
#         ax = add_line_ids(ax, photoionization_edges(spectrum=spectrum), "none")

#     ax.set_zorder(ax2.get_zorder() + 1)
#     ax.patch.set_visible(False)

#     # Plot the emitted and created spectrum

#     spectrum.set("spec")

#     for thing in ["Created", "Emitted"]:
#         x, y = pypython.get_xy_subset(spectrum["Freq."], spectrum[thing], xmin, xmax)

#         if spectrum.units == pypython.spectrum.enum.SpectrumUnits.F_LAM:
#             y *= spectrum["Lambda"]
#         else:
#             y *= spectrum["Freq."]

#         ax2.plot(x, y, label=thing, alpha=alpha)

#     ax2.legend(loc="upper right")
#     ax2 = set_axes_labels(ax2, spectrum)

#     fig = pyplt.finish_figure(fig)
#     fig.savefig(f"{spectrum.fp}/{spectrum.root}_reprocessing.png")

#     if display:
#         plt.show()

#     return fig, ax
