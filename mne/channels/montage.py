# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: Simplified BSD

import os
import os.path as op
import warnings

import numpy as np

from ..viz import plot_montage
from .channels import _contains_ch_type
from ..transforms import (_sphere_to_cartesian, apply_trans,
                          get_ras_to_neuromag_trans, _topo_to_sphere,
                          _str_to_frame, _frame_to_str)
from ..io.meas_info import _make_dig_points, _read_dig_points, _read_dig_fif
from ..io.pick import pick_types
from ..io.open import fiff_open
from ..io.constants import FIFF
from ..utils import logger, _check_fname

from ..externals.six import string_types
from ..externals.six.moves import map


class Montage(object):
    """Montage for EEG cap

    Montages are typically loaded from a file using read_montage. Only use this
    class directly if you're constructing a new montage.

    Parameters
    ----------
    pos : array, shape (n_channels, 3)
        The positions of the channels in 3d.
    ch_names : list
        The channel names.
    kind : str
        The type of montage (e.g. 'standard_1005').
    selection : array of int
        The indices of the selected channels in the montage file.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    def __init__(self, pos, ch_names, kind, selection):
        self.pos = pos
        self.ch_names = ch_names
        self.kind = kind
        self.selection = selection

    def __repr__(self):
        s = ('<Montage | %s - %d channels: %s ...>'
             % (self.kind, len(self.ch_names), ', '.join(self.ch_names[:3])))
        return s

    def plot(self, scale_factor=1.5, show_names=False):
        """Plot EEG sensor montage

        Parameters
        ----------
        scale_factor : float
            Determines the size of the points. Defaults to 1.5
        show_names : bool
            Whether to show the channel names. Defaults to False

        Returns
        -------
        fig : Instance of matplotlib.figure.Figure
            The figure object.
        """
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names)


def read_montage(kind, ch_names=None, path=None, unit='m', transform=False):
    """Read a generic (built-in) montage from a file

    This function can be used to read electrode positions from a user specified
    file using the `kind` and `path` parameters. Alternatively, use only the
    `kind` parameter to load one of the built-in montages:

    ===================   =====================================================
    Kind                  description
    ===================   =====================================================
    standard_1005         Electrodes are named and positioned according to the
                          international 10-05 system.
    standard_1020         Electrodes are named and positioned according to the
                          international 10-20 system.
    standard_alphabetic   Electrodes are named with LETTER-NUMBER combinations
                          (A1, B2, F4, etc.)
    standard_postfixed    Electrodes are named according to the international
                          10-20 system using postfixes for intermediate
                          positions.
    standard_prefixed     Electrodes are named according to the international
                          10-20 system using prefixes for intermediate
                          positions.
    standard_primed       Electrodes are named according to the international
                          10-20 system using prime marks (' and '') for
                          intermediate positions.

    biosemi16             BioSemi cap with 16 electrodes
    biosemi32             BioSemi cap with 32 electrodes
    biosemi64             BioSemi cap with 64 electrodes
    biosemi128            BioSemi cap with 128 electrodes
    biosemi160            BioSemi cap with 160 electrodes
    biosemi256            BioSemi cap with 256 electrodes

    easycap-M10           Brainproducts EasyCap with electrodes named
                          according to the 10-05 system
    easycap-M1            Brainproduct EasyCap with numbered electrodes

    EGI_256               Geodesic Sensor Net with 256 channels

    GSN-HydroCel-32       HydroCel Geodesic Sensor Net with 32 electrodes
    GSN-HydroCel-64_1.0   HydroCel Geodesic Sensor Net with 64 electrodes
    GSN-HydroCel-65_1.0   HydroCel Geodesic Sensor Net with 64 electrodes + Cz
    GSN-HydroCel-128      HydroCel Geodesic Sensor Net with 128 electrodes
    GSN-HydroCel-129      HydroCel Geodesic Sensor Net with 128 electrodes + Cz
    GSN-HydroCel-256      HydroCel Geodesic Sensor Net with 256 electrodes
    GSN-HydroCel-257      HydroCel Geodesic Sensor Net with 256 electrodes + Cz
    ===================   =====================================================

    Parameters
    ----------
    kind : str
        The name of the montage file (e.g. kind='easycap-M10' for
        'easycap-M10.txt'). Files with extensions '.elc', '.txt', '.csd',
        '.elp', '.hpts', '.sfp' or '.loc' ('.locs' and '.eloc') are supported.
    ch_names : list of str | None
        If not all electrodes defined in the montage are present in the EEG
        data, use this parameter to select subset of electrode positions to
        load. If None (default), all defined electrode positions are returned.
    path : str | None
        The path of the folder containing the montage file. Defaults to the
        mne/channels/data/montages folder in your mne-python installation.
    unit : 'm' | 'cm' | 'mm'
        Unit of the input file. If not 'm' (default), coordinates will be
        rescaled to 'm'.
    transform : bool
        If True, points will be transformed to Neuromag space.
        The fidicuals, 'nasion', 'lpa', 'rpa' must be specified in
        the montage file. Useful for points captured using Polhemus FastSCAN.
        Default is False.

    Returns
    -------
    montage : instance of Montage
        The montage.

    See Also
    --------
    read_dig_montage : To read subject-specific digitization information.

    Notes
    -----
    Built-in montages are not scaled or transformed by default.

    Montages can contain fiducial points in addition to electrode
    locations, e.g. ``biosemi-64`` contains 67 total channels.

    .. versionadded:: 0.9.0
    """

    if path is None:
        path = op.join(op.dirname(__file__), 'data', 'montages')
    if not op.isabs(kind):
        supported = ('.elc', '.txt', '.csd', '.sfp', '.elp', '.hpts', '.loc',
                     '.locs', '.eloc')
        montages = [op.splitext(f) for f in os.listdir(path)]
        montages = [m for m in montages if m[1] in supported and kind == m[0]]
        if len(montages) != 1:
            raise ValueError('Could not find the montage. Please provide the '
                             'full path.')
        kind, ext = montages[0]
        fname = op.join(path, kind + ext)
    else:
        kind, ext = op.splitext(kind)
        fname = op.join(path, kind + ext)

    if ext == '.sfp':
        # EGI geodesic
        dtype = np.dtype('S4, f8, f8, f8')
        data = np.loadtxt(fname, dtype=dtype)
        pos = np.c_[data['f1'], data['f2'], data['f3']]
        ch_names_ = data['f0'].astype(np.str)
    elif ext == '.elc':
        # 10-5 system
        ch_names_ = []
        pos = []
        with open(fname) as fid:
            for line in fid:
                if 'Positions\n' in line:
                    break
            pos = []
            for line in fid:
                if 'Labels\n' in line:
                    break
                pos.append(list(map(float, line.split())))
            for line in fid:
                if not line or not set(line) - set([' ']):
                    break
                ch_names_.append(line.strip(' ').strip('\n'))
        pos = np.array(pos)
    elif ext == '.txt':
        # easycap
        try:  # newer version
            data = np.genfromtxt(fname, dtype='str', skip_header=1)
        except TypeError:
            data = np.genfromtxt(fname, dtype='str', skiprows=1)
        ch_names_ = list(data[:, 0])
        theta, phi = data[:, 1].astype(float), data[:, 2].astype(float)
        x = 85. * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
        y = 85. * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = 85. * np.cos(np.deg2rad(theta))
        pos = np.c_[x, y, z]
    elif ext == '.csd':
        # CSD toolbox
        dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                 ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                 ('off_sph', 'f8')]
        try:  # newer version
            table = np.loadtxt(fname, skip_header=2, dtype=dtype)
        except TypeError:
            table = np.loadtxt(fname, skiprows=2, dtype=dtype)
        ch_names_ = table['label']
        theta = (2 * np.pi * table['theta']) / 360.
        phi = (2 * np.pi * table['phi']) / 360.
        pos = _sphere_to_cartesian(theta, phi, r=1.0)
        pos = np.asarray(pos).T
    elif ext == '.elp':
        # standard BESA spherical
        dtype = np.dtype('S8, S8, f8, f8, f8')
        try:
            data = np.loadtxt(fname, dtype=dtype, skip_header=1)
        except TypeError:
            data = np.loadtxt(fname, dtype=dtype, skiprows=1)

        az = data['f2']
        horiz = data['f3']

        radius = np.abs(az / 180.)
        angles = np.array([90. - h if a >= 0. else -90. - h
                           for h, a in zip(horiz, az)])

        sph_phi = (0.5 - radius) * 180.
        sph_theta = angles

        azimuth = sph_theta / 180.0 * np.pi
        elevation = sph_phi / 180.0 * np.pi
        r = 85.

        y, x, z = _sphere_to_cartesian(azimuth, elevation, r)

        pos = np.c_[x, y, z]
        ch_names_ = data['f1'].astype(np.str)
    elif ext == '.hpts':
        # MNE-C specified format for generic digitizer data
        dtype = [('type', 'S8'), ('name', 'S8'),
                 ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        data = np.loadtxt(fname, dtype=dtype)
        pos = np.vstack((data['x'], data['y'], data['z'])).T
        ch_names_ = data['name'].astype(np.str)
    elif ext in ('.loc', '.locs', '.eloc'):
        ch_names_ = np.loadtxt(fname, dtype='S4', usecols=[3]).tolist()
        dtype = {'names': ('angle', 'radius'), 'formats': ('f4', 'f4')}
        angle, radius = np.loadtxt(fname, dtype=dtype, usecols=[1, 2],
                                   unpack=True)

        sph_phi, sph_theta = _topo_to_sphere(angle, radius)

        azimuth = sph_theta / 180.0 * np.pi
        elevation = sph_phi / 180.0 * np.pi
        r = np.ones((len(ch_names_), ))

        x, y, z = _sphere_to_cartesian(azimuth, elevation, r)
        pos = np.c_[-y, x, z]
    else:
        raise ValueError('Currently the "%s" template is not supported.' %
                         kind)
    selection = np.arange(len(pos))

    if unit == 'mm':
        pos /= 1e3
    elif unit == 'cm':
        pos /= 1e2
    elif unit != 'm':
        raise ValueError("'unit' should be either 'm', 'cm', or 'mm'.")
    if transform:
        names_lower = [name.lower() for name in list(ch_names_)]
        if ext == '.hpts':
            fids = ('2', '1', '3')  # Alternate cardinal point names
        else:
            fids = ('nz', 'lpa', 'rpa')

        missing = [name for name in fids
                   if name not in names_lower]
        if missing:
            raise ValueError("The points %s are missing, but are needed "
                             "to transform the points to the MNE coordinate "
                             "system. Either add the points, or read the "
                             "montage with transform=False. " % missing)
        nasion = pos[names_lower.index(fids[0])]
        lpa = pos[names_lower.index(fids[1])]
        rpa = pos[names_lower.index(fids[2])]

        neuromag_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        pos = apply_trans(neuromag_trans, pos)

    if ch_names is not None:
        sel, ch_names_ = zip(*[(i, e) for i, e in enumerate(ch_names_)
                             if e in ch_names])
        sel = list(sel)
        pos = pos[sel]
        selection = selection[sel]
    else:
        ch_names_ = list(ch_names_)
    kind = op.split(kind)[-1]
    return Montage(pos=pos, ch_names=ch_names_, kind=kind, selection=selection)


class DigMontage(object):
    """Montage for Digitized data

    Montages are typically loaded from a file using read_dig_montage. Only use
    this class directly if you're constructing a new montage.

    Parameters
    ----------
    hsp : array, shape (n_points, 3)
        The positions of the headshape points in 3d.
        These points are in the native digitizer space.
    hpi : array, shape (n_hpi, 3)
        The positions of the head-position indicator coils in 3d.
        These points are in the MEG device space.
    elp : array, shape (n_hpi, 3)
        The positions of the head-position indicator coils in 3d.
        This is typically in the native digitizer space.
    point_names : list, shape (n_elp)
        The names of the digitized points for hpi and elp.
    nasion : array, shape (1, 3)
        The position of the nasion fidicual point.
    lpa : array, shape (1, 3)
        The position of the left periauricular fidicual point.
    rpa : array, shape (1, 3)
        The position of the right periauricular fidicual point.
    dev_head_t : array, shape (4, 4)
        A Device-to-Head transformation matrix.
    dig_ch_pos : dict
        Dictionary of channel positions.

        .. versionadded:: 0.12

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    def __init__(self, hsp, hpi, elp, point_names,
                 nasion=None, lpa=None, rpa=None, dev_head_t=None,
                 dig_ch_pos=None):
        self.hsp = hsp
        self.hpi = hpi
        self.elp = elp
        self.point_names = point_names

        self.nasion = nasion
        self.lpa = lpa
        self.rpa = rpa
        if dev_head_t is None:
            self.dev_head_t = np.identity(4)
        else:
            self.dev_head_t = dev_head_t
        self.dig_ch_pos = dig_ch_pos

    def __repr__(self):
        s = '<DigMontage | %d Dig Points, %d HPI points: %s ...>'
        s %= (len(self.hsp), len(self.point_names),
              ', '.join(self.point_names[:3]))
        return s

    def plot(self, scale_factor=1.5, show_names=False):
        """Plot EEG sensor montage

        Parameters
        ----------
        scale_factor : float
            Determines the size of the points. Defaults to 1.5
        show_names : bool
            Whether to show the channel names. Defaults to False

        Returns
        -------
        fig : Instance of matplotlib.figure.Figure
            The figure object.
        """
        from ..viz import plot_montage
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names)


_cardinal_ident_mapping = {
    FIFF.FIFFV_POINT_NASION: 'nasion',
    FIFF.FIFFV_POINT_LPA: 'lpa',
    FIFF.FIFFV_POINT_RPA: 'rpa',
}


def _check_frame(d, frame_str):
    """Helper to check coordinate frames"""
    if d['coord_frame'] != _str_to_frame[frame_str]:
        raise RuntimeError('dig point must be in %s coordinate frame, got %s'
                           % (frame_str, _frame_to_str[d['coord_frame']]))


def read_dig_montage(hsp=None, hpi=None, elp=None, point_names=None,
                     unit='mm', fif=None, transform=True, dev_head_t=False):
    """Read subject-specific digitization montage from a file

    Parameters
    ----------
    hsp : None | str | array, shape (n_points, 3)
        If str, this corresponds to the filename of the headshape points.
        This is typically used with the Polhemus FastSCAN system.
        If numpy.array, this corresponds to an array of positions of the
        headshape points in 3d. These points are in the native
        digitizer space.
    hpi : None | str | array, shape (n_hpi, 3)
        If str, this corresponds to the filename of Head Position Indicator
        (HPI) points. If numpy.array, this corresponds to an array
        of HPI points. These points are in device space.
    elp : None | str | array, shape (n_fids + n_hpi, 3)
        If str, this corresponds to the filename of electrode position
        points. This is typically used with the Polhemus FastSCAN system.
        Fiducials should be listed first: nasion, left periauricular point,
        right periauricular point, then the points corresponding to the HPI.
        These points are in the native digitizer space.
        If numpy.array, this corresponds to an array of fids + HPI points.
    point_names : None | list
        If list, this corresponds to a list of point names. This must be
        specified if elp is defined.
    unit : 'm' | 'cm' | 'mm'
        Unit of the input file. If not 'm', coordinates will be rescaled
        to 'm'. Default is 'mm'. This is applied only for hsp and elp files.
    fif : str | None
        FIF file from which to read digitization locations.
        If str (filename), all other arguments are ignored.

        .. versionadded:: 0.12

    transform : bool
        If True, points will be transformed to Neuromag space.
        The fidicuals, 'nasion', 'lpa', 'rpa' must be specified in
        the montage file. Useful for points captured using Polhemus FastSCAN.
        Default is True.
    dev_head_t : bool
        If True, a Dev-to-Head transformation matrix will be added to the
        montage. To get a proper `dev_head_t`, the hpi and the elp points
        must be in the same order. If False, an identity matrix will be added
        to the montage. Default is False.


    Returns
    -------
    montage : instance of DigMontage
        The digitizer montage.

    See Also
    --------
    read_montage : Function to read generic EEG templates

    Notes
    -----
    All digitized points will be transformed to head-based coordinate system
    if transform is True and fiducials are present.

    .. versionadded:: 0.9.0
    """
    if not isinstance(unit, string_types) or unit not in('m', 'mm', 'cm'):
        raise ValueError('unit must be "m", "mm", or "cm"')
    scale = dict(m=1., mm=1e-3, cm=1e-2)[unit]
    dig_ch_pos = None
    fids = None
    if fif is not None:
        # Use a different code path
        if dev_head_t or not transform:
            raise ValueError('transform must be True and dev_head_t must be '
                             'False for FIF dig montage')
        if not all(x is None for x in (hsp, hpi, elp, point_names)):
            raise ValueError('hsp, hpi, elp, and point_names must all be None '
                             'if fif is not None')
        _check_fname(fif, overwrite=True, must_exist=True)
        # Load the dig data
        f, tree = fiff_open(fif)[:2]
        with f as fid:
            dig = _read_dig_fif(fid, tree)
        # Split up the dig points by category
        hsp = list()
        hpi = list()
        elp = list()
        point_names = list()
        fids = dict()
        dig_ch_pos = dict()
        for d in dig:
            if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
                _check_frame(d, 'head')
                fids[_cardinal_ident_mapping[d['ident']]] = d['r']
            elif d['kind'] == FIFF.FIFFV_POINT_HPI:
                _check_frame(d, 'head')
                hpi.append(d['r'])
                elp.append(d['r'])
                point_names.append('HPI%03d' % d['ident'])
            elif d['kind'] == FIFF.FIFFV_POINT_EXTRA:
                _check_frame(d, 'head')
                hsp.append(d['r'])
            elif d['kind'] == FIFF.FIFFV_POINT_EEG:
                _check_frame(d, 'head')
                dig_ch_pos['EEG%03d' % d['ident']] = d['r']
        fids = np.array([fids[key] for key in ('nasion', 'lpa', 'rpa')])
        hsp = np.array(hsp)
        hsp /= scale  # will be multiplied later
        elp = np.array(elp)
        elp /= scale  # will be multiplied later
        transform = False
    if isinstance(hsp, string_types):
        hsp = _read_dig_points(hsp)
    if hsp is not None:
        hsp = hsp * scale
    if isinstance(hpi, string_types):
        ext = op.splitext(hpi)[-1]
        if ext == '.txt':
            hpi = _read_dig_points(hpi)
        elif ext in ('.sqd', '.mrk'):
            from ..io.kit import read_mrk
            hpi = read_mrk(hpi)
        else:
            raise TypeError('HPI file is not supported.')
    if isinstance(elp, string_types):
        elp = _read_dig_points(elp)
    if elp is not None:
        if len(elp) != len(point_names):
            raise ValueError("The elp file contains %i points, but %i names "
                             "were specified." % (len(elp), len(point_names)))
        elp = elp * scale
    if transform:
        if elp is None:
            raise ValueError("ELP points are not specified. Points are needed "
                             "for transformation.")
        names_lower = [name.lower() for name in point_names]

        # check that all needed points are present
        missing = tuple(name for name in ('nasion', 'lpa', 'rpa')
                        if name not in names_lower)
        if missing:
            raise ValueError("The points %s are missing, but are needed "
                             "to transform the points to the MNE coordinate "
                             "system. Either add the points, or read the "
                             "montage with transform=False." % str(missing))

        nasion = elp[names_lower.index('nasion')]
        lpa = elp[names_lower.index('lpa')]
        rpa = elp[names_lower.index('rpa')]

        # remove fiducials from elp
        mask = np.ones(len(names_lower), dtype=bool)
        for fid in ['nasion', 'lpa', 'rpa']:
            mask[names_lower.index(fid)] = False
        elp = elp[mask]

        neuromag_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)

        fids = np.array([nasion, lpa, rpa])
        fids = apply_trans(neuromag_trans, fids)
        elp = apply_trans(neuromag_trans, elp)
        hsp = apply_trans(neuromag_trans, hsp)
    elif fids is None:
        fids = [None] * 3
    if dev_head_t:
        from ..coreg import fit_matched_points
        trans = fit_matched_points(tgt_pts=elp, src_pts=hpi, out='trans')
    else:
        trans = np.identity(4)

    return DigMontage(hsp, hpi, elp, point_names, fids[0], fids[1], fids[2],
                      trans, dig_ch_pos)


def _set_montage(info, montage, update_ch_names=False):
    """Apply montage to data.

    With a Montage, this function will replace the EEG channel names and
    locations with the values specified for the particular montage.

    With a DigMontage, this function will replace the digitizer info with
    the values specified for the particular montage.

    Usually, a montage is expected to contain the positions of all EEG
    electrodes and a warning is raised when this is not the case.

    Parameters
    ----------
    info : instance of Info
        The measurement info to update.
    montage : instance of Montage | instance of DigMontage
        The montage to apply.
    update_ch_names : bool
        If True, overwrite the info channel names with the ones from montage.

    Notes
    -----
    This function will change the info variable in place.
    """
    if isinstance(montage, Montage):
        if update_ch_names:
            info['ch_names'] = montage.ch_names
            info['chs'] = list()
            for ii, ch_name in enumerate(montage.ch_names):
                ch_info = {'cal': 1., 'logno': ii + 1, 'scanno': ii + 1,
                           'range': 1.0, 'unit_mul': 0, 'ch_name': ch_name,
                           'unit': FIFF.FIFF_UNIT_V, 'kind': FIFF.FIFFV_EEG_CH,
                           'coord_frame': FIFF.FIFFV_COORD_HEAD,
                           'coil_type': FIFF.FIFFV_COIL_EEG}
                info['chs'].append(ch_info)

        if not _contains_ch_type(info, 'eeg'):
            raise ValueError('No EEG channels found.')

        sensors_found = []
        for pos, ch_name in zip(montage.pos, montage.ch_names):
            if ch_name not in info['ch_names']:
                continue

            ch_idx = info['ch_names'].index(ch_name)
            info['ch_names'][ch_idx] = ch_name
            info['chs'][ch_idx]['loc'] = np.r_[pos, [0.] * 9]
            sensors_found.append(ch_idx)

        if len(sensors_found) == 0:
            raise ValueError('None of the sensors defined in the montage were '
                             'found in the info structure. Check the channel '
                             'names.')

        eeg_sensors = pick_types(info, meg=False, ref_meg=False, eeg=True,
                                 exclude=[])
        not_found = np.setdiff1d(eeg_sensors, sensors_found)
        if len(not_found) > 0:
            not_found_names = [info['ch_names'][ch] for ch in not_found]
            warnings.warn('The following EEG sensors did not have a position '
                          'specified in the selected montage: ' +
                          str(not_found_names) + '. Their position has been '
                          'left untouched.')

    elif isinstance(montage, DigMontage):
        dig = _make_dig_points(nasion=montage.nasion, lpa=montage.lpa,
                               rpa=montage.rpa, hpi=montage.hpi,
                               dig_points=montage.hsp,
                               dig_ch_pos=montage.dig_ch_pos)
        info['dig'] = dig
        info['dev_head_t']['trans'] = montage.dev_head_t
        if montage.dig_ch_pos is not None:  # update channel positions, too
            eeg_ref_pos = montage.dig_ch_pos.get('EEG000', np.zeros(3))
            did_set = np.zeros(len(info['ch_names']), bool)
            is_eeg = np.zeros(len(info['ch_names']), bool)
            is_eeg[pick_types(info, meg=False, eeg=True, exclude=())] = True
            for ch_name, ch_pos in montage.dig_ch_pos.items():
                if ch_name == 'EEG000':
                    continue
                if ch_name not in info['ch_names']:
                    raise RuntimeError('Montage channel %s not found in info'
                                       % ch_name)
                idx = info['ch_names'].index(ch_name)
                did_set[idx] = True
                this_loc = np.concatenate((ch_pos, eeg_ref_pos))
                info['chs'][idx]['loc'][:6] = this_loc
            did_not_set = [info['chs'][ii]['ch_name']
                           for ii in np.where(is_eeg & ~did_set)[0]]
            if len(did_not_set) > 0:
                logger.warning('Did not set %s channel positions:\n%s'
                               % (len(did_not_set), ', '.join(did_not_set)))
    else:
        raise TypeError("Montage must be a 'Montage' or 'DigMontage' "
                        "instead of '%s'." % type(montage))
