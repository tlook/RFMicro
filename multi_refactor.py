"""
This file is part of Indoor Millimeter Wave MIMO Simulator (IMWMS)

This simulation is based on the work done by E. Torkildson, et al. in
"Indoor Millimeter Wave MIMO: Feasibility and Performance"
relevant referenced equation numbers can be found here unless otherwise noted:
http://ieeexplore.ieee.org/document/6042312/

timer.py taken from spinsys

"""

import numpy as np
from scipy import constants as con
import matplotlib.pyplot as plt
import timer as mactime


try:
    import cPickle as pickle
except ImportError:
    import pickle
with open('interpolator.pkl', 'rb') as f:
    qam_cap = pickle.load(f)  # Import 16-QAM approximation


""" conversion or shortcut functions """


def dBm2W(dBm):
    """
    This function takes power in dBm and converts to Watts
    :param dBm:
    :return watts:
    """
    watts = 10 ** ((dBm - 30) / 10)
    return watts


def dB2gain(dB):
    """
    This function takes power gain in decibels and converts to linear scale
    :param dB:
    :return gain:
    """
    gain = 10 ** (dB / 10.0)
    return gain


def gain2dB(gain):
    """
    This function takes linear scale gain and converts to decibels
    :param gain:
    :return db:
    """
    db = 10*np.log10(gain)
    return db


def abs2(x):
    """
    This function takes a vector x and finds the |x|^2 value
    :param x:
    :return mag2:
    """
    mag2 = x.real**2 + x.imag**2
    return mag2

""" General Room & Antenna Globals """
room_dim = [5.0, 5.0, 3.0]  # room dimension in meters
Tx = [2.5, 2.5, 1.5]  # center of transmit array, modified in later functions
Rx = [2.5, 0, 1.5]  # center of receive array, fixed in all calculations
N = 2  # number of subarrays
M = 4  # each subarray as MxM elements spaced wavelength/2 in a square or M x 1 line
wavelength = 5e-3  # wavelength in meters
freq = 60e9  # frequency in hertz
B = 2.16e9  # bandwidth in hertz
T = 300  # temperature in kelvin
subspace = np.sqrt(2.5 * wavelength / N)  # spacing between subarrays optimized for R = 2.5
arr_length = (N - 1) * subspace + 1.5 * wavelength  # total array length
Pa = -10  # power per transmit antenna in dBm
Pt = dBm2W(Pa) * N * (M ** 2)  # total power in watts
Pn = con.k * T * dB2gain(10) * B  # noise power at receiver Pn = kTBF
# where k is Boltzmann constant, T is temp in Kelvin, B is bandwidth in hertz,
# and F is noise factor in watts
nrm = wavelength / (4 * np.pi)  # normalization factor used in alter calcs

""" Wall properties """
epsilon = 2.8  # relative dielectric for plasterboard
conduct = 0.221  # conductivity of plasterboard
comp_epsilon = epsilon - (1j * conduct) / (2 * np.pi * freq * con.epsilon_0)  # complex valued
rx_array = np.zeros([1, 3])
tx_array = np.zeros([1, 3])


def gen_ant_index_sq():
    """
    Generates an array for the antenna index of a M x M square array with
    elements [j, k, l] being the jth subarry, kth column, and lth row.
    """
    return np.array([(j+1, k+1, l+1) for j in range(N) for k in range(M) for l in range(M)])


def gen_ant_index_1m():
    """ Generates an index array for 1xM subarrays """
    return np.array([(j+1, k+1, 1)] for j in range(N) for k in range(M))


def gen_ant_index_m1():
    """ Generate an index array for Mx1 subarrays """
    return np.array([(j+1, 1, l+1)] for j in range(N) for l in range(M))


def r_coef(theta, orientation):
    """
    The Fresnel reflection coefficients as given by equations (3.24) & (3.25)
    in T. S. Rappaport "Wireless Communications: Principles and Practice".
    comp_epsilon, defined globally is the complex form of the dielectric const.
    orientation is 0 for perpendicular E field to plane of incidence or 1 for
    parallel
    """
    assert 0 <= theta <= (np.pi / 2)
    a = np.sin(theta)
    b = np.sqrt(comp_epsilon - (np.cos(theta) ** 2))
    if theta == 0:
        gamma = 1
    elif orientation == 0 and theta != 0:  # for perpendicular coefficient off side walls
        gamma = (a - b) / (a + b)
    elif orientation == 1 and theta != 0:  # for parallel coefficient of ceiling
        gamma = (b - comp_epsilon * a) / (b + comp_epsilon * a)
    assert -1 <= gamma <= 1
    return gamma


def ula_pos(coord):
    """
    Generate a ULA with optimal spacing subspace = np.sqrt(2.5 * wavelength / N)
    with 2.5 corresponding to an expected distance of R=2.5 meters for a Tx in
    the middle of a 5 x 5 x 3 room
    """
    assert len(coord) == 3
    # generate optimally spaced ULA with N elements
    ula = np.empty([N, 3])
    for j in range(N):
        ula[j] = coord
        ula[j, 0] += ((j + 1) - (N + 1) / 2) * subspace
    return ula


def subarray_ant_pos_sq(ula):
    """
    generate M x M subarrays centered at each ULA element
    """
    positions = []
    for j in range(N):
        for k in range(M):
            for l in range(M):
                positions.append([ula[j][0] + wavelength * (2 * k - M + 1) / 4,
                                  ula[j][1],
                                  ula[j][2] - wavelength * (2 * l - M + 1) / 4])
    return np.array(positions)


def aoi(tvec, unitnormal):
    """
    calculate angle of incidence given transmit vector and unit normal vector
    of the wall.  Since the angle between these vectors is complementary to the
    angle of incidence, it is subtracted from pi/2.  Angles given in radians.
    tvec and unitnormal are 3-tuples.
    """
    theta = .5 * np.pi - np.arccos(np.dot(tvec, unitnormal))
    assert 0 <= theta <= (np.pi / 2)
    return theta


def get_tvec(tx_pos, rx_pos):
    """
    calculate the relative position vector between Tx element and Rx element
    points from Tx to Rx (or virtual Rx in case of reflection path)
    """
    tvec = rx_pos - tx_pos
    return tvec


def gen_v_pos(rx_pos):
    """
    given the position of a Rx element, generate virtual array positions for
    first and second order reflections
    """
    vrx_pos = np.empty([8, 3])
    x0, y0, z0 = room_dim
    rx_x, rx_y, rx_z = rx_pos
    vrx_pos[0] = rx_pos  # LOS
    vrx_pos[1] = [-rx_x, rx_y, rx_z]  # Reflect off left wall
    vrx_pos[2] = [-rx_x + 2 * x0, rx_y, rx_z]  # Reflect off right wall
    vrx_pos[3] = [rx_x, rx_y, -rx_z + 2 * z0]  # Reflect off ceiling
    vrx_pos[4] = [rx_x - 2 * x0, rx_y, rx_z]  # Reflect off left then right wall
    vrx_pos[5] = [rx_x + 2 * x0, rx_y, rx_z]  # Reflect off right then left wall
    vrx_pos[6] = [- rx_x, rx_y, -rx_z + 2 * z0]  # Reflect off left then ceiling
    vrx_pos[7] = [- rx_x + 2 * x0, rx_y, -rx_z + 2 * z0]  # Reflect off right wall then ceiling
    return vrx_pos


def get_path_params(mth_rx, nth_tx):
    """
    given coordinates of an Rx and Tx element, find the total path distance
    and relevant angles of incidence
    elements are 3-tuples (path length, angle of incidence 1, angle of incidence 2)
    Paths are as follows, with (R)ight wall, (L)eft wall, (C)eiling:
    0: LOS, 1: L, 2: R, 3: C, 4: LR, 5: RL, 6: LC, 7: RC
    """
    path = np.empty([8, 4])
    xvec = np.array([1, 0, 0])
    zvec = np.array([0, 0, 1])
    tx_pos = tx_array[nth_tx]
    rx_pos = gen_v_pos(rx_array[mth_rx])
    for i in range(8):
        tvec = get_tvec(tx_pos, rx_pos[i])
        tmag = np.linalg.norm(tvec)
        tvec /= tmag
        a1 = 0
        a2 = 0
        beta = find_elev(tvec)
        if i == 0:
            a1 = 0
        elif i == 1:
            a1 = aoi(tvec, -xvec)
        elif i == 2:
            a1 = aoi(tvec, xvec)
        elif i == 3:
            a1 = aoi(tvec, zvec)
        elif i == 4:
            a1 = aoi(tvec, -xvec)
            a2 = a1
        elif i == 5:
            a1 = aoi(tvec, xvec)
            a2 = a1
        elif i == 6:
            a1 = aoi(tvec, -xvec)
            a2 = aoi(tvec, zvec)
        elif i == 7:
            a1 = aoi(tvec, xvec)
            a2 = aoi(tvec, zvec)
        path[i] = [tmag, a1, a2, beta]
    return path


def find_steering_angles(tvec):
    """
    given a transmission vector, solve for theta (azimuthal) and phi (polar)
    steering angles
    """
    x, y, z = tvec
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return theta, phi


def find_elev(tvec):
    """
    find the angle of elevation from the broadside
    """
    x, y, z = tvec
    beta = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return beta


def rad_fade(beta, beta0):
    """
    calculate the reduction in amplitude due to radiation pattern
    """
    return np.cos(beta - beta0)


def h_matrix_element(mth_rx, nth_tx):
    """
    calculate the matrix element between the mth receiver and nth transmitter
    the first return is for the LOS component, the second return is for first
    order reflections, the last return is second order reflections.
    Factors fade due to path loss, phase due to total distance of the ray, and
    Fresnel coefficients for reflections.
    See equations (27), (28), (29), (30)
    """
    path = get_path_params(mth_rx, nth_tx)
    alpha = (-2j * np.pi / wavelength)
    hmn = np.empty(8, dtype=complex)
    for i in range(8):
        pmn, a1, a2, beta = path[i]
        or1 = 0
        or2 = 0
        if i == 3:
            or1 = 1
        if i == 6 or i == 7:
            or2 = 1
        hmn[i] =  nrm * r_coef(a1, or1) * r_coef(a2, or2) * np.exp(alpha * pmn) / pmn
    return hmn[0], sum(hmn[1:4]), sum(hmn[4:8])


def hmn_w_radpat(mth_rx, nth_tx, beta0):
    """
    Same as h_matrix_element, but with antenna pattern considered
    """
    path = get_path_params(mth_rx, nth_tx)
    alpha = (-2j * np.pi / wavelength)
    hmn = np.empty(8, dtype=complex)
    for i in range(8):
        pmn, a1, a2, beta = path[i]
        or1 = 0
        or2 = 0
        if i == 3:
            or1 = 1
        if i == 6 or i == 7:
            or2 = 1
        hmn[i] = rad_fade(beta, beta0[i]) * nrm * r_coef(a1, or1) * r_coef(a2, or2) * np.exp(alpha * pmn) / pmn
    return hmn[0], sum(hmn[1:4]), sum(hmn[4:8])


def generate_channel_matrix(subtype = 'square'):
    """
    Generates a channel matrix based on global variables Rx and Tx which are the
    positions of receiver and transmitter array.
    See equation (27), returns are the three terms
    """
    ula_rx = ula_pos(Rx)
    ula_tx = ula_pos(Tx)
    global rx_array, tx_array
    if subtype == 'square':
        rx_array = subarray_ant_pos_sq(ula_rx)
        tx_array = subarray_ant_pos_sq(ula_tx)
        dim = N * (M ** 2)
    elif subtype == '1m':
        rx_array = subarray_ant_pos_sq(ula_rx)
        tx_array = subarray_ant_pos_sq(ula_tx)
        dim = N * M
    elif subtype == 'm1':
        rx_array = subarray_ant_pos_sq(ula_rx)
        tx_array = subarray_ant_pos_sq(ula_tx)
        dim = N * M
    hlos = np.empty([dim, dim], dtype=complex)
    h1 = np.empty([dim, dim], dtype=complex)
    h2 = np.empty([dim, dim], dtype=complex)
    for i in range(dim):
        for j in range(dim):
            hlos[i, j], h1[i, j], h2[i, j] = h_matrix_element(i, j)
    return hlos, h1, h2


def chan_cap(singular_values, mu):
    """
    Shannon capacity for waterfilling power allocation scheme where mu is the
    water level and singular_values is the first N singular values of the
    channel matrix
    See equation (19)
    """
    s2 = (mu / Pn) * np.square(singular_values)
    sings = np.array([_ if _ > 1 else 1 for _ in s2])
    return sum(np.log2(sings))


def waterfilling(singular_values):
    """
    water filling algorithm for determining water level and power allocation
    for water filling benchmark.  Input is the first N singular values of the
    channel matrix
    """
    rem_chan = 0
    s2 = Pn / np.square(singular_values)
    mu = s2[-1]
    a = s2.size
    power = [(mu - sv) if (mu - sv) > 0 else 0 for sv in s2]
    while (sum(power) > Pt) and (rem_chan < a):
        rem_chan += 1
        mu = s2[-rem_chan -1]
        power = [(mu - sv) if (mu - sv) > 0 else 0 for sv in s2]
    residual = (Pt - sum(power)) / (a - rem_chan)
    for i in range(a - rem_chan):
        power[i] += residual
    mu += residual
    return power, mu


def bs_weight(theta_j, phi_j, col, row):
    """
    beamsteering weights for a square phased array given steering angels theta
    and phi, and the col and row of the element
    See equation (21)
    """
    ct = np.cos(theta_j)
    sp = np.sin(phi_j)
    cp = np.cos(phi_j)
    a = np.exp(-1j * np.pi * ((col - 1) * ct * sp + (row - 1) * cp))
    return a


def gen_steering_matrix(angles):
    """
    take theta (azimuthal) and phi (polar) steering angles and generate
    steering matrix as given by equation (23)
    """
    phase = np.matrix(np.zeros([N * (M ** 2), N], dtype=complex))
    sub_ind = np.array([(k + 1, l + 1) for k in range(M) for l in range(M)])
    for j in range(N):
        theta_j, phi_j = angles[j]
        for i in range(M ** 2):
            col, row = sub_ind[i]
            phase[i + (j * M ** 2), j] = bs_weight(theta_j, phi_j, col, row)
    return phase / M


def gen_angle_list():
    """
    Generate a list of steering angles for each Tx subarray to be beamsteered
    along.  First, the receiver virtual array positions are made for LOS and
    first order reflections.  Steering angles are then calculated for each Tx
    array to each virtual receiver.
    """
    ula_tx = ula_pos(Tx)
    ula_rx = ula_pos(Rx)
    length = ula_rx.shape[0]
    v_rx = np.empty([length * 4, 3])
    for i in range(length):
        v_rx[i * 4: i * 4 + 4, ] = gen_v_pos(ula_rx[i])[0:4]
    angles = np.array([(find_steering_angles(get_tvec(i, v_rx[j])), j) for i in ula_tx for j in range(v_rx.shape[0])])
    return angles


def C_MMSE(F):
    """
    Create the MMSE equalizer from which to calculate the signal to interference
    and noise ratio from.  Given by equation () and here F is H^hat
    """
    a = (Pt/N) * (((Pt/N) * (F * F.H) + (Pn * np.identity(N * M ** 2))).I * F)
    return a


def SINR_k(C, F, k):
    """
    Calculate the signal to interference and noise ratio as given by equation
    ()
    """
    a = Pt/N
    b = a * abs2(C[:,k].H * F[:,k])
    c = sum([a * abs2(C[:,j].H * F[:,k]) for j in range(C.shape[1])]) - b + Pn * (C[:,k].H * C[:,k]).real
    return b / c


def spec_eff(C, F):
    """
    Use 16-QAM to calculate the sum rate spectral efficiency from the signal to
    interference and noise ratios given by
    """
    a = [SINR_k(C, F, k) if SINR_k(C, F, k) < 1000 else 1000 for k in range(N)]
    b = sum([qam_cap(_) for _ in a])
    return b


def gen_cc_mesh(xpoints, ypoints):
    """
    Genereate mesh for contour and color plotting for waterfilling benchmark
    """
    timer = mactime.Timer(xpoints * ypoints)
    xlist = np.linspace(0, room_dim[0], xpoints)
    ylist = np.linspace(0, room_dim[1], ypoints)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.empty(X.shape)
    Z2 = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            global Tx
            Tx = [X[i, j], Y[i, j], 1.5]
            HL, H1, H2 = generate_channel_matrix()
            H = HL + H1 + H2
            U, s, V = np.linalg.svd(H)
            s = s[:N,]
            opt, mu = waterfilling(s)
            Z[i, j] = chan_cap(s, mu)
            H = H1 + H2
            U, s, V = np.linalg.svd(H)
            s = s[:N,]
            opt, mu = waterfilling(s)
            Z2[i, j] = chan_cap(s, mu)
            timer.progress()
    return X, Y, Z, Z2


def plot_contour_wf():
    X, Y, Z, Z2 = gen_cc_mesh(200, 200)
    levels = np.linspace(12.18, 34.74, 13)
    plt.figure(1)
    cp = plt.contour(X, Y, Z, levels, linewidths=.5)
    plt.clabel(cp, colors='k', fontsize=10)
    cb = plt.colorbar(cp)
    cb.set_label('Channel Capacity (bps/Hz)')
    plt.title('Channel Capacity as a Function of Transmitter Position')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.figure(2)
    cp1 = plt.pcolor(X, Y, Z2)
    cb1 = plt.colorbar(cp1)
    cb1.set_label('Channel Capacity (bps/Hz)')
    plt.title('No-LOS Channel Capacity as a Function of Transmitter Position')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.show()
    return


def optimal_steering():
    angles = gen_angle_list()
    HL, H1, H2 = generate_channel_matrix()
    H = HL + H1 + H2
    what = np.empty([(4 * N) ** 2, 3])
    k = 0
    for i in range(N * 4):
        for j in range(N * 4):
            a = np.append([angles[i, 0]], [angles[j+(N*4), 0]], axis=0)
            G = gen_steering_matrix(a)
            F = H * G
            C = C_MMSE(F)
            k1 = spec_eff(C, F)
            what[k] = [k1, i % 4, j % 4]
            k += 1
    print(what)
    rofl = np.amax(what, axis=0)
    return rofl


def gen_se_mesh(xpoints, ypoints):
    timer = mactime.Timer(xpoints * ypoints)
    xlist = np.linspace(0, room_dim[0], xpoints)
    ylist = np.linspace(0, room_dim[1], ypoints)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.empty(X.shape)
    Z2 = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            global Tx
            Tx = [X[i, j], Y[i, j], 1.5]
            Z[i, j], k1, k2 = optimal_steering()
            if k1 == 1:
                if k2 == 1:
                    Z2[i,j] = 1 # LL
                elif k2 == 2:
                    Z2[i,j] = 2 # OP
                elif k2 == 3:
                    Z2[i,j] = 3 # LC
                else:
                    Z2[i,j] = 0 # LOS?
            elif k1 == 2:
                if k2 == 1:
                    Z2[i,j] = 2 # OP
                elif k2 == 2:
                    Z2[i,j] = 4 # RR
                elif k2 == 3:
                    Z2[i,j] = 5 # RC
                else:
                    Z2[i,j] = 0
            elif k1 == 3:
                if k2 == 1:
                    Z2[i,j] = 3 # LC
                elif k2 == 2:
                    Z2[i,j] = 5 # RC
                elif k2 == 3:
                    Z2[i,j] = 6 # CC
                else:
                    Z2[i,j] = 0
            else:
                Z2[i,j] = 0
            timer.progress()
    return X, Y, Z, Z2


def plot_MMSE():
    X, Y, Z, Z2 = gen_se_mesh(200, 200)
    plt.figure(1)
    cp1 = plt.pcolor(X, Y, Z)
    cb1 = plt.colorbar(cp1)
    cb1.set_label('Channel Capacity (bps/Hz)')
    plt.title('Beamsteering/MMSE w/ LOS Blockage')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.figure(2)
    cp2 = plt.pcolor(X, Y, Z2)
    cb2 = plt.colorbar(cp2)
    cb2.set_label('Beamsteer Path')
    plt.title('Beamsteering/MMSE w/ LOS Blockage Optimal Direction')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.show()
    return


plot_MMSE()
