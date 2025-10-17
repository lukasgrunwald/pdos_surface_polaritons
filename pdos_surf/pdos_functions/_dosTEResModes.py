import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad

from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
import h5py
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.constants as consts

import pdos_surf.allowed_kz as allowed_kz
import pdos_surf.momentum_relations as epsFunc

fontsize = 8

mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 8
mpl.rc('text', usetex=True)

mpl.rcParams['text.latex.preamble'] =  r'\usepackage[helvet]{sfmath}'

def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
    kDArr = epsFunc.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    brack1 = 1 - np.sin(kArr * L) / (kArr * L)
    term1 = brack1 * 0.25 * (1 + np.exp(-2 * kDArr * L) - 2. * np.exp(- kDArr * L))
    brack2 = (1 - np.exp(- 2. * kDArr * L)) / (2. * kDArr * L) - np.exp(- kDArr * L)
    term2 = normPrefac * brack2 * np.sin(kArr * L / 2) ** 2
    return L / 4 * (term1 + term2)

def pdos_sum_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")
    kDArr = epsFunc.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kArr[None, :] * (L / 2. - zArr[:, None])) * 0.5 * (1 - np.exp(- kDArr[None, :] * L))
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def pdos_sum_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")
    kDArr = epsFunc.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
    func = 0.5 * (np.exp(kDArr[None, :] * zArr[:, None]) - np.exp(-kDArr[None, :] * zArr[:, None] - kDArr[None, :] * L)) * np.sin(kArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)


def calcDosTE(zArr, L, omega, wLO, wTO, epsInf):

    kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TERes")
    kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = pdos_sum_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg = pdos_sum_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos

def plotDosTESPhP(zArr, dos, L, omega, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z[\frac{c}{\omega}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosTERes.png")

def createPlotDosTERes():

    omega = 2. * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = .1

    zArr = np.linspace(-L / 2., L / 2., 500)

    dos = calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    plotDosTESPhP(zArr, dos, L, omega, wLO, wTO, epsInf)
