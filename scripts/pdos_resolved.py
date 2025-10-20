"""
Visualization of the different contributions
"""
##
import numpy as np
import matplotlib.pyplot as plt

from pdos_surf.pdos_per_mode import PdosTE
from pdos_surf.pdos_per_mode import PdosTM
from pdos_surf.pdos_per_mode import PdosSurf, PdosSurfAnalytic
from pdos_surf.pdos_per_mode import PdosEvaTE, PdosEvaTM
from pdos_surf.pdos_per_mode import PdosResTE, PdosResTM

import pdos_surf.momentum_relations as momentum_relations
from pdos_surf.util import CM

def plot_pdos_TE():
    omega = 4 * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 4.
    L = .1

    zArr = np.linspace(-L / 2000., L / 2000., 500)
    dos = PdosTE.pdos(zArr, L, omega, wLO, wTO, epsInf)

    fig, ax = plt.subplots(figsize = (8 * CM, 5 * CM))
    eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')
    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.tight_layout()
    plt.savefig("./figures/pdos_z/pdos_TE.png")
    plt.show()

def plot_pdos_TM():
    omega = 0.75 * 1e11
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 1.
    L = 3.0

    zArr = np.linspace(-L / 2., L / 2., 1000)
    dos = PdosTM.pdos(zArr, L, omega, wLO, wTO, epsInf)

    fig, ax = plt.subplots(figsize = (8 * CM, 5 * CM))
    eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))


    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z[\frac{c}{\omega}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)

    plt.savefig("../figures/pdos_z/pdos_TM.png")

def plot_pdos_Surf():
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    omega = 2.4 * 1e12
    print("wInf = {}".format(wInf * 1e-12))

    L = 1.
    zArr = np.linspace(-L / 1000., L / 1000., 1000)

    dos_para, dos_perp = PdosSurf.pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf)
    dosAna = PdosSurfAnalytic.pdos(zArr, omega, wLO, wTO, epsInf)

    fig, ax = plt.subplots(figsize = (8 * CM, 5 * CM), dpi = 300)

    ax.plot(zArr, dos_para + dos_perp, color='peru', lw=1.)
    ax.plot(zArr, dos_para, color='tab:blue', lw=1., label = "para")
    ax.plot(zArr, dos_perp, color='tab:red', lw=1., label = "perp")
    ax.plot(zArr, dosAna, color='steelblue', lw=1., linestyle = '--')
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{1000}$", r"$0$", r"$\frac{L}{1000}$"])

    ax.set_xlabel(r"$z[\frac{c}{\omega}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)

    plt.tight_layout()
    # plt.savefig("./figures/pdos_z/pdos_surf.png")
    plt.show()

def plot_pdos_EvaTE():
    omega = 0.75 * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = .1
    eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
    zArr = np.linspace(-L / 2., L / 2., 500)
    dos = PdosEvaTE.pdos(zArr, L, omega, wLO, wTO, epsInf)

    fig, ax = plt.subplots(figsize = (8 * CM, 5 * CM), dpi = 300)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    ax.set_xticks([- L / 2., 0., L / 2.])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z[\mathrm{m}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.tight_layout()
    plt.savefig("./figures/pdos_z/pdos_TEEva.png")
    plt.show()

def plot_pdos_EvaTM():
    omega = 1 * 1e11
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = 1.

    zArr = np.linspace(-L / 2., L / 2., 1000)
    eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
    dos = PdosEvaTM.pdos(zArr, L, omega, wLO, wTO, epsInf)

    fig, ax = plt.subplots(figsize = (8 * CM, 5 * CM))

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\rho / \rho_0$")


    plt.tight_layout()
    plt.savefig("./figures/pdos_z/pdos_TEEva.png")
    plt.show()

def plot_pdos_ResTE():
    omega = 2. * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = 2

    zArr = np.linspace(-L / 2., L / 2., 500)
    dos = PdosResTE.pdos(zArr, L, omega, wLO, wTO, epsInf)


    fig, ax = plt.subplots(figsize = (8 * CM, 5 * CM))
    eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')
    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))
    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z[\frac{c}{\omega}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.tight_layout()
    plt.savefig("./figures/pdos_z/pdos_TERes.png")
    plt.show()

def plot_pdos_ResTM():
    omega = 4 * 1e12
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = 5.

    zArr = np.linspace(-L / 2., L / 2., 1000)
    eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
    dos = PdosResTM.pdos(zArr, L, omega, wLO, wTO, epsInf)

    fig, ax = plt.subplots(figsize = (8 * CM, 5 * CM))

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))

    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    plt.tight_layout()
    plt.savefig("./figures/pdos_z/pdos_TMRes.png")
    plt.show()
