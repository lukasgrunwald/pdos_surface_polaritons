import numpy as np
import scipy.constants as consts
import scipy

import pdos_surf.momentum_relations as epsFunc

# —————————————————————————————————————————————————————————————————————————————————————— #
#                                     Root functions                                     #
# —————————————————————————————————————————————————————————————————————————————————————— #

# —————————————————————————————————————— TE-modes —————————————————————————————————————— #
def rootFuncTE(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromK(k, omega, wLO, wTO, epsInf)
    term1 = k * np.cos(k * L / 2.) * np.sin(kD * L / 2.)
    term2 = kD * np.cos(kD * L / 2.) * np.sin(k * L / 2.)
    return  term1 + term2

def rootFuncTEEva(kD, L, omega, wLO, wTO, epsInf):
    kVal = epsFunc.kDFromKEva(kD, omega, wLO, wTO, epsInf)
    term1 = kVal * np.sin(kD * L / 2)
    term2 = kD * np.cos(kD * L / 2) * np.tanh(kVal * L / 2)
    return term1 + term2

def rootFuncTERes(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKRes(k, omega, wLO, wTO, epsInf)
    term1 = k * np.cos(k * L / 2) * np.tanh(kD * L / 2)
    term2 = kD * np.sin(k * L / 2)
    return term1 + term2

# —————————————————————————————————————— TM-modes —————————————————————————————————————— #
def rootFuncTM(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromK(k, omega,wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    term1 = eps * k / kD * np.sin(k * L / 2.) * np.cos(kD * L / 2.)
    term2 = np.cos(k * L / 2.) * np.sin(kD * L / 2)
    return term1 + term2

def rootFuncTMEva(kD, L, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    kVal = epsFunc.kDFromKEva(kD, omega, wLO, wTO, epsInf)
    term1 = eps * kVal * np.tanh(kVal * L / 2) * np.cos(kD * L / 2)
    term2 = kD * np.sin(kD * L / 2)
    return term1 - term2

def rootFuncTMRes(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKRes(k, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    term1 = eps * k / kD * np.sin(k * L / 2)
    term2 = np.cos(k * L / 2) * np.tanh(kD * L / 2)
    return term1 - term2

# ——————————————————————————————————————— Surface —————————————————————————————————————— #
def rootFuncSurf(k, L, omega, wLO, wTO, epsInf):
    kD = epsFunc.kDFromKSurf(k, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    term1 = eps * k / kD * np.tanh(k * L / 2)
    term2 = np.tanh(kD * L / 2)
    return term1 + term2

def allowedKSurf(L, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    root = scipy.optimize.root_scalar(rootFuncSurf, args=(L, omega, wLO, wTO, epsInf),
                                          bracket=tuple([0., omega / consts.c * (10 + 10 / (epsAbs - 1))]))
    return np.array([root.root])

def findKsSurf(L, omega, wLO, wTO, epsInf):
    kVals = allowedKSurf(L, omega, wLO, wTO, epsInf)
    if (len(kVals) == 0):
        return 0
    else:
        return kVals[0]

def findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf):
    delOm = omega * 1e-8
    rootPlus = allowedKSurf(L, omega + delOm, wLO, wTO, epsInf)[0]
    rootMinus = allowedKSurf(L, omega - delOm, wLO, wTO, epsInf)[0]

    return (rootPlus - rootMinus) / (2. * delOm)

# ——————————————————————————————————————— General —————————————————————————————————————— #
def getUpperBound(mode, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    upperBound = 0
    if (mode == "TE" or mode == "TM" or mode == "Surf"):
            upperBound = omega / consts.c
    elif (mode == "TEEva" or mode == "TMEva"):
        upperBound = np.sqrt(eps - 1) * omega / consts.c - 1e-5
    elif (mode == "TERes" or mode == "TMRes"):
        if (eps < 0):
            upperBound = omega / consts.c
        elif(eps < 1 and eps > 0):
            upperBound = np.sqrt(1 - eps) * omega / consts.c - 1e-5

    return upperBound

def getLowerBound(mode, omega, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    lowerBound = 0
    if(eps >= 0 and eps <= 1):
        if (mode == "TE" or mode == "TM"):
            lowerBound = np.sqrt(1 - eps) * omega / consts.c + 1e-5

    return lowerBound

def getRoots(L, omega, wLO, wTO, epsInf, mode):
    rootFunc = rootFuncTE
    if(mode == "TE"):
        rootFunc = rootFuncTE
    elif (mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva
    elif (mode == "TERes"):
        rootFunc = rootFuncTERes
    elif (mode == "TMRes"):
        rootFunc = rootFuncTMRes
    elif (mode == "Surf"):
        rootFunc = rootFuncSurf
    else:
        print("Error: specified mode doesn't exist!")
        exit()

    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    lowerBound = getLowerBound(mode, omega, wLO, wTO, epsInf)

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    NCoarse = 100
    coarseDisvision = np.linspace(lowerBound, upperBound, NCoarse, endpoint=True)
    iteration = 0
    maxIters = 20
    intervals = np.zeros((0, 2))
    for indCoarse in range(NCoarse - 1):
        subdivision = np.zeros(0)
        indsSigns = np.zeros(0)
        NDivision = int(1. * omega / consts.c * (np.sqrt(np.abs(eps)) + 1.) * L + 3) // NCoarse
        numRootsOld = -1
        numRootsOld2 = -1
        while(iteration < maxIters):
            #print("NDiscrete = {} at iteration {}".format(NDiscrete, iteration))
            subdivision = np.linspace(coarseDisvision[indCoarse], coarseDisvision[indCoarse + 1], NDivision, endpoint=True)
            rootFuncAtPoints = rootFunc(subdivision, L, omega, wLO, wTO, epsInf)
            signs = rootFuncAtPoints[:-1] * rootFuncAtPoints[1:]
            indsSigns = np.where(signs < 0)[0]
            #print("Numer of Roots = {} at iteration {}".format(len(indsSigns), iteration))
            if(numRootsOld == indsSigns.shape[0] and numRootsOld == numRootsOld2):
                break
            else:
                NDivision = int(1.5 * NDivision)
                numRootsOld2 = numRootsOld
                numRootsOld = indsSigns.shape[0]
                iteration += 1
        iteration = 0
        intervalsTemp = np.append([subdivision[indsSigns]], [subdivision[indsSigns + 1]], axis=0)
        #print("N IntervalsTemp.shape = {}".format(intervalsTemp.shape))
        intervals = np.append(intervals, np.swapaxes(intervalsTemp, 0, 1), axis = 0)

    if(iteration == maxIters):
        print("Warning: reach maximum number of iterations")

    #print("NDiscrete = {} after iter = {}".format(NDiscrete, iteration))
    #print("N Intervals.shape = {}".format(intervals.shape))

    nRoots = intervals.shape[0]
    roots = np.zeros(nRoots)
    for rootInd, root in enumerate(roots):
        tempRoot = scipy.optimize.root_scalar(rootFunc, args = (L, omega, wLO, wTO, epsInf), bracket=tuple(intervals[rootInd, :]))
        roots[rootInd] = tempRoot.root


    if (mode == "TEEva" or mode == "TMEva"):
        roots = epsFunc.kDFromKEva(roots, omega, wLO, wTO, epsInf)

    if(mode == "TE" or mode == "TEEva" or mode == "TERes"):
        if(len(roots) != 0):
            if(roots[0] < 1e-13):
                roots = roots[1:]
    if(lowerBound < 1e-13 and (mode == "TM" or mode == "TMEva" or mode == "TMRes")):
        if(len(roots) == 0):
            roots = np.array([1e-12])
        elif(roots[0] > 1e-12):
            roots = np.append([1e-12], roots)

    return roots

def computeAllowedKs(L, omega, wLO, wTO, epsInf, mode):
    roots = getRoots(L, omega, wLO, wTO, epsInf, mode)
    #createRootsFuncPlotWithLines(roots, L, omega, wLO, wTO, epsInf, mode, "Roots")
    #print("Number of roots found for {} mode = {}".format(mode, len(roots)))
    return roots

def findKsDerivativeW(roots, L, omega, wLO, wTO, epsInf, mode):
    delOm = omega * 1e-12
    rootsPlus = getRoots(L, omega + delOm, wLO, wTO, epsInf, mode)
    rootsPlus = rootsPlus[:len(roots)]
    return (rootsPlus - roots) / (delOm)

def createRootsFuncPlotWithLines(lines, L, omega, wLO, wTO, epsInf, mode, nameAdd):
    rootFunc = rootFuncTE
    if(mode == "TM"):
        rootFunc = rootFuncTM
    elif(mode == "TEEva"):
        rootFunc = rootFuncTEEva
    elif (mode == "TMEva"):
        rootFunc = rootFuncTMEva
    elif (mode == "TERes"):
        rootFunc = rootFuncTERes
    elif (mode == "TMRes"):
        rootFunc = rootFuncTMRes
    elif (mode == "Surf"):
        rootFunc = rootFuncSurf

    upperBound = getUpperBound(mode, omega, wLO, wTO, epsInf)
    lowerBound = getLowerBound(mode, omega, wLO, wTO, epsInf)

    kArr = np.linspace(lowerBound, upperBound, 1000)
    rootFuncVals = rootFunc(kArr, L, omega, wLO, wTO, epsInf)

    plotRootFuncs.plotRootFuncWithRoots(kArr, rootFuncVals, lines, omega, mode, nameAdd)
