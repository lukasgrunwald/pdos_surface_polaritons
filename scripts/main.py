##
import sys
import numpy as np
import scipy.constants as consts

from pdos_surf.frequency_calculation import produceFreqDataV2 as produceFreqDataV2

def main():
    print("Compute full Dos and all modes")

    #! This is something run specific!!
    wSubArrInd = 0
    wArrSubdivisions = 100 #! Using a temporary upper frequency

    #numbers for sto
    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    epsInf = 1.
    L = 1.

    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    # zArr = np.logspace(np.log10(1e1 * lambda0), np.log10(1e-3 * lambda0), 200, endpoint=True, base = 10)
    zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), 200, endpoint=True, base = 10)
    # zArr = np.append([L / 4.], zArr)
    # indArr = np.array([1, 30, 60, 92], dtype=int)
    # zArr = zArr[indArr]

    ###new version of freq-int handling
    # for wSubArrIndTemp in tqdm(np.arange(wArrSubdivisions)):
        # produceFreqDataV2.produceFreqData(wSubArrIndTemp, wArrSubdivisions, zArr, wLO, wTO, epsInf, L)
    produceFreqDataV2.produceFreqData(wSubArrInd, wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

if __name__ == "__main__":
    main()
