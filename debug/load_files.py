import numpy as np

def load_dos_resolved(filepath):
    data = np.load(filepath)

    dosTE = data['dosTETotal']  # shape (3, n_freq, n_z)
    dosTM = data['dosTMPara']    # shape (3, n_freq, n_z)
    wArr = data['wArr']
    zArr = data['zArr']
    wLO = data['wLO'].item()
    wTO = data['wTO'].item()
    epsInf = data['epsInf'].item()

    return wArr, zArr, dosTE, dosTM, wLO, wTO, epsInf
