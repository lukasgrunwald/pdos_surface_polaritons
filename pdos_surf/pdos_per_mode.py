"""

"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy.constants as consts
from time import perf_counter

from . import momentum_relations as momentum_relations
from . import allowed_kz as allowed_kz

class AbstractModeFunction(ABC):
    @staticmethod
    @abstractmethod
    def norm_sqrt(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def pdos_sum_pos(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def pdos_sum_neg(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def pdos(*args, **kwargs) -> np.ndarray:
        pass

# —————————————————————————————————— Propagating modes ————————————————————————————————— #

class PdosTE(AbstractModeFunction):
    """
    Propagating TE modes. These only have parallel components
    """
    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        kDVal = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)
        term1 = (1 - np.sin(kArr * L) / (kArr * L)) * np.sin(kDVal * L / 2) ** 2
        term2 = normPrefac * (1 - np.sin(kDVal * L) / (kDVal * L)) * np.sin(kArr * L / 2) ** 2
        return L / 4 * (term1 + term2)

    @staticmethod
    def pdos_sum_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosTE.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)
        func = np.sin(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosTE.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)
        func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf):
        kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TE")
        kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TE")

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPos = PdosTE.pdos_sum_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNeg = PdosTE.pdos_sum_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

        dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
        return dos

class PdosTM(AbstractModeFunction):
    """
    Propagating TM modes. These have both parallel and perpendicular components
    """
    @staticmethod
    def norm_sqrt(kVal, L, omega, wLO, wTO, epsInf):
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromK(kVal, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        brack11 = omega**2 / (consts.c**2 * kVal**2)
        brack12 = (omega**2 / (consts.c**2 * kVal**2) - 2) * np.sin(kVal * L) / (kVal * L)
        term1 = (brack11 + brack12) * np.sin(kDVal * L / 2.)**2
        brack21 = eps * omega**2 / (consts.c**2 * kDVal**2)
        brack22 = (eps * omega**2 / (consts.c**2 * kDVal**2) - 2) * np.sin(kDVal * L) / (kDVal * L)
        term2 = normPrefac * (brack21 + brack22) * np.sin(kVal * L / 2.)**2

        return L / 4. * (term1 + term2)

    @staticmethod
    def pdos_sum_para_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)

        func = np.sin(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_para_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)

        func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_perp_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)

        func = np.sqrt(omega ** 2 / (consts.c ** 2 * kArr[None, :] ** 2) - 1) *\
              np.cos(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_perp_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        NSqr = PdosTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)

        func = - np.sqrt(eps * omega**2 / (consts.c**2 * kDArr[None, :]**2) - 1) *\
              np.cos(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def _pdos(zArr, L, omega, wLO, wTO, epsInf):
        kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TM")
        kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")

        if(len(kArr) == 0):
            return 0

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPosPara = PdosTM.pdos_sum_para_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNegPara = PdosTM.pdos_sum_para_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosPosPerp = PdosTM.pdos_sum_perp_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNegPerp = PdosTM.pdos_sum_perp_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

        dosPara = np.pi * consts.c / (2. * omega) * np.append(dosNegPara, dosPosPara)
        dosPerp = np.pi * consts.c / (2. * omega) * np.append(dosNegPerp, dosPosPerp)
        return (dosPara, dosPerp)

    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf):
        tpl = PdosTM._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosTM._pdos(zArr, L, omega, wLO, wTO, epsInf)

# ————————————————————————— Evanescent modes (inside material) ————————————————————————— #

class PdosEvaTE(AbstractModeFunction):
    @staticmethod
    def norm_sqrt(kVal, L, omega, wLO, wTO, epsInf):
        kDVal = momentum_relations.kDFromKEva(kVal, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        brack1 = (1 - np.exp(- 2. * kVal * L)) / (2. * kVal * L) - np.exp(- kVal * L)
        term1 = brack1 * np.sin(kDVal * L / 2) ** 2
        brack2 = 1 - np.sin(kDVal * L) / (kDVal * L)
        term2 = normPrefac * brack2 * 0.25 * (1 + np.exp(-2 * kVal * L) - 2. * np.exp(- kVal * L))
        return L / 4 * (term1 + term2)

    @staticmethod
    def pdos_sum_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosEvaTE.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)

        func = 0.5 * (np.exp(- kArr[None, :] * zArr[:, None]) - np.exp(kArr * zArr[:, None] - kArr[None, :] * L)) * np.sin(kDArr[None, :] * L / 2.)
        diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosEvaTE.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)

        func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * 0.5 * (1 - np.exp(- kArr[None, :] * L))
        diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf):
        kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TEEva")
        kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TEEva")

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPos = PdosEvaTE.pdos_sum_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNeg = PdosEvaTE.pdos_sum_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

        dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
        return dos

class PdosEvaTM(AbstractModeFunction):
    @staticmethod
    def norm_sqrt(kVal, L, omega, wLO, wTO, epsInf):
        kDVal = momentum_relations.kDFromKEva(kVal, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

        brack11 = np.exp(- kVal * L) * omega**2 / (consts.c**2 * kVal**2)
        brack12 = (omega**2 / (consts.c**2 * kVal**2) + 2) * (1 - np.exp(- 2 * kVal * L)) / (2 * kVal * L)
        term1 = (brack11 + brack12) * np.sin(kDVal * L / 2.)**2
        brack21 = eps * omega**2 / (consts.c**2 * kDVal**2)
        brack22 = (eps * omega**2 / (consts.c**2 * kDVal**2) - 2) * np.sin(kDVal * L) / (kDVal * L)
        term2 = normPrefac * (brack21 + brack22) * 0.25 * (1 + np.exp(-2. * kVal * L) - 2. * np.exp(-kVal * L))

        return L / 4. * (term1 + term2)

    @staticmethod
    def pdos_sum_para_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosEvaTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)

        func = 0.5 * (np.exp(- kArr[None, :] * zArr[:, None]) -\
                       np.exp(kArr[None, :] * (zArr[:, None] - L))) * np.sin(kDArr[None, :] * L / 2.)
        diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_para_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosEvaTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)

        func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * 0.5 * (1 - np.exp(-kArr[None, :] * L))
        diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_perp_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosEvaTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)

        func = np.sqrt(omega ** 2 / (consts.c ** 2 * kArr[None, :] ** 2) + 1) * 0.5 *\
              (np.exp(- kArr[None, :] * zArr[:, None]) +\
                np.exp(kArr[None, :] * (zArr[:, None] - L))) * np.sin(kDArr[None, :] * L / 2.)
        diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_perp_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosEvaTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

        func = np.sqrt(eps * omega**2 / (consts.c**2 * kDArr[None, :]**2) - 1) *\
          np.cos(kDArr[None, :] * (L / 2. + zArr[:, None])) * 0.5 * (1 - np.exp(-kArr[None, :] * L))
        diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def _pdos(zArr, L, omega, wLO, wTO, epsInf):
        kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TMEva")
        kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TMEva")

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPosPara = PdosEvaTM.pdos_sum_para_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNegPara = PdosEvaTM.pdos_sum_para_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosPosPerp = PdosEvaTM.pdos_sum_perp_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNegPerp = PdosEvaTM.pdos_sum_perp_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosPara = np.pi * consts.c / (2. * omega) * np.append(dosNegPara, dosPosPara)
        dosPerp = np.pi * consts.c / (2. * omega) * np.append(dosNegPerp, dosPosPerp)
        return (dosPara, dosPerp)


    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf):
        tpl = PdosEvaTM._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosEvaTM._pdos(zArr, L, omega, wLO, wTO, epsInf)

# —————————————————————————— Resonant modes (outside material) ————————————————————————— #

class PdosResTE(AbstractModeFunction):
    """Pdos of resonant TE modes existing outside the material"""
    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        brack1 = 1 - np.sin(kArr * L) / (kArr * L)
        term1 = brack1 * 0.25 * (1 + np.exp(-2 * kDArr * L) - 2. * np.exp(- kDArr * L))
        brack2 = (1 - np.exp(- 2. * kDArr * L)) / (2. * kDArr * L) - np.exp(- kDArr * L)
        term2 = normPrefac * brack2 * np.sin(kArr * L / 2) ** 2
        return L / 4 * (term1 + term2)

    @staticmethod
    def pdos_sum_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosResTE.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
        func = np.sin(kArr[None, :] * (L / 2. - zArr[:, None])) * 0.5 * (1 - np.exp(- kDArr[None, :] * L))
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos_sum_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosResTE.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)

        func = 0.5 * (np.exp(kDArr[None, :] * zArr[:, None]) -\
                       np.exp(-kDArr[None, :] * zArr[:, None] - kDArr[None, :] * L)) * np.sin(kArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf):
        kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TERes")
        kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPos = PdosResTE.pdos_sum_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNeg = PdosResTE.pdos_sum_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

        dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
        return dos

class PdosResTM(AbstractModeFunction):
    """Pdos of resonant TM modes existing outside the material"""
    @staticmethod
    def norm_sqrt(kVal, L, omega, wLO, wTO, epsInf):
        kDArr = momentum_relations.kDFromKRes(kVal, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

        brack11 =  omega**2 / (consts.c**2 * kVal**2)
        brack12 = (omega**2 / (consts.c**2 * kVal**2) - 2) * np.sin(kVal * L) / (kVal * L)
        term1 = (brack11 + brack12) * 0.25 * (1 - np.exp(- 2. * kDArr * L) - 2 * np.exp(- kDArr * L))
        brack21 = np.exp(- kDArr * L) * eps * omega**2 / (consts.c**2 * kDArr**2)
        brack22 = (eps * omega**2 / (consts.c**2 * kDArr**2) + 2) * (1 - np.exp(-kDArr * L)) / (2 * kDArr * L)
        term2 = normPrefac * (brack21 + brack22) * np.sin(kVal * L)**2

        return L / 4. * (term1 + term2)

    @staticmethod
    def pdos_sum_para_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosResTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)

        func = np.sin(kArr[None, :] * (L / 2. - zArr[:, None])) * 0.5 * (1 - np.exp(-kDArr[None, :] * L))
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func ** 2 * diffFac, axis=1)

    @staticmethod
    def pdos_sum_para_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosResTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)

        func = 0.5 * (np.exp(kDArr[None, :] * zArr[:, None]) -\
                       np.exp(-kDArr[None, :] * (zArr[:, None] + L))) * np.sin(kArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func ** 2 * diffFac, axis=1)

    @staticmethod
    def pdos_sum_perp_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosResTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)

        func = np.sqrt(omega**2 / (consts.c**2 * kArr[None, :]**2) - 1) * np.cos(kArr[None, :] * (L / 2. - zArr[:, None])) * 0.5 * (1 - np.exp(-kDArr[None, :] * L))
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func ** 2 * diffFac, axis=1)

    @staticmethod
    def pdos_sum_perp_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosResTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

        func = np.sqrt(eps * omega**2 / (consts.c**2 * kDArr[None, :]**2) + 1) * 0.5 * ( np.exp(kDArr[None, :] * zArr[:, None]) +  np.exp(-kDArr[None, :] * (zArr[:, None] + L))) * np.sin(kArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func ** 2 * diffFac, axis=1)

    @staticmethod
    def _pdos(zArr, L, omega, wLO, wTO, epsInf) -> Tuple[np.ndarray, np.ndarray]:
        kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TMRes")
        kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TMRes")

        if(len(kArr) == 0):
            return 0

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPosPara = PdosResTM.pdos_sum_para_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNegPara = PdosResTM.pdos_sum_para_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosPosPerp = PdosResTM.pdos_sum_perp_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNegPerp = PdosResTM.pdos_sum_perp_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosPara = np.pi * consts.c / (2. * omega) * np.append(dosNegPara, dosPosPara)
        dosPerp = np.pi * consts.c / (2. * omega) * np.append(dosNegPerp, dosPosPerp)

        return (dosPara, dosPerp)

    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf):
        tpl = PdosResTM._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosResTM._pdos(zArr, L, omega, wLO, wTO, epsInf)

# ———————————————————————————————————— Surface modes ——————————————————————————————————— #

class PdosSurf(AbstractModeFunction):
    """Double evanescent waves localized at the interface"""
    @staticmethod
    def norm_sqrt(kVal, L, omega, wLO, wTO, epsInf):
        kDArr = momentum_relations.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        brack11 = np.exp(- kVal * L) * omega ** 2 / (consts.c ** 2 * kVal ** 2)
        brack12 = (omega ** 2 / (consts.c ** 2 * kVal ** 2) + 2) * (1 - np.exp(- 2 * kVal * L)) / (2 * kVal * L)
        term1 = (brack11 + brack12) * 0.25 * (1 + np.exp(-2. * kDArr * L) - 2. * np.exp(-kDArr * L))

        brack21 = np.exp(- kDArr * L) * eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2)
        brack22 = (eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2) + 2) * (1 - np.exp(- 2 * kDArr * L)) / (2 * kDArr * L)
        term2 = normPrefac * (brack21 + brack22) * 0.25 * (1 + np.exp(-2. * kVal * L) - 2. * np.exp(-kVal * L))

        return L / 4. * (term1 + term2)

    @staticmethod
    def pdos_sum_para_pos(zArr, kVal, L, omega, wLO, wTO, epsInf):
        NSqr = PdosSurf.norm_sqrt(kVal, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = 0.25 * (np.exp(- kVal * zArr) - np.exp(kVal * (zArr - L))) * (1 - np.exp(-kDVal * L))
        diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def pdos_sum_para_neg(zArr, kVal, L, omega, wLO, wTO, epsInf):
        NSqr = PdosSurf.norm_sqrt(kVal, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = 0.25 * (np.exp(kDVal * zArr) - np.exp(- kDVal * (L + zArr))) * (1 - np.exp(-kVal * L))
        diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def pdos_sum_perp_pos(zArr, kVal, L, omega, wLO, wTO, epsInf):
        NSqr = PdosSurf.norm_sqrt(kVal, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = 0.25 * np.sqrt(omega ** 2 / (consts.c ** 2 * kVal ** 2) + 1) * (np.exp(- kVal * zArr) + np.exp(kVal * (zArr - L))) * (1 - np.exp(-kDVal * L))
        diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def pdos_sum_perp_neg(zArr, kVal, L, omega, wLO, wTO, epsInf):
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        NSqr = PdosSurf.norm_sqrt(kVal, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = -0.25 * np.sqrt(eps * omega**2 / (consts.c**2 * kDVal**2) + 1) * ( np.exp(kDVal * zArr) +  np.exp(-kDVal * (L + zArr))) * (1 - np.exp(-kVal * L))
        diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def _pdos(zArr, L, omega, wLO, wTO, epsInf):
        kVal = allowed_kz.findKsSurf(L, omega, wLO, wTO, epsInf)
        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPosPara = PdosSurf.pdos_sum_para_pos(zPosArr, kVal, L, omega, wLO, wTO, epsInf)
        dosNegPara = PdosSurf.pdos_sum_para_neg(zNegArr, kVal, L, omega, wLO, wTO, epsInf)
        dosPosPerp = PdosSurf.pdos_sum_perp_pos(zPosArr, kVal, L, omega, wLO, wTO, epsInf)
        dosNegPerp = PdosSurf.pdos_sum_perp_neg(zNegArr, kVal, L, omega, wLO, wTO, epsInf)
        dosPara = np.pi * consts.c / (2. * omega) * np.append(dosNegPara, dosPosPara)
        dosPerp = np.pi * consts.c / (2. * omega) * np.append(dosNegPerp, dosPosPerp)

        return (dosPara, dosPerp)

    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf):
        tpl = PdosSurf._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosSurf._pdos(zArr, L, omega, wLO, wTO, epsInf)

class PdosSurfAnalytic(AbstractModeFunction):
    """
    Analytic implementation of the surface modes.
    Without distinguishing components parallel and perpendicular to the surface!
    """
    @staticmethod
    def norm_sqrt(kVal, L, omega, wLO, wTO, epsInf):
        return PdosSurf.norm_sqrt(kVal, L, omega, wLO, wTO, epsInf)

    @staticmethod
    def pdos_sum_pos(zArr, omega, wLO, wTO, epsInf):
        epsAbs = np.abs(momentum_relations.epsilon(omega, wLO, wTO, epsInf))
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        fac1 = np.pi / np.sqrt(1 - 1. / epsAbs) ** 3
        fac2 = np.sqrt(epsAbs) / (epsAbs + normPrefac / epsAbs)
        expFac = np.exp(- 2. * omega * zArr / (consts.c * np.sqrt(epsAbs - 1)))
        diffExtraFac = 1 + 1. / epsInf * omega**2 / (1 - 1. / epsAbs) * (wLO**2 - wTO**2) / (wLO**2 - omega**2)**2
        return fac1 * fac2 * expFac * diffExtraFac

    @staticmethod
    def pdos_sum_neg(zArr, omega, wLO, wTO, epsInf):
        epsAbs = np.abs(momentum_relations.epsilon(omega, wLO, wTO, epsInf))
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        fac1 = np.pi / np.sqrt(1 - 1. / epsAbs) ** 3
        fac2 = 1. / np.sqrt(epsAbs) / (epsAbs + normPrefac / epsAbs)
        expFac = np.exp(2. * omega * zArr * epsAbs / (consts.c * np.sqrt(epsAbs - 1)))
        diffExtraFac = 1 + 1. / epsInf * omega**2 / (1 - 1. / epsAbs) * (wLO**2 - wTO**2) / (wLO**2 - omega**2)**2
        return fac1 * fac2 * expFac * diffExtraFac

    @staticmethod
    def pdos(zArr, omega, wLO, wTO, epsInf):
        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPos = PdosSurfAnalytic.pdos_sum_pos(zPosArr, omega, wLO, wTO, epsInf)
        dosNeg = PdosSurfAnalytic.pdos_sum_neg(zNegArr, omega, wLO, wTO, epsInf)

        dos = np.append(dosNeg, dosPos)
        return dos

    @staticmethod
    def pdos_frequency_pos(zArr, omegaArr, wLO, wTO, epsInf):
        epsAbs = np.abs(momentum_relations.epsilon(omegaArr, wLO, wTO, epsInf))
        normPrefac = momentum_relations.normFac(omegaArr, wLO, wTO, epsInf)

        fac1 = np.pi / np.sqrt(1 - 1. / epsAbs[:, None]) ** 3
        fac2 = np.sqrt(epsAbs[:, None]) / (epsAbs[:, None] + normPrefac[:, None] / epsAbs[:, None])
        expFac = np.exp(- 2. * omegaArr[:, None] * zArr[None, :] / (consts.c * np.sqrt(epsAbs[:, None] - 1)))
        diffExtraFac = 1 + 1. / epsInf * omegaArr[:, None] ** 2 / (1 - 1. / epsAbs[:, None]) * (wLO ** 2 - wTO ** 2) / (
                wLO ** 2 - omegaArr[:, None] ** 2) ** 2
        return fac1 * fac2 * expFac * diffExtraFac

    @staticmethod
    def pdos_int(omega, zVal, wLO, wTO, epsInf):
        """Implementation for subsequent integration"""
        epsAbs = np.abs(momentum_relations.epsilon(omega, wLO, wTO, epsInf))
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        fac1 = np.pi / np.sqrt(1 - 1. / epsAbs) ** 3
        fac2 = np.sqrt(epsAbs) / (epsAbs + normPrefac / epsAbs)
        expFac = np.exp(- 2. * omega * zVal / (consts.c * np.sqrt(epsAbs - 1)))
        diffExtraFac = 1 + 1. / epsInf * omega**2 / (1 - 1. / epsAbs) * (wLO**2 - wTO**2) / (wLO**2 - omega**2)**2
        return fac1 * fac2 * expFac * diffExtraFac

#! We should still implement the checks for when a given solution can exist!
