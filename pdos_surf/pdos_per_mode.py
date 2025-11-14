"""

"""
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import numpy as np
import scipy.constants as consts
from multiprocessing import Pool

from . import momentum_relations as momentum_relations
from . import allowed_kz as allowed_kz

class AbstractModeFunction(ABC):
    """Template for mode resolved pdos implementation"""
    @staticmethod
    @abstractmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        pass

    # ———————————————————————————————— Pdos sum methods ———————————————————————————————— #
    @staticmethod
    @abstractmethod
    def pdos_sum_pos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        pass

    @staticmethod
    @abstractmethod
    def pdos_sum_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        pass

    @staticmethod
    def pdos_sum_para_pos(*args, **kwargs):
        raise NotImplementedError("Method pdos_sum_para_pos not implemented!")

    @staticmethod
    def pdos_sum_para_neg(*args, **kwargs):
        raise NotImplementedError("Method pdos_sum_perp_neg not implemented!")

    @staticmethod
    def pdos_sum_perp_pos(*args, **kwargs):
        raise NotImplementedError("Method pdos_sum_perp_pos not implemented!")

    @staticmethod
    def pdos_sum_perp_neg(*args, **kwargs):
        raise NotImplementedError("Method pdos_sum_perp_neg not implemented!")

    # —————————————————————————————— Full & resolved Pdos —————————————————————————————— #
    @staticmethod
    @abstractmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf) -> np.ndarray:
        pass

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Method pdos_para_perp not implemented!")
        pass

    # ————————————————————————————— Frequency space methods ———————————————————————————— #
    @staticmethod
    @abstractmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        """Implementation for which the PDOS is non zero"""
        pass

    @classmethod
    def pdos_w(cls, omega, zArr, L, wLO, wTO, epsInf) -> np.ndarray:
        if cls.pdos_non_zero_flag(omega, wLO, wTO, epsInf):
            return cls.pdos(zArr, L, omega, wLO, wTO, epsInf)
        else:
            return np.zeros(len(zArr))

    @classmethod
    def pdos_para_perp_w(cls, omega, zArr, L, wLO, wTO, epsInf) -> Tuple[np.ndarray, np.ndarray]:
        if cls.pdos_non_zero_flag(omega, wLO, wTO, epsInf):
            return cls.pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf)
        else:
            return (np.zeros(len(zArr)), np.zeros(len(zArr)))

# —————————————————————————————————— Propagating modes ————————————————————————————————— #

class PdosTE(AbstractModeFunction):
    """
    Propagating TE modes. These only have parallel components
    """
    @staticmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        epsilon = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        return epsilon > 0

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
        kArr = allowed_kz.compute_allowed_kz(L, omega, wLO, wTO, epsInf, "TE")
        kzArrDel = allowed_kz.compute_derivative_kz(kArr, L, omega, wLO, wTO, epsInf, "TE")

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPos = PdosTE.pdos_sum_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNeg = PdosTE.pdos_sum_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

        dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
        return dos

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        #! TE modes don't have a perpendicular component! Implemented for unified interface
        pdos_para = PdosTE.pdos(zArr, L, omega, wLO, wTO, epsInf)
        pdos_perp = np.zeros_like(pdos_para)
        return (pdos_para, pdos_perp)

class PdosTM(AbstractModeFunction):
    """
    Propagating TM modes. These have both parallel and perpendicular components
    """
    @staticmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        epsilon = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        return epsilon > 0

    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromK(kArr, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        brack11 = omega**2 / (consts.c**2 * kArr**2)
        brack12 = (omega**2 / (consts.c**2 * kArr**2) - 2) * np.sin(kArr * L) / (kArr * L)
        term1 = (brack11 + brack12) * np.sin(kDVal * L / 2.)**2
        brack21 = eps * omega**2 / (consts.c**2 * kDVal**2)
        brack22 = (eps * omega**2 / (consts.c**2 * kDVal**2) - 2) * np.sin(kDVal * L) / (kDVal * L)
        term2 = normPrefac * (brack21 + brack22) * np.sin(kArr * L / 2.)**2

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
    def _pdos(zArr, L, omega, wLO, wTO, epsInf) -> Tuple[np.ndarray, np.ndarray]:
        kArr = allowed_kz.compute_allowed_kz(L, omega, wLO, wTO, epsInf, "TM")
        kzArrDel = allowed_kz.compute_derivative_kz(kArr, L, omega, wLO, wTO, epsInf, "TM")

        if(len(kArr) == 0):
            return (np.zeros(len(zArr)), np.zeros(len(zArr)))

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
    def pdos(zArr, L, omega, wLO, wTO, epsInf) -> np.ndarray:
        tpl = PdosTM._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosTM._pdos(zArr, L, omega, wLO, wTO, epsInf)

# ————————————————————————— Evanescent modes (inside material) ————————————————————————— #

class PdosEvaTE(AbstractModeFunction):
    @staticmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        epsilon = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        return epsilon > 1

    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        kDVal = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        brack1 = (1 - np.exp(- 2. * kArr * L)) / (2. * kArr * L) - np.exp(- kArr * L)
        term1 = brack1 * np.sin(kDVal * L / 2) ** 2
        brack2 = 1 - np.sin(kDVal * L) / (kDVal * L)
        term2 = normPrefac * brack2 * 0.25 * (1 + np.exp(-2 * kArr * L) - 2. * np.exp(- kArr * L))
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
        kArr = allowed_kz.compute_allowed_kz(L, omega, wLO, wTO, epsInf, "TEEva")
        kzArrDel = allowed_kz.compute_derivative_kz(kArr, L, omega, wLO, wTO, epsInf, "TEEva")

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPos = PdosEvaTE.pdos_sum_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNeg = PdosEvaTE.pdos_sum_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

        dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
        return dos

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        #! TE modes don't have a perpendicular component! Implemented for unified interface
        pdos_para = PdosEvaTE.pdos(zArr, L, omega, wLO, wTO, epsInf)
        pdos_perp = np.zeros_like(pdos_para)
        return (pdos_para, pdos_perp)

class PdosEvaTM(AbstractModeFunction):
    @staticmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        epsilon = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        return epsilon > 1

    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        kDVal = momentum_relations.kDFromKEva(kArr, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

        brack11 = np.exp(- kArr * L) * omega**2 / (consts.c**2 * kArr**2)
        brack12 = (omega**2 / (consts.c**2 * kArr**2) + 2) * (1 - np.exp(- 2 * kArr * L)) / (2 * kArr * L)
        term1 = (brack11 + brack12) * np.sin(kDVal * L / 2.)**2
        brack21 = eps * omega**2 / (consts.c**2 * kDVal**2)
        brack22 = (eps * omega**2 / (consts.c**2 * kDVal**2) - 2) * np.sin(kDVal * L) / (kDVal * L)
        term2 = normPrefac * (brack21 + brack22) * 0.25 * (1 + np.exp(-2. * kArr * L) - 2. * np.exp(-kArr * L))

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
    def _pdos(zArr, L, omega, wLO, wTO, epsInf) -> Tuple[np.ndarray, np.ndarray]:
        kArr = allowed_kz.compute_allowed_kz(L, omega, wLO, wTO, epsInf, "TMEva")
        kzArrDel = allowed_kz.compute_derivative_kz(kArr, L, omega, wLO, wTO, epsInf, "TMEva")

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
    def pdos(zArr, L, omega, wLO, wTO, epsInf) -> np.ndarray:
        tpl = PdosEvaTM._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosEvaTM._pdos(zArr, L, omega, wLO, wTO, epsInf)

# —————————————————————————— Resonant modes (outside material) ————————————————————————— #

class PdosResTE(AbstractModeFunction):
    """Pdos of resonant TE modes existing outside the material"""
    @staticmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        epsilon = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        return epsilon < 1.0

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
        #kzArrDel = findAllowedKsSPhP.compute_derivative_kz(kArr, L, omega, wLO, wTO, epsInf, "TERes")
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
        kArr = allowed_kz.compute_allowed_kz(L, omega, wLO, wTO, epsInf, "TERes")
        kzArrDel = allowed_kz.compute_derivative_kz(kArr, L, omega, wLO, wTO, epsInf, "TERes")

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPos = PdosResTE.pdos_sum_pos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
        dosNeg = PdosResTE.pdos_sum_neg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

        dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
        return dos

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        #! TE modes don't have a perpendicular component! Implemented for unified interface
        pdos_para = PdosResTE.pdos(zArr, L, omega, wLO, wTO, epsInf)
        pdos_perp = np.zeros_like(pdos_para)
        return (pdos_para, pdos_perp)

class PdosResTM(AbstractModeFunction):
    """Pdos of resonant TM modes existing outside the material"""
    @staticmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        epsilon = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        return epsilon < 1.0

    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

        brack11 =  omega**2 / (consts.c**2 * kArr**2)
        brack12 = (omega**2 / (consts.c**2 * kArr**2) - 2) * np.sin(kArr * L) / (kArr * L)
        term1 = (brack11 + brack12) * 0.25 * (1 - np.exp(- 2. * kDArr * L) - 2 * np.exp(- kDArr * L))
        brack21 = np.exp(- kDArr * L) * eps * omega**2 / (consts.c**2 * kDArr**2)
        brack22 = (eps * omega**2 / (consts.c**2 * kDArr**2) + 2) * (1 - np.exp(-kDArr * L)) / (2 * kDArr * L)
        term2 = normPrefac * (brack21 + brack22) * np.sin(kArr * L)**2

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

        func = np.sqrt(omega**2 / (consts.c**2 * kArr[None, :]**2) - 1) *\
              np.cos(kArr[None, :] * (L / 2. - zArr[:, None])) * 0.5 * (1 - np.exp(-kDArr[None, :] * L))
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func ** 2 * diffFac, axis=1)

    @staticmethod
    def pdos_sum_perp_neg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
        NSqr = PdosResTM.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDArr = momentum_relations.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)

        func = np.sqrt(eps * omega**2 / (consts.c**2 * kDArr[None, :]**2) + 1) *\
              0.5 * ( np.exp(kDArr[None, :] * zArr[:, None]) +  np.exp(-kDArr[None, :] * (zArr[:, None] + L))) * np.sin(kArr[None, :] * L / 2.)
        diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
        return np.sum(1. / NSqr[None, :] * func ** 2 * diffFac, axis=1)

    @staticmethod
    def _pdos(zArr, L, omega, wLO, wTO, epsInf) -> Tuple[np.ndarray, np.ndarray]:
        kArr = allowed_kz.compute_allowed_kz(L, omega, wLO, wTO, epsInf, "TMRes")
        kzArrDel = allowed_kz.compute_derivative_kz(kArr, L, omega, wLO, wTO, epsInf, "TMRes")

        if(len(kArr) == 0):
            return (np.zeros(len(zArr)), np.zeros(len(zArr)))

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
    def pdos(zArr, L, omega, wLO, wTO, epsInf) -> np.ndarray:
        tpl = PdosResTM._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosResTM._pdos(zArr, L, omega, wLO, wTO, epsInf)

# ———————————————————————————————————— Surface modes ——————————————————————————————————— #

class PdosSurf(AbstractModeFunction):
    """Double evanescent waves localized at the interface"""
    @staticmethod
    def pdos_non_zero_flag(omega, wLO, wTO, epsInf) -> bool:
        wInf = momentum_relations.w_inf(wLO, wTO, epsInf)
        return (omega < wInf) and (omega > wTO)

    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        kDArr = momentum_relations.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        brack11 = np.exp(- kArr * L) * omega ** 2 / (consts.c ** 2 * kArr ** 2)
        brack12 = (omega ** 2 / (consts.c ** 2 * kArr ** 2) + 2) * (1 - np.exp(- 2 * kArr * L)) / (2 * kArr * L)
        term1 = (brack11 + brack12) * 0.25 * (1 + np.exp(-2. * kDArr * L) - 2. * np.exp(-kDArr * L))

        brack21 = np.exp(- kDArr * L) * eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2)
        brack22 = (eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2) + 2) * (1 - np.exp(- 2 * kDArr * L)) / (2 * kDArr * L)
        term2 = normPrefac * (brack21 + brack22) * 0.25 * (1 + np.exp(-2. * kArr * L) - 2. * np.exp(-kArr * L))

        return L / 4. * (term1 + term2)

    @staticmethod
    def pdos_sum_para_pos(zArr, kArr, L, omega, wLO, wTO, epsInf):
        NSqr = PdosSurf.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = 0.25 * (np.exp(- kArr * zArr) - np.exp(kArr * (zArr - L))) * (1 - np.exp(-kDVal * L))
        diffFac = (1. + consts.c ** 2 * kArr / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def pdos_sum_para_neg(zArr, kArr, L, omega, wLO, wTO, epsInf):
        NSqr = PdosSurf.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = 0.25 * (np.exp(kDVal * zArr) - np.exp(- kDVal * (L + zArr))) * (1 - np.exp(-kArr * L))
        diffFac = (1. + consts.c ** 2 * kArr / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def pdos_sum_perp_pos(zArr, kArr, L, omega, wLO, wTO, epsInf):
        NSqr = PdosSurf.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = 0.25 * np.sqrt(omega ** 2 / (consts.c ** 2 * kArr ** 2) + 1) *\
              (np.exp(- kArr * zArr) + np.exp(kArr * (zArr - L))) * (1 - np.exp(-kDVal * L))
        diffFac = (1. + consts.c ** 2 * kArr / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def pdos_sum_perp_neg(zArr, kArr, L, omega, wLO, wTO, epsInf):
        eps = momentum_relations.epsilon(omega, wLO, wTO, epsInf)
        NSqr = PdosSurf.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
        kDVal = momentum_relations.kDFromKSurf(kArr, omega, wLO, wTO, epsInf)
        kzDel = allowed_kz.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)

        func = -0.25 * np.sqrt(eps * omega**2 / (consts.c**2 * kDVal**2) + 1) *\
              ( np.exp(kDVal * zArr) +  np.exp(-kDVal * (L + zArr))) * (1 - np.exp(-kArr * L))
        diffFac = (1. + consts.c ** 2 * kArr / omega * kzDel)
        return 1. / NSqr * func ** 2 * diffFac

    @staticmethod
    def _pdos(zArr, L, omega, wLO, wTO, epsInf) -> Tuple[np.ndarray, np.ndarray]:
        kArr = allowed_kz.findKsSurf(L, omega, wLO, wTO, epsInf)
        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        indNeg = np.where(zArr < 0)
        indPos = np.where(zArr >= 0)
        zPosArr = zArr[indPos]
        zNegArr = zArr[indNeg]

        dosPosPara = PdosSurf.pdos_sum_para_pos(zPosArr, kArr, L, omega, wLO, wTO, epsInf)
        dosNegPara = PdosSurf.pdos_sum_para_neg(zNegArr, kArr, L, omega, wLO, wTO, epsInf)
        dosPosPerp = PdosSurf.pdos_sum_perp_pos(zPosArr, kArr, L, omega, wLO, wTO, epsInf)
        dosNegPerp = PdosSurf.pdos_sum_perp_neg(zNegArr, kArr, L, omega, wLO, wTO, epsInf)
        dosPara = np.pi * consts.c / (2. * omega) * np.append(dosNegPara, dosPosPara)
        dosPerp = np.pi * consts.c / (2. * omega) * np.append(dosNegPerp, dosPosPerp)

        return (dosPara, dosPerp)

    @staticmethod
    def pdos(zArr, L, omega, wLO, wTO, epsInf) -> np.ndarray:
        tpl = PdosSurf._pdos(zArr, L, omega, wLO, wTO, epsInf)
        return sum(tpl)

    @staticmethod
    def pdos_para_perp(zArr, L, omega, wLO, wTO, epsInf):
        return PdosSurf._pdos(zArr, L, omega, wLO, wTO, epsInf)

class PdosSurfAnalytic:
    """
    Analytic implementation of the surface modes.
    Without distinguishing components parallel and perpendicular to the surface!
    """
    @staticmethod
    def norm_sqrt(kArr, L, omega, wLO, wTO, epsInf):
        return PdosSurf.norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)

    @staticmethod
    def pdos_sum_pos(zArr, omega, wLO, wTO, epsInf):
        """pdos above the substrate surface"""
        epsAbs = np.abs(momentum_relations.epsilon(omega, wLO, wTO, epsInf))
        normPrefac = momentum_relations.normFac(omega, wLO, wTO, epsInf)

        fac1 = np.pi / np.sqrt(1 - 1. / epsAbs) ** 3
        fac2 = np.sqrt(epsAbs) / (epsAbs + normPrefac / epsAbs)
        expFac = np.exp(- 2. * omega * zArr / (consts.c * np.sqrt(epsAbs - 1)))
        diffExtraFac = 1 + 1. / epsInf * omega**2 / (1 - 1. / epsAbs) * (wLO**2 - wTO**2) / (wLO**2 - omega**2)**2
        return fac1 * fac2 * expFac * diffExtraFac

    @staticmethod
    def pdos_sum_neg(zArr, omega, wLO, wTO, epsInf):
        """pdos below the substrate surface"""
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
        return (fac1 * fac2 * expFac * diffExtraFac).T

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

def pdos_para_perp_array(cls, wArr, zArr, L, wLO, wTO, epsInf, n_processes=None):
    """
    Calculate pdos for a given mode for given class cls, for frequency array.
    Results are split into parallel and perpendicular component
    """
    if n_processes == 1:
        # Single-process version
        results = [cls.pdos_para_perp_w(w, zArr=zArr, L=L, wLO=wLO, wTO=wTO, epsInf=epsInf)
                   for w in wArr]
    else:
        # Multi-process version with spawn-safe context
        worker = partial(cls.pdos_para_perp_w, zArr=zArr, L=L, wLO=wLO, wTO=wTO, epsInf=epsInf)
        with Pool(processes=n_processes) as pool:
            results = pool.map(worker, wArr)

    # worker = partial(cls.pdos_para_perp_w, zArr=zArr, L=L, wLO=wLO, wTO=wTO, epsInf=epsInf)
    # with Pool(processes=n_processes) as pool:
    #     results = pool.map(worker, wArr)

    pdos = np.empty((2, len(zArr), len(wArr)))
    for i, (para, perp) in enumerate(results):
        pdos[0, :, i] = para
        pdos[1, :, i] = perp

    return pdos

def pdos_para_perp_multi(clsArr, omega, zArr, L, wLO, wTO, epsInf):
    """
    Compute PDOS for multiple modes at a single frequency. First entry must be the one we
    iterate over.

    Returns: (n_modes, 2, len(zArr))
    """
    n_modes = len(clsArr)
    results = np.empty((n_modes, 2, len(zArr)))

    for i, cls in enumerate(clsArr):
        para, perp = cls.pdos_para_perp_w(omega, zArr, L, wLO, wTO, epsInf)
        results[i, 0, :] = para
        results[i, 1, :] = perp

    return results

def pdos_para_perp_array_multi(cls_arr, wArr, zArr, L, wLO, wTO, epsInf, n_processes=None):
    """
    Calculate pdos for multiple modes, for frequency array.
    Results are split into parallel and perpendicular component.

    Returns: (n_modes, 2, len(zArr), len(wArr))
    """
    args_list = [(cls_arr, omega, zArr, L, wLO, wTO, epsInf) for omega in wArr]

    with Pool(processes=n_processes) as pool:
        results = pool.starmap(pdos_para_perp_multi, args_list)

    # transpose: (len(wArr), n_modes, 2, len(zArr)) -> (n_modes, 2, len(zArr), len(wArr))
    all_results = np.array(results)
    pdos_array = np.transpose(all_results, (1, 2, 3, 0))

    return pdos_array
