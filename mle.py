import logging
import pytest
from math import *
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def L(mu: float, sigma: float, x: [float, List[float]]) -> float:
    def L_(mu, sigma, x: float):
        L__ret = (1/(sqrt(2*pi*(sigma**2))))*exp(-((x-mu)**2)/(2*(sigma**2)))
        logger.debug(f"mu: {mu}, sigma: {sigma}, x: {x} -> L_: {L__ret}")
        return L__ret
    if isinstance(x, float):
        x = [x]
    L_ret = 1
    for datapoint in x:
        L_ret *= L_(mu, sigma, datapoint)
    return L_ret

def test_L():
    # https://www.youtube.com/watch?v=Dn6b9fCIUpM
    assert L(mu=28.0, sigma=2.0, x=32.0) == pytest.approx(0.03, rel=1)
    assert L(mu=30.0, sigma=2.0, x=32.0) == pytest.approx(0.12, rel=1)
    assert L(mu=28.0, sigma=2.0, x=[32.0,34.0]) == pytest.approx(6e-5, rel=1)  # 6e-6 in video..?

def L_sigma(mu: float, sigmas: List[float], x: List[float]):
    from collections.abc import Iterable
    if not isinstance(x, Iterable) or len(x) < 2 :
        raise TypeError(f"You cannot calculate the likelyhood of a std dev with one data point. Input: {x}")
    L_sigma_ret = []
    for sigma in sigmas:
        L_sigma_ret.append(L(mu, sigma, x))
    return L_sigma_ret

def test_L_sigma():
    datapoints = [1,2,3,4]
    sigmas = np.linspace(1,2)
    L_sigmas = L_sigma(2.5, sigmas, datapoints)
    assert sigmas[L_sigmas.index(max(L_sigmas))] == pytest.approx(pd.Series(datapoints).std(), rel=1)
    with pytest.raises(TypeError):
        L_sigma(mu=1, sigmas=[1], x=[1])

def L_mu(mus: List[float], sigma: float, x):
    L_mu_ret = []
    for mu in mus:
        L_mu_ret.append(L(mu, sigma, x))
    return L_mu_ret

def test_L_mu():
    mus = [1,2,2.5,3,4]
    L_mus = L_mu(mus, 0.5, [1,2,3,4])
    assert mus[L_mus.index(max(L_mus))] == 2.5

def main():
    datapoints = [1,2,3,4]
    default_mu = 1.5
    default_sigma = 1
    scan_mu = [1, 2, 2.5, 3, 4]
    scan_sigma = np.linspace(0.1,4)# [0.5, 1, 1.5, 2]
    # Results
    lmu = L_mu(scan_mu, default_sigma, x=datapoints)
    lsig = L_sigma(default_mu, scan_sigma, x=datapoints)
    lmu_pd = pd.Series(lmu)
    lsig_pd = pd.Series(lsig)
    # lmu_pd.plot()
    # lsig_pd.plot()
    # plt.show()
    logger.info(f"Likelyhood scam mus. Optimal mu    {scan_mu[lmu_pd.idxmax()]}")
    logger.info(f"Likelyhood scan sig: Optimal sigma {scan_sigma[lsig_pd.idxmax()]}")

if __name__ == "__main__":
    main()