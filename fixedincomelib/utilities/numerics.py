import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Monotone, single-curvature-per-interval IFR interpolator.

    The supplied nodes are *point samples* of the instantaneous forward curve
    (not discrete/average forwards), so each node value must be reproduced
    exactly: this rules out Hagan-West's original three-constraint quadratic
    (value, value, bucket-average), since there is no separate average quote
    to match. Instead, on each segment [t_k, t_{k+1}] we fit the quadratic

        q(x) = v_k + b*x + c*x^2,   x = (t - t_k) / (t_{k+1} - t_k) in [0, 1]

    through the two exact node values, with the third degree of freedom
    fixed by a Hagan-West/PCHIP-style shape-preserving derivative estimate
    d_k at the left node (a weighted harmonic mean of the two neighbouring
    secant slopes, zero at a local extremum, one-sided at the ends). d_k is
    then clamped so that both q'(0) and q'(1) keep the sign of the segment's
    own secant S_k = (v_{k+1} - v_k) / (t_{k+1} - t_k):

        S_k > 0  =>  0 <= d_k <= 2*S_k
        S_k < 0  =>  2*S_k <= d_k <= 0
        S_k == 0 =>  d_k = 0

    Since q' is linear in x, same-signed endpoint slopes imply q' never
    changes sign on [0, 1] - the segment is monotone. Being quadratic, q''
    is constant on the segment, so curvature never changes sign either -
    the segment is convex or concave throughout, with no interior
    inflection. The curve is continuous (adjacent segments share the exact
    node value) but, as in Hagan-West's own construction, the first
    derivative may kink at a node. Outside [axis1[0], axis1[-1]] the curve
    is extended flat at the boundary node value, matching the FLAT
    extrapolation used elsewhere in this library. A single-node curve is
    treated as a constant.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

        t = np.asarray(self.axis1_, dtype=float)
        v = np.asarray(self.values_, dtype=float)
        n = self.length_

        if n < 2:
            self.h_ = np.array([])
            self.b_ = np.array([])
            self.c_ = np.array([])
            return

        h = np.diff(t)
        s = np.diff(v) / h
        d = np.empty(n - 1)

        for k in range(n - 1):
            if k == 0:
                if n == 2:
                    d_raw = s[0]
                else:
                    h0, h1 = h[0], h[1]
                    s0, s1 = s[0], s[1]
                    d_raw = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1)
            else:
                s_left, s_right = s[k - 1], s[k]
                if s_left * s_right <= 0.0:
                    d_raw = 0.0
                else:
                    h_left, h_right = h[k - 1], h[k]
                    w1 = 2.0 * h_right + h_left
                    w2 = h_right + 2.0 * h_left
                    d_raw = (w1 + w2) / (w1 / s_left + w2 / s_right)

            s_k = s[k]
            if s_k > 0.0:
                d[k] = min(max(d_raw, 0.0), 2.0 * s_k)
            elif s_k < 0.0:
                d[k] = max(min(d_raw, 0.0), 2.0 * s_k)
            else:
                d[k] = 0.0

        self.h_ = h
        self.b_ = d * h
        self.c_ = (s - d) * h

    def _segment(self, x: float) -> int:
        t = self.axis1_
        idx = int(np.searchsorted(t, x, side='right')) - 1
        return min(max(idx, 0), self.length_ - 2)

    def interpolate(self, x: float) -> float:
        v = self.values_
        if self.length_ == 1:
            return float(v[0])
        t = self.axis1_
        if x <= t[0]:
            return float(v[0])
        if x >= t[-1]:
            return float(v[-1])
        k = self._segment(x)
        xi = (x - t[k]) / self.h_[k]
        return float(v[k] + self.b_[k] * xi + self.c_[k] * xi * xi)

    def integrate(self, start_x: float, end_x: float) -> float:
        v = self.values_
        if self.length_ == 1:
            return float(v[0] * (end_x - start_x))

        t = self.axis1_
        total = 0.0

        lo, hi = start_x, min(end_x, t[0])
        if hi > lo:
            total += v[0] * (hi - lo)

        for k in range(self.length_ - 1):
            lo, hi = max(start_x, t[k]), min(end_x, t[k + 1])
            if hi > lo:
                h_k = self.h_[k]
                x_lo = (lo - t[k]) / h_k
                x_hi = (hi - t[k]) / h_k
                total += h_k * (
                    v[k] * (x_hi - x_lo)
                    + self.b_[k] / 2.0 * (x_hi ** 2 - x_lo ** 2)
                    + self.c_[k] / 3.0 * (x_hi ** 3 - x_lo ** 3)
                )

        lo, hi = max(start_x, t[-1]), end_x
        if hi > lo:
            total += v[-1] * (hi - lo)

        return float(total)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
