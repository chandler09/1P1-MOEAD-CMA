import math
import numpy as np
from numpy.random import standard_normal
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.reference_direction import get_partition_closest_to_points
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
import time
import warnings


warnings.filterwarnings('ignore')
alpha = 1e-6  # penalty coefficient for the box constraints


class ECMA:
    """
    This Class is mostly inherited from the implementation by CyberAgentAILab: https://github.com/CyberAgentAILab/cmaes
    """
    def __init__(self, xl, xu, x=None, f=np.full(3, np.inf, float), g=np.inf):
        self._xl, self._xu = xl, xu
        self._dim = len(xl)
        self.mean = np.random.uniform(xl, xu, self._dim) if x is None else x.copy()
        self._sgm = .5 * np.mean(xu - xl)

        self._C = np.eye(self._dim)
        self._A = np.eye(self._dim)
        self._pc = np.zeros(self._dim)

        # Selection :
        self._lambda = 1

        # Step size control :
        self._d = 1. + self._dim / (2. * self._lambda)
        self._ptarg = 1. / (5. + np.sqrt(self._lambda) / 2.)
        self._cp = self._ptarg * self._lambda / (2. + self._ptarg * self._lambda)

        # Covariance matrix adaptation
        self._cc = 2. / (self._dim + 2.)
        self._ccov = 2. / (self._dim ** 2 + 6.)
        self._pthresh = .44

        self._psucc = self._ptarg
        self.f = f.copy()
        self.g = g
        self._t = 0
        self._moment = np.zeros(self._dim, float)

    def sample(self):
        for _ in range(100):
            x = self.mean + self._sgm * self._A.dot(standard_normal(self._dim))
            if not np.any(np.logical_or(x < self._xl, x > self._xu)):
                break
        return x

    def update(self, x, x_repaired, f, g0, g):
        suc = 1. if g0 < self.g else 0.
        self._psucc = (1. - self._cp) * self._psucc + self._cp * float(suc) / self._lambda

        if suc > 0.:
            x_step = (x - self.mean) / self._sgm
            self.mean = x_repaired
            self.f = f.copy()
            self.g = g
            if self._psucc < self._pthresh:
                self._pc = (1. - self._cc) * self._pc + np.sqrt(self._cc * (2. - self._cc)) * x_step
                self._C = (1. - self._ccov) * self._C + self._ccov * np.outer(self._pc, self._pc)
            else:
                self._pc = (1. - self._cc) * self._pc
                self._C = (1. - self._ccov) * self._C + self._ccov * (
                            np.outer(self._pc, self._pc) + self._cc * (2. - self._cc) * self._C)

        self._sgm = self._sgm * np.exp(1. / self._d * (self._psucc - self._ptarg) / (1. - self._ptarg))
        self._sgm = min(self._sgm, 1e32)
        try:
            self._A = np.linalg.cholesky(self._C)
        except np.linalg.LinAlgError:
            self._C, self._A, self._sgm = np.eye(self._dim), np.eye(self._dim), .5 * np.mean(self._xu - self._xl)
        self._t += 1

    def inject(self, x, f, g):
        if g >= self.g:
            return
        self.mean, self.f, self.g = x.copy(), f.copy(), g


class MOEAD_CMA_1P1:
    def __init__(self, problem, N, Z, decomp='PBI', theta=5., seed=0):
        """
        :param problem: test problem under the Problem class in PYMOO
        :param N: population size
        :param Z: utopian point, initialize it with a nadir point which also serves as the reference point of HV
        :param decomp: aggregation function, can either be 'TCH' or 'PBI'
        :param theta: the parameter in PBI
        :param seed: random seed
        """
        np.random.seed(seed)
        self._p = problem
        self._z = np.array(Z)
        self._M, self._D = problem.n_obj, problem.n_var
        self._xl, self._xu = problem.xl, problem.xu

        self._evals = 0
        self._gen = 0
        self._hv = HV(ref_point=np.array(Z) + .1)
        self._igd = None
        if hasattr(problem, 'pareto_front'):
            pf = problem.pareto_front(500 if self._M == 2 else 990)
            self._igd = IGD(pf) if pf is not None else IGD(np.array([self._p.zl]))

        n_partitions = get_partition_closest_to_points(N, problem.n_obj)
        self._lambda = get_reference_directions("uniform", n_dim=problem.n_obj, n_partitions=n_partitions)
        self._lambda = self._lambda[np.argsort(-np.var(self._lambda, axis=1))]  # preference
        dist = euclidean_distances(self._lambda, self._lambda)
        self._B = np.argsort(dist, axis=1)[:, :4 + math.floor(3 * math.log(self._D))]  # neighborhood
        if decomp == 'TCH':
            self.decompose = ASF()  # Tchebicheff()
        elif decomp == 'PBI':
            self.decompose = PBI(theta=theta)
        self._N = len(self._lambda)  # population size
        self._cma = [ECMA(self._xl, self._xu, f=np.full(problem.n_obj, np.inf, float)) for _ in range(self._N)]
        pop = np.vstack([self._cma[i].mean for i in range(self._N)])
        f = self._evaluate(pop)
        g = self.decompose.do(f, self._lambda, utopian_point=self._z, _type="one_to_one").flatten()
        for i in range(self._N):
            self._cma[i].f = f[i]
            self._cma[i].g = g[i]

    def _evaluate(self, pop):
        self._evals += len(pop)
        f = self._p.evaluate(pop)
        self._z = np.minimum(np.min(f, axis=0), self._z)
        return f

    def _evolve(self):
        pop0 = np.vstack([self._cma[i].sample() for i in range(self._N)])  # mating
        pop = np.clip(pop0, self._xl, self._xu)  # Repair
        penalty = np.linalg.norm(pop - pop0, axis=1) ** 2  # penalty for the box constraints
        f = self._evaluate(pop)  # evaluation & update of z
        g = self.decompose.do(f, self._lambda, utopian_point=self._z, _type="many_to_many")

        g0 = np.diag(g) + alpha * penalty
        cluster = np.argmin(cosine_distances(f - self._z, self._lambda), axis=1)

        for i in range(self._N):
            self._cma[i].g = self.decompose.do(self._cma[i].f, self._lambda[i], utopian_point=self._z)[0, 0]
            self._cma[i].update(pop0[i], pop[i], f[i], g0[i], g[i, i])
            pick = np.where(cluster == i)[0]
            if not len(pick):
                continue
            x, y, z = pop[pick], f[pick], g[pick, i]
            pick = np.argmin(z)
            self._cma[i].inject(x[pick], y[pick], z[pick])

        self._gen += 1
        return pop, f

    def get_cma_means(self):
        return np.array([es.mean for es in self._cma])

    def get_cma_f_means(self):
        return np.array([es.f for es in self._cma])

    def solve(self, n_eval, verbose=True):
        hist = {'f': [], 'x': [], 'f_mu': [], 'mu': []}
        tiktok = time.time()
        while self._evals < n_eval:
            pop, f = self._evolve()
            hist['f'].append(f)  # objective values of the current population
            hist['x'].append(pop)  # current population
            hist['f_mu'].append(self.get_cma_f_means())  # objective values of the means of the CMAs
            hist['mu'].append(self.get_cma_means())  # means of the CMAs
            if verbose and not self._gen % 10:
                print('gen: {}, hv:{}, {:.2f}s'.format(self._gen, self._hv(f), time.time() - tiktok))
                tiktok = time.time()
        return hist

