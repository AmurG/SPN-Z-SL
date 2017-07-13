import numpy as np
from scipy.stats import multivariate_normal

#A few things to understand here : the 1e-4 bit is due to the fact that the covariance matrix, if not enforced with that bit, can end up with too many zeroes and hence singular. This is a problem : such a covariance matrix doesn't define a valid probability distribution and cannot be used to get the logpdf method ( you can try this yourself by removing that part and testing on a dataset with many zeroes such as GasSenH ) 


class MultiNormalStat:

	@staticmethod
	def create(nvar):
		stat = MultiNormalStat()
		stat.mean = np.zeros(nvar)
		stat.cov = np.identity(nvar)
		return stat

	@staticmethod
	def create_copy(mean, cov):
		stat = MultiNormalStat()
		stat.mean = mean.copy()
		stat.cov = cov.copy()
		return stat

	def __repr__(self):
		return "Normal({0}, {1})".format(self.mean, self.cov.flatten())

	def evaluate(self, x):
		try:
			return multivariate_normal.logpdf(x, self.mean, self.cov)
		except:
			self.cov[np.diag_indices_from(self.cov)] += 1e-4
			return multivariate_normal.logpdf(x, self.mean, self.cov)

#The part where the covariance and mean get updated ( the update function ) is covered here : https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

	def update(self, x, n):
		k = x.shape[0]
		mean = self.mean
		cov = self.cov

		mean_new = (n * mean + x.sum(axis=0)) / (n + k)

		dx = x - mean
		dm = mean_new - mean

		cov_new = (n*cov + dx.T.dot(dx)) / (n + k) - np.outer(dm, dm)

		self.mean = mean_new
		self.cov = cov_new

	def iterate_corrs(self, corrthresh):
		v = np.diag(self.cov).copy()
		v[v<1e-4] = 1e-4
		corrs = np.abs(self.cov) / np.sqrt(np.outer(v, v))
		rows, cols = np.unravel_index(np.argsort(corrs.flatten()), corrs.shape)

		for i, j in zip(reversed(rows), reversed(cols)):
			if corrs[i, j] < corrthresh:
				break
			yield i, j

	def distill(self):
		return self.mean

	def extract(self, ind):
		return MultiNormalStat.create_copy(self.mean[ind], self.cov[np.ix_(ind,ind)])

	def extract_from_obs(self, ind, x):
		cov = self.cov[np.ix_(ind,ind)]
		stat = MultiNormalStat.create_copy(x.mean(axis=0), np.diag(np.diag(cov)))
		return stat

