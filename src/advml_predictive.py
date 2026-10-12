"""W7: original CPU teaching implementations, not full-network Bayesian inference.

All noise arguments are standard deviations. Predictive covariance of f excludes
new observation noise; var_y includes it. Data are transparent synthetic data.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, gammaln


def _array(value, ndim=None):
    a = np.asarray(value, dtype=float)
    if not a.size or not np.all(np.isfinite(a)) or (ndim is not None and a.ndim != ndim):
        raise ValueError('Expected a nonempty finite array with the documented dimensions.')
    return a


def truth(x):
    """Known synthetic latent function; not given to the fitted models."""
    x = _array(x)
    return np.sin(1.5*x) + .25*x


def make_data(n=32, noise=.2, seed=19):
    """Return nested training observations and an independent fixed test grid.

    Inputs lie in [-3,-.6] or [.6,3]. For fixed noise/seed, increasing n retains
    all previous pairs. Test outcomes use a separate random stream; no tuning.
    """
    if int(n) != n or not 2 <= n <= 200 or not 0 < noise <= 2:
        raise ValueError('Use integer n in [2,200] and noise SD in (0,2].')
    n=int(n); seed=int(seed)
    rng=np.random.default_rng(seed)
    pool=rng.uniform(-3,3,1000); pool=pool[np.abs(pool)>=.6][:200]
    eps=np.random.default_rng(seed+101).normal(size=200)
    order=np.argsort(pool[:n]); x=pool[:n][order]
    y=truth(x)+noise*eps[:n][order]
    xt=np.linspace(-4,4,241)
    yt=truth(xt)+noise*np.random.default_rng(seed+202).normal(size=len(xt))
    return dict(x=x,y=y,xt=xt,yt=yt,ft=truth(xt),noise=float(noise),seed=seed)


def gaussian_nll(y, mean, variance):
    """Sum Gaussian negative log densities, including normalization constants."""
    y,mean,var=np.broadcast_arrays(_array(y),_array(mean),_array(variance))
    if np.any(var<=0): raise ValueError('Variance must be strictly positive.')
    return float(np.sum(.5*(np.log(2*np.pi*var)+(y-mean)**2/var)))


def bernoulli_nll(y, logits):
    """Stable sum binary cross entropy; logits are real, labels must be 0 or 1."""
    y,z=np.broadcast_arrays(_array(y),_array(logits))
    if not np.all((y==0)|(y==1)): raise ValueError('Binary labels required.')
    return float(np.sum(np.logaddexp(0,z)-y*z))


def poisson_nll(y, log_rate):
    """Sum Poisson NLL. log_rate is eta; exp(eta) is the expected count."""
    y,z=np.broadcast_arrays(_array(y),_array(log_rate))
    if np.any(y<0) or np.any(y!=np.floor(y)) or np.any(z>700):
        raise ValueError('Nonnegative integer counts and finite representable rates required.')
    return float(np.sum(np.exp(z)-y*z+gammaln(y+1)))


def total_variance(weights, means, conditional_variances):
    """Finite mixture mean, within-component variance, between-component variance."""
    w,m,v=(_array(a,1) for a in (weights,means,conditional_variances))
    if w.shape!=m.shape or w.shape!=v.shape or np.any(w<0) or np.any(v<0) or not np.isclose(w.sum(),1):
        raise ValueError('Matching vectors, nonnegative values and normalized weights required.')
    mu=float(w@m); within=float(w@v); between=float(w@((m-mu)**2))
    return dict(mean=mu,within=within,between=between,total=within+between)


def _prediction(mean, cov, noise):
    cov=(cov+cov.T)/2
    if np.min(np.diag(cov)) < -1e-8: raise ArithmeticError('Materially negative predictive variance.')
    vf=np.maximum(np.diag(cov),0)
    return dict(mean=mean,cov_f=cov,var_f=vf,var_y=vf+noise**2)


@dataclass
class BayesianLinear:
    """Exact Gaussian posterior conditional on fixed features and known noise."""
    noise: float=.2
    prior_variance: float=1.

    def fit(self, features, y):
        """features: (n,d), y: (n,); zero-mean isotropic Gaussian weight prior."""
        X=_array(features,2); y=_array(y,1)
        if X.shape[0]!=len(y) or self.noise<=0 or self.prior_variance<=0:
            raise ValueError('Check data dimensions and positive variances.')
        precision=np.eye(X.shape[1])/self.prior_variance+X.T@X/self.noise**2
        factor=cho_factor(precision,lower=True)
        self.cov=cho_solve(factor,np.eye(X.shape[1]))
        self.mean=cho_solve(factor,X.T@y/self.noise**2)
        return self

    def predict(self, features):
        """Return mean (m,), full cov_f (m,m), var_f and var_y (m,)."""
        X=_array(features,2)
        if not hasattr(self,'mean'): raise RuntimeError('Fit the model first.')
        if X.shape[1]!=len(self.mean): raise ValueError('Feature dimension changed.')
        return _prediction(X@self.mean,X@self.cov@X.T,self.noise)


def rbf_kernel(x, z, length=1., amplitude=1.):
    """One-dimensional RBF covariance. amplitude is a function standard deviation."""
    x=_array(x,1);z=_array(z,1)
    if length<=0 or amplitude<=0: raise ValueError('Positive length and amplitude required.')
    return amplitude**2*np.exp(-.5*((x[:,None]-z[None,:])/length)**2)


@dataclass
class GaussianProcess:
    """Zero-mean RBF GP; exact conditional inference with fixed hyperparameters."""
    noise: float=.2
    length: float=1.
    amplitude: float=1.

    def fit(self, x, y):
        """Store a Cholesky factor of K + noise**2 I; never invert K explicitly."""
        self.x=_array(x,1).copy(); y=_array(y,1)
        if len(self.x)!=len(y) or self.noise<=0: raise ValueError('Check data and positive noise SD.')
        K=rbf_kernel(self.x,self.x,self.length,self.amplitude)
        self.factor=cho_factor(K+self.noise**2*np.eye(len(y)),lower=True)
        self.alpha=cho_solve(self.factor,y)
        return self

    def predict(self, x):
        """Joint latent-function posterior; new observation variance adds noise**2."""
        x=_array(x,1)
        if not hasattr(self,'factor'): raise RuntimeError('Fit the model first.')
        cross=rbf_kernel(x,self.x,self.length,self.amplitude)
        cov=rbf_kernel(x,x,self.length,self.amplitude)-cross@cho_solve(self.factor,cross.T)
        return _prediction(cross@self.alpha,cov,self.noise)


@dataclass
class TinyMLP:
    """Point-estimated one-hidden-layer tanh network: a minimal nonlinear predictor.

    This small network illustrates the output head of a DNN; it is not a BNN.
    Objective: mean squared error / 2 + regularization * ||theta||^2 / 2.
    """
    width: int=12
    regularization: float=.0003
    seed: int=19
    maxiter: int=1500

    def _unpack(self, theta):
        h=self.width
        return theta[:h],theta[h:2*h],theta[2*h:3*h],theta[-1]

    def _objective(self, theta, x, y):
        a,b,v,c=self._unpack(theta)
        h=np.tanh(x[:,None]/3*a+b)
        residual=h@v+c-y
        g=residual/len(y); back=g[:,None]*v*(1-h*h)
        grad=np.r_[np.sum(back*x[:,None]/3,axis=0),np.sum(back,axis=0),h.T@g,g.sum()]
        return (.5*np.mean(residual**2)+.5*self.regularization*(theta@theta),
                grad+self.regularization*theta)

    def fit(self, x, y):
        """Fit by L-BFGS; status records convergence, iterations and objective change."""
        x=_array(x,1);y=_array(y,1)
        if len(x)!=len(y) or self.width<1 or self.regularization<0:
            raise ValueError('Invalid training inputs or network settings.')
        theta=np.random.default_rng(self.seed).normal(0,.5,3*self.width+1)
        initial=self._objective(theta,x,y)[0]
        result=minimize(self._objective,theta,args=(x,y),jac=True,method='L-BFGS-B',
                        options={'maxiter':self.maxiter,'ftol':1e-10,'gtol':1e-7})
        if not np.all(np.isfinite(result.x)) or result.fun>initial+1e-9:
            raise ArithmeticError('Optimization did not produce a finite improvement.')
        self.theta=result.x
        self.status=dict(converged=bool(result.success),iterations=int(result.nit),
            message=str(result.message),initial_objective=float(initial),final_objective=float(result.fun))
        return self

    def features(self, x):
        """Frozen learned features (n,width+1), including an intercept column."""
        if not hasattr(self,'theta'): raise RuntimeError('Fit the network first.')
        x=_array(x,1);a,b,_,_=self._unpack(self.theta)
        return np.column_stack([np.ones(len(x)),np.tanh(x[:,None]/3*a+b)])

    def predict(self, x):
        """Point predictions only; repeated calls are not posterior samples."""
        _,_,v,c=self._unpack(self.theta)
        return self.features(x)@np.r_[c,v]


def prior_nn_functions(x, draws=6, width=40, weight_sd=1., seed=7):
    """Prior function draws; no data conditioning. Output weights scale as 1/sqrt(width)."""
    x=_array(x,1)
    if draws<1 or width<1 or weight_sd<=0: raise ValueError('Positive settings required.')
    rng=np.random.default_rng(seed)
    a=rng.normal(0,weight_sd,(draws,width));b=rng.normal(0,1,(draws,width))
    v=rng.normal(0,1/np.sqrt(width),(draws,width))
    return np.einsum('dmx,dm->dx',np.tanh(a[:,:,None]*x+b[:,:,None]),v)


def draw_functions(mean, covariance, draws=5, seed=7):
    """Joint Gaussian function draws; preserve cross-input correlation."""
    m=_array(mean,1);C=_array(covariance,2)
    if C.shape!=(len(m),len(m)) or draws<1: raise ValueError('Check covariance dimensions/draws.')
    if not np.allclose(C,C.T,atol=1e-10): raise ValueError('Covariance must be symmetric.')
    vals,vecs=np.linalg.eigh(C)
    if vals.min() < -1e-8: raise ValueError('Covariance must be positive semidefinite.')
    z=np.random.default_rng(seed).normal(size=(int(draws),len(m)))
    return m+z@(vecs*np.sqrt(np.maximum(vals,0))).T


def evaluate_predictions(y, mean, variance):
    """Observed-y RMSE, mean NLL, empirical pointwise 95% coverage and mean width."""
    y=_array(y,1);m=_array(mean,1);v=_array(variance,1)
    if y.shape!=m.shape or y.shape!=v.shape or np.any(v<=0): raise ValueError('Matching vectors/positive variance required.')
    radius=1.959963984540054*np.sqrt(v)
    return dict(rmse=float(np.sqrt(np.mean((y-m)**2))),
        mean_nll=gaussian_nll(y,m,v)/len(y),coverage_95=float(np.mean(np.abs(y-m)<=radius)),
        mean_width=float(np.mean(2*radius)),count=len(y))
