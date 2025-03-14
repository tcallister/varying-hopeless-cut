import numpyro
nChains = 1
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random
import arviz as az
from baseline_population import baseline
from getData import *

# Run over several chains to check convergence

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections(1.,hopeless_cut=6.3)
sampleDict = getSamples(sample_limit=2000)

# Set up NUTS sampler over our likelihood
kernel = NUTS(baseline)
mcmc = MCMC(kernel,num_warmup=500,num_samples=1000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(191109)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"../data/hopeless_cut_6.3.cdf")

