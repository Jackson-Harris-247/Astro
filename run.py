import os
import sys
from multiprocessing import Pool, cpu_count
import emcee
import tqdm
import corner
import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from build import Transformations as TR

LOCAL_DATA_FILE = 'data.vot'

# Module-level model functions (must remain top-level for multiprocessing pickling)
KConv = 4.74057

def muLModel(lDeg, bDeg, plx, A, B, C, K, Vx, Vy, Vz):
	l = np.radians(lDeg)
	b = np.radians(bDeg)
	rot = (A * np.cos(2 * l) - C * np.sin(2 * l) + B) * np.cos(b)
	solar = plx * (Vx * np.sin(l) - Vy * np.cos(l))
	return (rot + solar) / KConv

def muBModel(lDeg, bDeg, plx, A, B, C, K, Vx, Vy, Vz):
	l = np.radians(lDeg)
	b = np.radians(bDeg)
	rot = -(A * np.sin(2 * l) + C * np.cos(2 * l) + K) * np.sin(b) * np.cos(b)
	solar = plx * ((Vx * np.cos(l) + Vy * np.sin(l)) * np.sin(b) - Vz * np.cos(b))
	return (rot + solar) / KConv

def logLikelihood(theta, l, b, plx, dist, muLObs, muBObs, sigmaMeasL, sigmaMeasB):
	A, B, C, K, Vx, Vy, Vz, sigmaV = theta
	mL = muLModel(l, b, plx, A, B, C, K, Vx, Vy, Vz)
	mB = muBModel(l, b, plx, A, B, C, K, Vx, Vy, Vz)
	s = sigmaV / (KConv * dist)
	varL = sigmaMeasL**2 + s**2
	varB = sigmaMeasB**2 + s**2
	if np.any(varL <= 0) or np.any(varB <= 0):
		return -np.inf
	val = -0.5 * np.sum((muLObs - mL)**2 / varL + np.log(varL) + (muBObs - mB)**2 / varB + np.log(varB))
	return val if np.isfinite(val) else -np.inf

def logPrior(theta):
	A, B, C, K, Vx, Vy, Vz, sigmaV = theta
	if not (-50 < A < 50 and -50 < B < 50 and -50 < C < 50 and -50 < K < 50):
		return -np.inf
	if not (-100 < Vx < 100 and -100 < Vy < 100 and -100 < Vz < 100):
		return -np.inf
	if not (0 < sigmaV < 100):
		return -np.inf
	return 0.0

def logProbability(theta, l, b, plx, dist, muLObs, muBObs, sigmaMeasL, sigmaMeasB):
	lp = logPrior(theta)
	if not np.isfinite(lp):
		return -np.inf
	ll = logLikelihood(theta, l, b, plx, dist, muLObs, muBObs, sigmaMeasL, sigmaMeasB)
	return (lp + ll) if np.isfinite(ll) else -np.inf

# Main Execution Guard
if __name__ == '__main__':
	# Fetch Gaia credentials from environment variables
	GAIA_USER = os.getenv('GAIA_USER')
	GAIA_PASS = os.getenv('GAIA_PASS')

	if len(sys.argv) >= 2 and sys.argv[1] == 'query':
		if GAIA_PASS:
			Gaia.login(user=GAIA_USER, password=GAIA_PASS)
		cone_radius_deg = 1.0
		ra_list, dec_list = [], []

		print("Generating search parameters for the sky points...")
		for l in np.arange(0, 360, 15.0):
			for b in [-20.0, -10.0, 0.0, 10.0, 20.0]:
				coord = SkyCoord(l=l * u.deg, b=b * u.deg, frame='galactic').icrs
				ra_list.append(coord.ra.degree)
				dec_list.append(coord.dec.degree)

		centers_table = Table({'ra': ra_list, 'dec': dec_list})

		query = f"""
				SELECT DISTINCT
				g.source_id, g.ra, g.dec, g.parallax, g.pmra, g.pmdec,
				g.radial_velocity, g.pmra_error, g.pmdec_error, g.pmra_pmdec_corr
			FROM gaiadr3.gaia_source AS g
			JOIN tap_upload.centers AS u
			ON 1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, {cone_radius_deg}))
			WHERE g.radial_velocity IS NOT NULL
				AND g.parallax_over_error > 10
				AND g.parallax BETWEEN 1.0 AND 5.0
		"""

		print("Submitting cross-match job to Gaia TAP service...")
		job = Gaia.launch_job_async(
			query,
			upload_resource=centers_table,
			upload_table_name="centers"
		)
		data = job.get_results()
		print(f"Retrieved {len(data)} total sources across all regions in 1 query.")
		data.write(LOCAL_DATA_FILE, format='votable', overwrite=True)
	else:
		data = Table.read(LOCAL_DATA_FILE, format='votable')

	# Convert table to C++ CelestialObject instances
	CelestialObjects = [
		TR.CelestialObject(
			TR.Coordinate(
				row["ra"],
				row["dec"],
				row["parallax"],
				row["pmra"],
				row["pmdec"],
				row["radial_velocity"],
				row["pmra_error"],
				row["pmdec_error"],
				row["pmra_pmdec_corr"]
			)
		)
		for row in data
	]

	print(f"Created {len(CelestialObjects)} Celestial Objects.")

	# Executes C++ coordinate transformations and HDBSCAN
	# (Releases GIL in C++ pybind binding to allow multi-threading)
	FilteredObjects = TR.mainProcessing(CelestialObjects)
	print("C++ Data Processing Completed.")

	# Prepare vectors for MCMC
	lArray = np.array([obj.getGACoordinate().getPSVector()[0] for obj in FilteredObjects])
	bArray = np.array([obj.getGACoordinate().getPSVector()[1] for obj in FilteredObjects])
	plxArray = np.array([obj.getEQCoordinate().getPSVector()[2] for obj in FilteredObjects])
	distArray = np.array([obj.getGACoordinate().getPSVector()[2] for obj in FilteredObjects])
	muLObsArray = np.array([obj.getGACoordinate().getPSVector()[4] for obj in FilteredObjects])
	muBObsArray = np.array([obj.getGACoordinate().getPSVector()[5] for obj in FilteredObjects])

	sigmaMeasArray = np.array([obj.getEQCoordinate().calcError() for obj in FilteredObjects])
	sigmaMeasLArray = np.array([sigmaMeas[0] for sigmaMeas in sigmaMeasArray])
	sigmaMeasBArray = np.array([sigmaMeas[1] for sigmaMeas in sigmaMeasArray])

	# Apply valid mask
	valid_mask = (
    	(plxArray >= 1.0) & (plxArray <= 5.0) &
		(distArray > 0.2) & (distArray < 1.0) &
		(np.abs(muLObsArray) < 100) &
		(np.abs(muBObsArray) < 100) &
		(sigmaMeasLArray > 0) &
		(sigmaMeasBArray > 0) &
		np.isfinite(distArray) &
		np.isfinite(sigmaMeasLArray) &
		np.isfinite(sigmaMeasBArray)
	)

	lArray = lArray[valid_mask]
	bArray = bArray[valid_mask]
	plxArray = plxArray[valid_mask]
	distArray = distArray[valid_mask]
	muLObsArray = muLObsArray[valid_mask]
	muBObsArray = muBObsArray[valid_mask]
	sigmaMeasLArray = sigmaMeasLArray[valid_mask]
	sigmaMeasBArray = sigmaMeasBArray[valid_mask]

	# MCMC Parameters
	nwalkers = 32
	ndim = 8
	start = [15.0, -12.0, -3.2, -3.3, 11.0, 12.0, 7.0, 20.0]
	p0 = start + 5e-1 * np.random.randn(nwalkers, ndim)

	initial_args = (lArray, bArray, plxArray, distArray, muLObsArray, muBObsArray, sigmaMeasLArray, sigmaMeasBArray)

	initial_probs = [logProbability(p, *initial_args) for p in p0]
	if not np.all(np.isfinite(initial_probs)):
		raise ValueError("One or more initial walker positions returned -inf or NaN log probability.")

	# Parallel MCMC Sampling via Multiprocessing
	num_cores = cpu_count()
	print(f"Starting MCMC ensemble sampler on {num_cores} CPU cores...")

	with Pool(processes=num_cores) as pool:
		sampler = emcee.EnsembleSampler(
			nwalkers,
			ndim,
			logProbability,
			args=initial_args,
			pool=pool
		)

		print("Running burn-in (500 steps)...")
		state = sampler.run_mcmc(p0, 500, progress=True)
		sampler.reset()

		print("Running main sampling chain (2000 steps)...")
		sampler.run_mcmc(state, 2000, progress=True)

	# Visualizations & Results Export
	flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
	flat_log_prob = sampler.get_log_prob(discard=500, thin=15, flat=True)
	mle_index = np.argmax(flat_log_prob)
	mle_theta = flat_samples[mle_index]

	labels = ["A", "B", "C", "K", "Vx", "Vy", "Vz", "sigmaV"]

	# Corner Plot
	fig = corner.corner(
		flat_samples,
		labels=labels,
		truths=mle_theta,
		truth_color="blue",
		show_titles=True,
		quantiles=[0.02275013, 0.50, 0.97724987]
	)
	fig.savefig("corner_plot.svg")

	# Chain Convergence Plot
	samples = sampler.get_chain()
	fig, axes = plt.subplots(ndim, figsize=(10, 12), sharex=True)

	for i in range(ndim):
		ax = axes[i]
		ax.plot(samples[:, :, i], "k", alpha=0.3)
		ax.set_xlim(0, len(samples))
		ax.set_ylabel(labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)

	axes[-1].set_xlabel("Step number")
	plt.tight_layout()
	plt.savefig("chain_convergence.svg")

	print("Job Finished Successfully.")
