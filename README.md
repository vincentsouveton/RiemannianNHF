# Riemannian Neural Hamiltonian Flows (RNHF)

PyTorch code for the paper *Riemannian Neural Hamiltonian Flows: Geodesic
Symplectic Transport and Interpretability*.
It contains RNHF and its baseline RCNF on the plane R^2, the hyperbolic plane
H^2 and the sphere S^2 (Sections 4.1-4.3), and the surrogate Hamiltonian
Monte Carlo experiment on IceCube data (Section 4.4).

Everything runs on a CPU, in float64, with fixed random seeds.

## Files

| file | content |
|---|---|
| `geometry.py` | the three model spaces: exponential map, logarithm, parallel transport, distance, tangent frames |
| `targets.py` | targets and base distributions |
| `models.py` | RNHF (geodesic kick-drift-kick integrator, ELBO) and RCNF |
| `train.py`, `evaluate.py` | training and evaluation of one config, outputs in `results/<config name>/` |
| `metrics.py` | metrics of Appendix I |
| `plotting.py` | diagnostic figures of `evaluate.py` |
| `refocusing.py` | Figure 6 |
| `make_figures.py` | Figures 3, 4 and 5 and the figure of Section 4.4 |
| `hmc.py` | geodesic HMC on S^2, exact or with a surrogate energy (Section 3.3) |
| `icecube.py` | IceCube data, point-source energy (Eq. 21), maximum of the likelihood |
| `icecube_experiment.py`, `icecube_summary.py` | experiment and metrics of Section 4.4 |
| `configs/` | one YAML file per experiment |

## Installation

```bash
pip install -r requirements.txt
```

## Sections 4.1-4.3: R^2, H^2, S^2

```bash
for s in r2 h2 s2; do
  python train.py --config configs/$s.yaml --model rnhf
  python train.py --config configs/$s.yaml --model rcnf
  python evaluate.py --config configs/$s.yaml          # metrics.json, scores of Section 4.3
done
python train.py --config configs/r2_gauss.yaml --model rnhf   # Gaussian target of Figure 4

python make_figures.py samples      # figures/samples_all.png
python make_figures.py lens         # figures/hamiltonian_lens_learned.png
python make_figures.py potentials   # figures/potentials_all.png
python refocusing.py                # figures/refocusing_sigma_p.png
```

`evaluate.py` regenerates the training set of `train.py` (same seed), so that
the metrics compare each model with its own training data. Training an RNHF
takes a few minutes per space.

## Section 4.4: IceCube

Download the public point-source sample 2008-2018 (IceCube Collaboration,
2021, Harvard Dataverse, doi:10.7910/DVN/VKL316) and unzip it so that the event
files are in `data/icecube_10year_ps/events/*.tab`; only the `events` folder is
used. Then

```bash
python icecube_experiment.py --seed 0
python icecube_experiment.py --seed 1
python icecube_experiment.py --seed 2
python icecube_experiment.py --seed 100 --reference-only    # long reference run
python icecube_experiment.py --seed 101 --reference-only    # (two chains, pooled)
python icecube_summary.py --long-seeds 100 101              # metrics, figures/icecube_table.tex
python make_figures.py icecube                              # figures/icecube_posterior.png
```

The first call builds `results/icecube/dataset.pt` (scrambled right
ascensions, injected events around NGC 1068, background density, maximum of
the likelihood), shared by all seeds. Each seed then runs exact geodesic HMC
from the maximum of the likelihood (burn-in, then M = 10,000 states), trains
RNHF for 200 epochs on the first N = 1,000 states, and runs the surrogate HMC
with E_hat = -log U for the remaining 9,000 transitions. Each stage is
saved in `results/icecube/seed<s>/` and reused if the script is run again;
`report.json` holds the metrics of Section 4.4. An evaluation of the energy
sums over the 1.13 million events, so that one seed takes about one hour on two
CPU cores, almost all of it in the exact HMC chain.
