# Trust architecture update

## What changed
- Added a canonical `trust.py` module so latent trust and noisy trust reads are separated.
- Workers, firms, landowners, news firms, and the planner now have a stable trust state.
- News credibility now feeds trust instead of mutating trust in place.
- Employment, investment, and rent decisions now use observed trust reads.
- Step metrics now include trust, institutional trust, polarity, and information-health proxies.
- The architecture experiment now updates trust cleanly per step and respects the `--steps` flag.

## Files
- `trust.py`
- `agents.py`
- `information.py`
- `metrics.py`
- `planner.py`
- `run_architecture_experiment.py`
