NP = 1
OPTS =

all:
	@ilamb-fetch --no-check-certificate -y --collection Test-Data
	@mpirun -n ${NP} ilamb-run  \
	--define_regions Test-Data/Whittaker.nc \
	--df_errs `python -c "import ILAMB; print(ILAMB.__path__[0] + '/data/quantiles_Whittaker_cmip5v6.parquet')"` \
	--config test.cfg           \
	--model_root ./Test-Data/MODELS/      \
	--regions global shsa       \
	--title "ILAMB Benchmark: test runs" ${OPTS}
	@python score_diff.py scores_test.csv.gold _build/scores.csv 

build:
	@pushd ../; pip install ./; popd

test_run_script:
	nosetests test_run_script.py

clean:
	@rm -rf _build
