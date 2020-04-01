NP = 1
OPTS =

all:
	@ilamb-fetch --remote_root https://www.ilamb.org/Test-Data --no-check-certificate -y
	@mpirun -n ${NP} ilamb-run  \
	--config test.cfg           \
	--model_root ./MODELS/      \
	--regions global shsa       \
	--title "ILAMB Benchmark: test runs" ${OPTS}
	@python score_diff.py scores_test.csv.gold _build/scores.csv 

build:
	@pushd ../; pip install ./; popd

test_run_script:
	nosetests test_run_script.py

clean:
	@rm -rf _build
