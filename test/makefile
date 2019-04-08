NP = 1
OPTS = ''

test_sample:
	@mpirun -n ${NP} ilamb-run                \
        --config test.cfg                         \
        --model_root ${ILAMB_ROOT}/MODELS/TestModels \
        --regions global shsa \
	--title "ILAMB Benchmark: test runs" \
	${OPTS}
	python score_diff.py scores_test.csv.gold _build/scores.csv 

test_run_script:
	nosetests test_run_script.py

clean:
	@rm -rf _build
