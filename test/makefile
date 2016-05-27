NP = 1

test_sample:
	mpirun -n ${NP} python ../demo/driver.py  \
        --config test.cfg                         \
        --model_root ${ILAMB_ROOT}/MODELS/CLM/    \
	--models CLM45bgc_CRUNCEP CLM45bgc_GSWP3  \
        --regions global shsa                     \
	--quiet
	python score_diff.py scores_test.csv.gold scores.csv
