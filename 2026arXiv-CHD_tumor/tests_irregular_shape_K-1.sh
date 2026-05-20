#!/bin/bash

SERVER=1
SAVE_DIR="tests_irregular_shape_K-1"

###

mkdir -p tests_irregular_shape_K-1/reference_test
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1 --nx 100 --tsteps 500 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/reference_test/CHD_DG-UPW" --server $SERVER) > "$SAVE_DIR/reference_test/output.txt" 2> "$SAVE_DIR/reference_test/time.txt"

mkdir -p tests_irregular_shape_K-1/reference_test_symmetric
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1 --nx 100 --tsteps 500 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/reference_test_symmetric/CHD_DG-UPW" --server $SERVER --symmetric 1) > "$SAVE_DIR/reference_test_symmetric/output.txt" 2> "$SAVE_DIR/reference_test_symmetric/time.txt"

####

mkdir -p $SAVE_DIR/test_P0-2
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 2.0 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 500 --dt 0.025  --plot 10 --save 1 --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_P0-2/CHD_DG-UPW" --server $SERVER) > "$SAVE_DIR/test_P0-2/output.txt" 2> "$SAVE_DIR/test_P0-2/time.txt"

mkdir -p $SAVE_DIR/test_P0-2_symmetric
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 2.0 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 500 --dt 0.025  --plot 10 --save 1 --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_P0-2_symmetric/CHD_DG-UPW" --server $SERVER --symmetric 1) > "$SAVE_DIR/test_P0-2_symmetric/output.txt" 2> "$SAVE_DIR/test_P0-2_symmetric/time.txt"

# ####

mkdir -p $SAVE_DIR/test_P0-0.001
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.001 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 1000 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_P0-0.001/CHD_DG-UPW" --server $SERVER) > "$SAVE_DIR/test_P0-0.001/output.txt" 2> "$SAVE_DIR/test_P0-0.001/time.txt"

mkdir -p $SAVE_DIR/test_P0-0.001_symmetric
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.001 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 1000 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_P0-0.001_symmetric/CHD_DG-UPW" --server $SERVER --symmetric 1) > "$SAVE_DIR/test_P0-0.001_symmetric/output.txt" 2> "$SAVE_DIR/test_P0-0.001_symmetric/time.txt"

# ####

mkdir -p $SAVE_DIR/test_P0-0.05
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.05 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 2000 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_P0-0.05/CHD_DG-UPW" --server $SERVER) > "$SAVE_DIR/test_P0-0.05/output.txt" 2> "$SAVE_DIR/test_P0-0.05/time.txt"

mkdir -p $SAVE_DIR/test_P0-0.05_symmetric
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.05 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 2000 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_P0-0.05_symmetric/CHD_DG-UPW" --server $SERVER --symmetric 1) > "$SAVE_DIR/test_P0-0.05_symmetric/output.txt" 2> "$SAVE_DIR/test_P0-0.05_symmetric/time.txt"

# ####

mkdir -p $SAVE_DIR/test_chi-1
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 1.0 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 1000 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_chi-1/CHD_DG-UPW" --server $SERVER) > "$SAVE_DIR/test_chi-1/output.txt" 2> "$SAVE_DIR/test_chi-1/time.txt"

mkdir -p $SAVE_DIR/test_chi-1_symmetric
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 1.0 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 1000 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_chi-1_symmetric/CHD_DG-UPW" --server $SERVER --symmetric 1) > "$SAVE_DIR/test_chi-1_symmetric/output.txt" 2> "$SAVE_DIR/test_chi-1_symmetric/time.txt"

# ####

mkdir -p $SAVE_DIR/test_chi-0.5
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.5 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 1700 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_chi-0.5/CHD_DG-UPW" --server $SERVER) > "$SAVE_DIR/test_chi-0.5/output.txt" 2> "$SAVE_DIR/test_chi-0.5/time.txt"

mkdir -p $SAVE_DIR/test_chi-0.5_symmetric
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.5 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 1700 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_chi-0.5_symmetric/CHD_DG-UPW" --server $SERVER --symmetric 1) > "$SAVE_DIR/test_chi-0.5_symmetric/output.txt" 2> "$SAVE_DIR/test_chi-0.5_symmetric/time.txt"

# ####

mkdir -p $SAVE_DIR/test_chi-0.01
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.01 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 500 --dt 0.1 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_chi-0.01/CHD_DG-UPW" --server $SERVER) > "$SAVE_DIR/test_chi-0.01/output.txt" 2> "$SAVE_DIR/test_chi-0.01/time.txt"

mkdir -p $SAVE_DIR/test_chi-0.01_symmetric
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.01 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --eta -1  --nx 100 --tsteps 500 --dt 0.1 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/test_chi-0.01_symmetric/CHD_DG-UPW" --server $SERVER --symmetric 1) > "$SAVE_DIR/test_chi-0.01_symmetric/output.txt" 2> "$SAVE_DIR/test_chi-0.01_symmetric/time.txt"