# #!/bin/bash

SAVE_DIR="residuals_test_dt-1e-5"

mkdir -p $SAVE_DIR/residuals_test_nx-200
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4  --nx 200 --tsteps 100 --dt 1e-5  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/residuals_test_nx-200/CHD_DG-UPW") > "$SAVE_DIR/residuals_test_nx-200/output.txt" 2> "$SAVE_DIR/residuals_test_nx-200/time.txt"

mkdir -p $SAVE_DIR/residuals_test_nx-250
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4  --nx 250 --tsteps 100 --dt 1e-5  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/residuals_test_nx-250/CHD_DG-UPW") > "$SAVE_DIR/residuals_test_nx-250/output.txt" 2> "$SAVE_DIR/residuals_test_nx-250/time.txt"

mkdir -p $SAVE_DIR/residuals_test_nx-300
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4  --nx 300 --tsteps 100 --dt 1e-5  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/residuals_test_nx-300/CHD_DG-UPW") > "$SAVE_DIR/residuals_test_nx-300/output.txt" 2> "$SAVE_DIR/residuals_test_nx-300/time.txt"

mkdir -p $SAVE_DIR/residuals_test_nx-350
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4  --nx 350 --tsteps 100 --dt 1e-5  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/residuals_test_nx-350/CHD_DG-UPW") > "$SAVE_DIR/residuals_test_nx-350/output.txt" 2> "$SAVE_DIR/residuals_test_nx-350/time.txt"

mkdir -p $SAVE_DIR/residuals_test_nx-400
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4  --nx 400 --tsteps 100 --dt 1e-5  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/residuals_test_nx-400/CHD_DG-UPW") > "$SAVE_DIR/residuals_test_nx-400/output.txt" 2> "$SAVE_DIR/residuals_test_nx-400/time.txt"

mkdir -p $SAVE_DIR/residuals_test_nx-450
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4  --nx 450 --tsteps 100 --dt 1e-5  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/residuals_test_nx-450/CHD_DG-UPW") > "$SAVE_DIR/residuals_test_nx-450/output.txt" 2> "$SAVE_DIR/residuals_test_nx-450/time.txt"

mkdir -p $SAVE_DIR/residuals_test_nx-500
(time python CHD_tumor_DG-UPW.py --K 1 --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4  --nx 500 --tsteps 100 --dt 1e-5  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "$SAVE_DIR/residuals_test_nx-500/CHD_DG-UPW") > "$SAVE_DIR/residuals_test_nx-500/output.txt" 2> "$SAVE_DIR/residuals_test_nx-500/time.txt"