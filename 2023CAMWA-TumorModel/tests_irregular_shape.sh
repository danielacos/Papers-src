#!/bin/bash

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 500 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "reference_test/tumor_DG-UPW" --server 1 | tee "reference_test/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 500 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "reference_test_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "reference_test_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 2.0 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 500 --dt 0.025  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_P0-2/tumor_DG-UPW" --server 1 | tee "test_P0-2/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 2.0 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 2000 --dt 0.025  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_P0-2_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_P0-2_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.001 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 2500 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_P0-0.001/tumor_DG-UPW" --server 1 | tee "test_P0-0.001/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.001 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 2000 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_P0-0.001_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_P0-0.001_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.05 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 5000 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_P0-0.05/tumor_DG-UPW" --server 1 | tee "test_P0-0.05/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.1 --P0 0.05 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 2000 --dt 0.1  --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_P0-0.05_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_P0-0.05_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1.0 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 1250 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_chi-1/tumor_DG-UPW" --server 1 | tee "test_chi-1/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1.0 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 1250 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_chi-1_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_chi-1_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.5 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 2500 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_chi-0.5/tumor_DG-UPW" --server 1 | tee "test_chi-0.5/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.5 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 2500 --dt 0.01 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_chi-0.5_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_chi-0.5_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.01 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 500 --dt 0.1 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_chi-0.01/tumor_DG-UPW" --server 1 | tee "test_chi-0.01/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 0.01 --P0 0.5 --Cu 2.8 --Cn 2.8e-4 --nx 100 --tsteps 500 --dt 0.1 --plot 10 --save 1  --initial_cond 'single_tumor' --savefile "test_chi-0.01_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_chi-0.01_symmetric/output.txt"