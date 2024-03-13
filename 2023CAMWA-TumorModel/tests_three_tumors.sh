#!/bin/bash

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 25.0 --Cu 100 --Cn 100 --nx 100 --tsteps 250 --dt 0.001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-25_dt-1e-3_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-25_dt-1e-3_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 25.0 --Cu 100 --Cn 100 --nx 100 --tsteps 250 --dt 0.001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-25_dt-1e-3_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-25_dt-1e-3_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 25.0 --Cu 100 --Cn 100 --nx 100 --tsteps 2500 --dt 0.0001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-25_dt-1e-4_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-25_dt-1e-4_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 25.0 --Cu 100 --Cn 100 --nx 100 --tsteps 2500 --dt 0.0001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-25_dt-1e-4_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-25_dt-1e-4_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 25.0 --Cu 100 --Cn 100 --nx 100 --tsteps 25000 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-25_dt-1e-5_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-25_dt-1e-5_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 25.0 --Cu 100 --Cn 100 --nx 100 --tsteps 25000 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-25_dt-1e-5_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-25_dt-1e-5_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 75.0 --Cu 100 --Cn 100 --nx 100 --tsteps 250 --dt 0.001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-75_dt-1e-3_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-75_dt-1e-3_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 75.0 --Cu 100 --Cn 100 --nx 100 --tsteps 250 --dt 0.001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-75_dt-1e-3_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-75_dt-1e-3_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 75.0 --Cu 100 --Cn 100 --nx 100 --tsteps 2500 --dt 0.0001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-75_dt-1e-4_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-75_dt-1e-4_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 75.0 --Cu 100 --Cn 100 --nx 100 --tsteps 2500 --dt 0.0001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-75_dt-1e-4_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-75_dt-1e-4_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 75.0 --Cu 100 --Cn 100 --nx 100 --tsteps 25000 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-75_dt-1e-5_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-75_dt-1e-5_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 75.0 --Cu 100 --Cn 100 --nx 100 --tsteps 25000 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-75_dt-1e-5_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-75_dt-1e-5_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 100 --tsteps 250 --dt 0.001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-125_dt-1e-3_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-125_dt-1e-3_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 100 --tsteps 250 --dt 0.001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-125_dt-1e-3_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-125_dt-1e-3_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 100 --tsteps 2500 --dt 0.0001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-125_dt-1e-4_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-125_dt-1e-4_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 100 --tsteps 2500 --dt 0.0001  --plot 10 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-125_dt-1e-4_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-125_dt-1e-4_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 100 --tsteps 7500 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-125_dt-1e-5_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-125_dt-1e-5_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 100 --tsteps 7500 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-125_dt-1e-5_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-125_dt-1e-5_symmetric/output.txt"

####

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 200 --tsteps 7500 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-125_dt-1e-5_nx-200_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-125_dt-1e-5_nx-200_symmetric/output.txt"

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 1e-16 --P0 125.0 --Cu 100 --Cn 100 --nx 200 --tsteps 7500 --dt 0.00001  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-125_dt-1e-5_nx-200_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-125_dt-1e-5_nx-200_symmetric/output.txt"

####

python tumor_FEM_fenicsx.py --eps 0.1 --delta 0.01 --chi0 10 --P0 125.0 --Cu 100 --Cn 100 --nx 200 --tsteps 8000 --dt 0.000005  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_FEM_P0-125_chi-10_dt-5e-6_nx-200_symmetric/tumor_FEM" --server 1 --symmetric 1 | tee "test_FEM_P0-125_chi-10_dt-5e-6_nx-200_symmetric/output.txt"

python tumor_DG-UPW_fenicsx.py --eps 0.1 --delta 0.01 --chi0 10 --P0 125.0 --Cu 100 --Cn 100 --nx 200 --tsteps 8000 --dt 0.000005  --plot 100 --save 1  --initial_cond 'three_tumors' --savefile "test_DG_P0-125_chi-10_dt-5e-6_nx-200_symmetric/tumor_DG-UPW" --server 1 --symmetric 1 | tee "test_DG_P0-125_chi-10_dt-5e-6_nx-200_symmetric/output.txt"
