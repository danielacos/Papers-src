import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
plt.rcParams['text.latex.preamble'] = r'\boldmath'
plt.rc('font', size=15)
plt.rcParams['mathtext.fontset'] = 'cm'

ls = ['--', ':', '-.']
markers = ['o', 's', '*']
color = ['orange', 'red', 'blue']

K_list = ['1e-1', '1', '10']
K_list_legend = [r'$K=0.1$', r'$K=1$', r'$K=10$']

# for test in ['reference_test', 'test_chi-0.01', 'test_chi-0.5', 'test_chi-1', 'test_P0-0.001', 'test_P0-0.05', 'test_P0-2']:
for test in ['reference_test']:
    i = 0
    while i < 2:
        time_vector = []
        max_u_list = []
        min_u_list = []
        max_n_list = []
        min_n_list = []
        energy = []

        for K in K_list:
            file_path = f'tests/tests_irregular_shape_K-{K}/{test}'

            if i == 1: file_path += "_symmetric"

            df = pd.read_csv(file_path + "/output.txt", sep=' ', header=36, index_col=0, skipinitialspace=True)
            print(df)

            time_vector.append(df['t'].array)

            max_u_list.append(df['u_max'].array)
            min_u_list.append(df['u_min'].array)
            max_n_list.append(df['n_max'].array)
            min_n_list.append(df['n_min'].array)
            energy.append(df['energy'].array)

            print(f"Test: {test}, K: {K}:")
            print(f"Max u: {max(max_u_list[-1]):.4e}, Min u: {min(min_u_list[-1]):.4e}")
            print(f"Max n: {max(max_n_list[-1]):.4e}, Min n: {min(min_n_list[-1]):.4e}\n")
        
        fig_u, axs_u = plt.subplots(2)
        for j in range(len(max_u_list)):
            axs_u[0].plot(time_vector[j],np.maximum(max_u_list[j]-1,0),ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)
            axs_u[1].plot(time_vector[j],np.maximum(-min_u_list[j],0),ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)

            axs_u[0].set_xlabel(r'$t$')
            axs_u[0].set_ylabel(r'$\|\max(u^m-1,0)\|_{L^\infty(\Omega)}$', size=11)
            order_max_u = np.floor(np.log10(max(np.maximum(max_u_list[j]-1, 1e-16))))
            axs_u[0].set_ylim(-10**(order_max_u), 10**(order_max_u))
            axs_u[0].set_yticks([-10**(order_max_u), 0, 10**(order_max_u)])

            axs_u[1].set_xlabel(r'$t$')
            axs_u[1].set_ylabel(r'$\|\max(-u^m,0)\|_{L^\infty(\Omega)}$', size=11)
            order_min_u = np.floor(np.log10(abs(max(np.maximum(-min_u_list[j], 1e-16)))))
            if i==0:
                axs_u[1].set_ylim(-5*10**(order_min_u),5*10**(order_min_u))
                axs_u[1].set_yticks([-5*10**(order_min_u), 0, 5*10**(order_min_u)])
            else:
                axs_u[1].set_ylim(-2*10**(order_min_u),2*10**(order_min_u))
                axs_u[1].set_yticks([-2*10**(order_min_u), 0, 2*10**(order_min_u)])

            axs_u[0].legend(facecolor='white', loc='lower center', ncol=3)
            axs_u[1].legend(facecolor='white', loc='lower center', ncol=3)
            plt.subplots_adjust(hspace=0.5, bottom=0.16)
            plt.gcf().tight_layout() # tighter figure for better visualization
            if i==0:
                plt.savefig('tests/' + test + '_min_max_u.pdf', format='pdf')
            else:
                plt.savefig('tests/' + test + '_min_max_u_symmetric.pdf', format='pdf')
            # plt.show()
        plt.close(fig_u)

        fig_n, axs_n = plt.subplots(2)
        for j in range(len(max_n_list)):
            axs_n[0].plot(time_vector[j],np.maximum(max_n_list[j]-1,0),ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)
            axs_n[1].plot(time_vector[j],np.maximum(-min_n_list[j],0),ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)
            axs_n[0].set_xlabel(r'$t$', size=11)
            axs_n[0].set_ylabel(r'$\|\max(n^m-1,0)\|_{L^\infty(\Omega)}$', size=11)
            order_max_n = np.floor(np.log10(max(np.maximum(max_n_list[j]-1, 1e-16))))
            axs_n[0].set_ylim(-10**(order_max_n), 10**(order_max_n))
            axs_n[0].set_yticks([-10**(order_max_n), 0, 10**(order_max_n)])

            axs_n[1].set_xlabel(r'$t$', size=11)
            axs_n[1].set_ylabel(r'$\|\max(-n^m,0)\|_{L^\infty(\Omega)}$', size=11)
            order_min_n = np.floor(np.log10(abs(max(np.maximum(-min_n_list[j], 1e-16)))))
            axs_n[1].set_ylim(-10**(order_min_n),10**(order_min_n))
            axs_n[1].set_yticks([-10**(order_min_n), 0, 10**(order_min_n)])

            axs_n[0].legend(facecolor='white', loc='lower center', ncol=3)
            axs_n[1].legend(facecolor='white', loc='lower center', ncol=3)
            plt.subplots_adjust(hspace=0.5, bottom=0.16)
            plt.gcf().tight_layout() # tighter figure for better visualization
            if i==0:
                plt.savefig('tests/' + test + '_min_max_n.pdf', format='pdf')
            else:
                plt.savefig('tests/' + test + '_min_max_n_symmetric.pdf', format='pdf')
            # plt.show()
        plt.close(fig_n)

        for j in range(len(energy)):
            plt.plot(time_vector[j], energy[j], ls[j], color=color[j], label=K_list_legend[j], marker=markers[j], markevery=0.1)
            # plt.title("Discrete energy")
            plt.xlabel(r'$t$')
            plt.ylabel(r'$E(\Pi_1^h u^m, n^m)$')
            plt.legend(facecolor='white')
            plt.gcf().tight_layout() # tighter figure for better visualization
            if i==0:
                plt.savefig('tests/' + test + '_energy.pdf', format='pdf')
            else:
                plt.savefig('tests/' + test + '_energy_symmetric.pdf', format='pdf')
            # plt.show()
        plt.close()
        
        i += 1