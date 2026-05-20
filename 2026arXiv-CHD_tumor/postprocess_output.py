import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
plt.rcParams['text.latex.preamble'] = r'\boldmath'
plt.rc('font', size=15)

ls = ['--', ':', '-.']
markers = ['o', 's', '*']
color = ['orange', 'red', 'blue']

K_list = ['1e-1', '1', '10']
K_list_legend = [r'$K=0.1$', r'$K=1$', r'$K=10$']

for test in ['reference_test', 'test_chi-0.01', 'test_chi-0.5', 'test_chi-1', 'test_P0-0.001', 'test_P0-0.05', 'test_P0-2']:
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

            df = pd.read_csv(file_path + "/output.txt", sep=' ', header=33, index_col=0, skipinitialspace=True)
            print(df)

            time_vector.append(df['t'].array)

            max_u_list.append(df['u_max'].array)
            min_u_list.append(df['u_min'].array)
            max_n_list.append(df['n_max'].array)
            min_n_list.append(df['n_min'].array)
            energy.append(df['energy'].array)
        
        fig_u, axs_u = plt.subplots(2)
        for j in range(len(max_u_list)):
            axs_u[0].plot(time_vector[j],max_u_list[j],ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)
            axs_u[1].plot(time_vector[j],min_u_list[j],ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)
            axs_u[0].set_xlabel(r'$t$')
            axs_u[0].set_ylabel(r'$\max\ u(t)$')
            axs_u[0].set_ylim(0.9, 1.05)
            axs_u[0].set_yticks([0.9,0.95, 1, 1.05])
            # axs_u[1].plot(time_vector,np.zeros(len(time_vector)),'-',c='w',label='_nolegend_')
            axs_u[1].set_xlabel(r'$t$')
            axs_u[1].set_ylabel(r'$\min\ u(t)$')
            axs_u[1].set_ylim(-0.05, 0.05)
            axs_u[1].set_yticks([-0.05, 0, 0.05])
            axs_u[0].legend(facecolor='white', loc='lower center', ncol=3)
            axs_u[1].legend(facecolor='white', loc='lower center', ncol=3)
            plt.subplots_adjust(hspace=0.5, bottom=0.16)
            plt.gcf().tight_layout() # tighter figure for better visualization
            if i==0:
                plt.savefig('tests/' + test + '_min_max_u.png')
            else:
                plt.savefig('tests/' + test + '_min_max_u_symmetric.png')
            # plt.show()
        plt.close(fig_u)

        fig_n, axs_n = plt.subplots(2)
        for j in range(len(max_n_list)):
            axs_n[0].plot(time_vector[j],max_n_list[j],ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)
            axs_n[1].plot(time_vector[j],min_n_list[j],ls[j],c=color[j],label=K_list_legend[j], marker=markers[j], markevery=0.1)
            axs_n[0].set_xlabel(r'$t$')
            axs_n[0].set_ylabel(r'$\max\ n(t)$')
            axs_n[0].set_ylim(top=1.05)
            # axs_n[1].plot(time_vector,np.zeros(len(time_vector)),'-',c='w',label='_nolegend_')
            axs_n[1].set_xlabel(r'$t$')
            axs_n[1].set_ylabel(r'$\min\ n(t)$')
            if max(min_n_list[j]) > 0.05:
                axs_n[1].set_ylim(bottom=-0.05)
            else:
                axs_n[1].set_ylim(-0.05, 0.05)
                axs_n[1].set_yticks([-0.05, 0, 0.05])
            axs_n[0].legend(facecolor='white')
            axs_n[1].legend(facecolor='white', loc='lower center', ncol=3)
            plt.subplots_adjust(hspace=0.5, bottom=0.16)
            plt.gcf().tight_layout() # tighter figure for better visualization
            if i==0:
                plt.savefig('tests/' + test + '_min_max_n.png')
            else:
                plt.savefig('tests/' + test + '_min_max_n_symmetric.png')
            # plt.show()
        plt.close(fig_n)

        for j in range(len(energy)):
            plt.plot(time_vector[j], energy[j], ls[j], color=color[j], label=K_list_legend[j], marker=markers[j], markevery=0.1)
            # plt.title("Discrete energy")
            plt.xlabel(r'$t$')
            plt.ylabel(r'$E(t)$')
            plt.legend(facecolor='white')
            plt.gcf().tight_layout() # tighter figure for better visualization
            if i==0:
                plt.savefig('tests/' + test + '_energy.png')
            else:
                plt.savefig('tests/' + test + '_energy_symmetric.png')
            # plt.show()
        plt.close()
        
        i += 1