# FEniCS version: 2019.1.0

import numpy as np
import matplotlib.pyplot as plt

T = 400.0
nt = 40000
nx = 30

adv = 100.0
stokes = 1

if adv == 0.0 and stokes != 1:
    # open the file in read binary mode
    file = open("adv-%d/max_u_FE_nt-%d_T-%.3f_P1_adv-%.1f_nx-1000" %(adv,nt,T,adv), "rb")
    #read the file to numpy array
    max_u_FE_exact = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/min_u_FE_nt-%d_T-%.3f_P1_adv-%.1f_nx-1000" %(adv,nt,T,adv), "rb")
    #read the file to numpy array
    min_u_FE_exact = np.load(file)
    #close the file
    file.close

if stokes:
    # open the file in read binary mode
    file = open("stokes/max_u_FE_stokes_nt-%d_T-%.3f_P1" %(nt,T), "rb")
    #read the file to numpy array
    max_u_FE = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("stokes/min_u_FE_stokes_nt-%d_T-%.3f_P1" %(nt,T), "rb")
    #read the file to numpy array
    min_u_FE = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("stokes/max_u_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P1" %(nt,T), "rb")
    #read the file to numpy array
    max_u_DG_SIP_Sigmoidal = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("stokes/min_u_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P1" %(nt,T), "rb")
    #read the file to numpy array
    min_u_DG_SIP_Sigmoidal = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("stokes/max_w_DG-UPW_stokes_nt-%d_T-%.3f_P0" %(nt,T), "rb")
    #read the file to numpy array
    max_u_DG_UPW = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("stokes/min_w_DG-UPW_stokes_nt-%d_T-%.3f_P0" %(nt,T), "rb")
    #read the file to numpy array
    min_u_DG_UPW = np.load(file)
    #close the file
    file.close
else:
    # open the file in read binary mode
    file = open("adv-%d/max_u_FE_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    max_u_FE = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/min_u_FE_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    min_u_FE = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/max_u_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    max_u_DG_SIP_Sigmoidal = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/min_u_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    min_u_DG_SIP_Sigmoidal = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/max_w_DG-UPW_nt-%d_T-%.3f_P0_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    max_u_DG_UPW = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/min_w_DG-UPW_nt-%d_T-%.3f_P0_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    min_u_DG_UPW = np.load(file)
    #close the file
    file.close

time_steps = np.linspace(0,T,len(max_u_FE))

fig, axs = plt.subplots(2)

# fig.suptitle("Max and min")

axs[0].plot(time_steps,np.array([1 for i in range(len(time_steps))]),'-',c='lightgray',linewidth=2,label='_nolegend_')
# if adv == 0.0:
#     axs[0].plot(time_steps,max_u_FE_exact,'',c='blue',linewidth=1, markersize=3, markevery=10)
axs[0].plot(time_steps,max_u_FE,'--',c='orange')
axs[0].plot(time_steps,max_u_DG_SIP_Sigmoidal,':',c='red')
axs[0].plot(time_steps,max_u_DG_UPW,'-.',c='green')
axs[0].grid(linestyle='--',color='lightgray')

axs[1].plot(time_steps,np.array([0 for i in range(len(time_steps))]),'-',c='lightgray',linewidth=2,label='_nolegend_')
# if adv == 0.0:
#     axs[1].plot(time_steps,min_u_FE_exact,'.-',c='blue')
axs[1].plot(time_steps,min_u_FE,'--',c='orange')
axs[1].plot(time_steps,min_u_DG_SIP_Sigmoidal,':',c='red')
axs[1].plot(time_steps,min_u_DG_UPW,'-.',c='green')
axs[1].grid(linestyle='--',color='lightgray')

plt.subplots_adjust(hspace=0.5, bottom=0.16)

# plt.figlegend(['Exact','FE','DG-SIP','DG-UPW'], loc = 'center', ncol=4, labelspacing=0. )
plt.figlegend(['FE','DG-SIP','DG-UPW'], loc = 'center', ncol=3, labelspacing=0. )


plt.show()
plt.close()

# plt.plot(time_steps,np.array([1 for i in range(len(time_steps))]),'--',c='gray',)
# plt.plot(time_steps,np.array([0 for i in range(len(time_steps))]),'--',c='gray',)
# plt.plot(time_steps,max_u_FE)
# plt.plot(time_steps,min_u_FE)
# plt.plot(time_steps,max_u_DG_SIP_Sigmoidal)
# plt.plot(time_steps,min_u_DG_SIP_Sigmoidal)
# plt.plot(time_steps,max_u_DG_UPW)
# plt.plot(time_steps,min_u_DG_UPW)
# plt.show()
# plt.close()

if adv == 0.0 and stokes != 1:
    # open the file in read binary mode
    file = open("adv-%d/energy_FE_nt-%d_T-%.3f_P1_adv-%.1f_nx-1000" %(adv,nt,T,adv), "rb")
    #read the file to numpy array
    energy_FE_exact = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/energy_FE_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    energy_FE = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/energy_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    energy_DG_SIP_Sigmoidal = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("adv-%d/energy_DG-UPW_nt-%d_T-%.3f_P0_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    #read the file to numpy array
    energy_DG_UPW = np.load(file)
    #close the file
    file.close

    # file = open("adv-%d/energy_DG-UPW_coupled_nt-%d_T-%.3f_P0_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
    # #read the file to numpy array
    # energy_DG_UPW_coupled = np.load(file)
    # #close the file
    # file.close

    # plt.title("Energy")
    # plt.plot(time_steps,energy_FE_exact,'-',c='blue')
    plt.plot(time_steps,energy_FE,'--',c='orange')
    plt.plot(time_steps,energy_DG_SIP_Sigmoidal,':',c='red')
    # plt.plot(time_steps,energy_DG_UPW_coupled,'o-',c='yellow',markersize=2)
    plt.plot(time_steps,energy_DG_UPW,'-.',c='green')
    # plt.legend(['Exact','FE','DG-SIP','DG-UPW'], loc = 'upper right')
    plt.legend(['FE','DG-SIP','DG-UPW'], loc = 'upper right')
    plt.grid(linestyle='--',color='lightgray')
    plt.show()
    plt.close()

if stokes:
    # open the file in read binary mode
    file = open("stokes/dynamics_FE_stokes_nt-%d_T-%.3f_P1" %(nt,T), "rb")
    #read the file to numpy array
    dynam_FE = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("stokes/dynamics_DG-SIP-Sigmoidal_stokes_nt-%d_T-%.3f_P1" %(nt,T), "rb")
    #read the file to numpy array
    dynam_DG_SIP_Sigmoidal = np.load(file)
    #close the file
    file.close

    # open the file in read binary mode
    file = open("stokes/dynamics_w_DG-UPW_stokes_nt-%d_T-%.3f_P0" %(nt,T), "rb")
    #read the file to numpy array
    dynam_DG_UPW = np.load(file)
    #close the file
    file.close

else:
    if adv>0.0:
        # open the file in read binary mode
        file = open("adv-%d/dynamics_FE_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
        #read the file to numpy array
        dynam_FE = np.load(file)
        #close the file
        file.close

        # open the file in read binary mode
        file = open("adv-%d/dynamics_DG-SIP-Sigmoidal_nt-%d_T-%.3f_P1_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
        #read the file to numpy array
        dynam_DG_SIP_Sigmoidal = np.load(file)
        #close the file
        file.close

        # open the file in read binary mode
        file = open("adv-%d/dynamics_w_DG-UPW_nt-%d_T-%.3f_P0_adv-%.1f_nx-%d" %(adv,nt,T,adv,nx), "rb")
        #read the file to numpy array
        dynam_DG_UPW = np.load(file)
        #close the file
        file.close

if (stokes or adv>0.0):
    plt.plot(time_steps[1:len(time_steps)],dynam_FE,'--',c='orange')
    plt.plot(time_steps[1:len(time_steps)],dynam_DG_SIP_Sigmoidal,':',c='red')
    # plt.plot(time_steps,energy_DG_UPW_coupled,'o-',c='yellow',markersize=2)
    plt.plot(time_steps[1:len(time_steps)],dynam_DG_UPW,'-.',c='green')
    plt.legend(['FE','DG-SIP','DG-UPW'], loc = 'upper right')
    plt.grid(linestyle='--',color='lightgray')
    plt.show()
    plt.close()
