import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib


class fourpop_dend():
    def __init__(self, GRIK4: float, D2: iter,  custom_args:dict = None, p: dict = None):
        # Parameters
        if p != None:
            # print("Using custom parameters")
            self.p = p
        else:
            # print("No parameter dict given. Creating model with default parameters")
            default_params = self.default_params()
            self.p = default_params

        if custom_args:
            self.p.update(custom_args)
            
        # set important attr from params
        self.NT = int(self.p['T']/self.p['dt'])
        self.dt = self.p['dt']
        self.timerange = np.linspace(0, self.p['T'], self.NT)
        self.GRIK4 = GRIK4
        self.D2 = D2

        # Create the D2 and GRIK4 arrays
        self.I_GRIK = np.array([0, 0, self.GRIK4*self.p['JGRIKPV'], self.GRIK4*self.p['JGRIKPV']])
                
        # create background current array (since this comes up a lot)
        self.I0 = np.array([self.p['I0E'], self.p['I0E'], self.p['I0I'], self.p['I0I']])

        # create adjacency matrix
        self.A = self.adjacency()
    
    def default_params(self):
        p = {}
        # ---------- PARAMS ------------
        # firing rate 
        p['aE'] = 310
        p['bE'] = 125
        p['dE'] = 0.16

        p['aI'] = 615 
        p['bI'] = 177 
        p['dI'] = 0.087

        p['T'] = 7.      # [s]
        p['dt'] = 0.002  # [s]
        p['Tstim'] = 2.  # [s]
        p['Tdur']  = 1.5 # [s]

        p['sigma'] = 0.2       # nA
        p['tauN'] = 0.100      # slow excitatory synaptic time constant
        p['tauG'] = 0.005       # [s] inhibitory synaptic time constant
        p['tauA'] = 0.002        # [s]

        p['Jampa_ext'] = 5.2e-4 # nA
        p['mu'] = 38
        p['coh'] = 40
        p['t_pulse'] = 0.05 # s
        p['n_pulse'] = 16
        p['P_pulse'] = 1. # s

        p['betaSOM'] = 90
        p['I_SOM_rh'] = 0.04    # nA rheobase SOM
        p['I_SOM_bg'] = 0.15    # baseline SOM input current

        p['JVIP'] = -0.001    # VIP to SOM
        p['JSOM'] = -0.0002  # SOM to Exc
        p['JGRIKPV']  = 0.0053  

        p['gamma'] = 0.641   # firing rate to NMDA activation multiplier

        p['JNE'] = 0.42345756
        p['JNI'] = 0.57431432
        p['JGE'] = -0.46992407
        p['JGI'] = -0.64211589

        p['I0E'] = 0.77071143
        p['I0I'] = 1.02669666
        
        # parameters for dendrite
        p['bg'] = 5.56
        p['k']  = 10.54   # nS
        p['g']  = 6.54   # nS
        p['V0'] = 0.78   # mV
        p['gLD'] = 4.0   # nS
        p['EL'] = -70.   # mV
        p['Gc'] = 96. # nS           # dendrite to soma coupling
        p['gD'] = 25.7   # 1/(mV)  # current to conductance translation
        p['E_reset'] = -70 # mV
        p['tau_G_dend'] = 0.02   # ms (dendritic GABA time constant)
        p['g_G']        = 4.0    # nS (max GABA conductance)

        p['See'] = 0.32
        p['Sei'] = 0.25
        p['Sie'] = 0
        p['Sii'] = 0

        p['Ns'] = 2
        p['w_e'] = 1
        p['w_i'] = 1

        return p
    
    def timed_stochastic_pulse(self, seed):
        seq_1 = np.zeros(self.NT)
        seq_2 = np.zeros(self.NT)
        rng = np.random.default_rng(seed)
        random_set1 = rng.random(self.p['n_pulse'])
        random_set2 = rng.random(self.p['n_pulse'])

        pulse_sequence = np.where(random_set1 < self.p['P_pulse'], random_set2, 0)
        for i, pulse in enumerate(pulse_sequence):
            i_start = int(self.p['Tstim']/self.dt) + i*int((self.p['Tdur']/self.p['n_pulse'])/self.dt)
            if pulse == 0:
                continue
            elif pulse < self.p['coh']/100:
                seq_1[i_start:i_start+int(self.p['t_pulse']/self.dt)] = 1
            elif pulse > self.p['coh']/100:
                seq_2[i_start:i_start+int(self.p['t_pulse']/self.dt)] = 1
        return seq_1, seq_2
    
    def I_DS(self, Idend, rSOM):
        gE = self.p['gD'] * Idend
        gI = rSOM*self.p['tau_G_dend']*self.p['g_G']

        g12 = self.p['bg']*(self.p['gLD'] + gI)
        beta = self.p['k']*np.exp(gI/self.p['g'])
        VD = 30*(1 + np.tanh((gE - g12)/beta)) + self.p['V0'] + self.p['EL']
        return VD, self.p['Gc']*(VD - self.p['E_reset'])*1e-3
    
    def adjacency(self):
        # calculate and return the input matrix
        p = self.p
        Ns = p['Ns']

        what_ee = Ns*p['w_e']/(Ns + p['See']*(2 - Ns)) 
        what_ei = Ns*p['w_e']/(Ns + p['Sei']*(2 - Ns)) 
        what_ie = Ns*p['w_i']/(Ns + p['Sie']*(2 - Ns)) 
        what_ii = Ns*p['w_i']/(Ns + p['Sii']*(2 - Ns)) 

        wpee = what_ee + p['See']*what_ee 
        wmee = what_ee - p['See']*what_ee 
        wpei = what_ei + p['Sei']*what_ei 
        wmei = what_ei - p['Sei']*what_ei 
        wpie = what_ie + p['Sie']*what_ie 
        wmie = what_ie - p['Sie']*what_ie 
        wpii = what_ii + p['Sii']*what_ii 
        wmii = what_ii - p['Sii']*what_ii 

        A = np.zeros([4,4]) 

        A[0,:] = [wpee*p['JNE'],wmee*p['JNE'],wpie*p['JGE'],wmie*p['JGE']]  #inputs to E1
        A[1,:] = [wmee*p['JNE'],wpee*p['JNE'],wmie*p['JGE'],wpie*p['JGE']]  #inputs to E2
        A[2,:] = [wpei*p['JNI'],wmei*p['JNI'],wpii*p['JGI'],wmii*p['JGI']]  #inputs to I1
        A[3,:] = [wmei*p['JNI'],wpei*p['JNI'],wmii*p['JGI'],wpii*p['JGI']]  #inputs to I2

        return A
    
    def plot_stim(self, seed):
        pulse_1, pulse_2 = self.timed_stochastic_pulse(seed)
        plt.figure(figsize=(3,0.3), dpi=150)
        stim = (pulse_1-pulse_2)
        #print(stim[int(self.p['Tstim']/self.p['dt']):int((self.p['Tstim']+self.p['Tdur'])/self.p['dt'])])
        cmap = matplotlib.colors.ListedColormap(['#545a5e', '#FFFFFF', '#f434e7'])
        plt.imshow(stim[np.newaxis,:], cmap=cmap, aspect='auto')
        plt.axis('off')
    
    def run(self, seed=1234):
        def F(I, a, b, d):
            return (a*I - b)/(1 - np.exp(-d*(a*I - b)))
        
        NT = self.NT
        dt = self.dt
        A = self.A

        gwn = np.random.randn(4, NT)
        
        Ieta = np.zeros((4, NT+1))

        S = np.zeros((4, NT+1))
        I = np.zeros((4, NT))
        I_dend = np.zeros((2, NT))
        VD = np.zeros((2,NT))
        IDtoS = np.zeros((2,NT))
        gE_arr = np.zeros((2,NT))
        S[0:2, 0] = (np.random.rand(2)*0.05 + 0.0)
        S[2:4, 0] = (np.random.rand(2)*0.05 + 0.275)
        
        r = np.zeros((4, NT))

        # firing rates for interneurons involved in disinhibition
        # 0 VIP1; 1 VIP2; 2 SOM1; 3 SOM2;
        r_vs = np.zeros((4, NT))
        
        pulses_1, pulses_2 = self.timed_stochastic_pulse(int(seed))
        Istim1 = self.p['Jampa_ext']*self.p['mu']*pulses_1
        Istim2 = self.p['Jampa_ext']*self.p['mu']*pulses_2

        for t, time in enumerate(self.timerange): #Loop through time for a trial

            #---- Stimulus------------------------------------------------------
            # Istim1  = ((self.p['Tstim']/dt < t) & (t<(self.p['Tstim']+self.p['Tdur'])/dt)) * (self.p['Jampa_ext']*self.p['mu']*(1+self.p['coh']/100)) # To population 1
            # Istim2 = ((self.p['Tstim']/dt < t) & (t<(self.p['Tstim']+self.p['Tdur'])/dt)) * (self.p['Jampa_ext']*self.p['mu']*(1-self.p['coh']/100)) # To population 2

            # Interneuron output
            r_vs[0:2,t] = 5 * self.D2
            rate_SOM = self.p['betaSOM']*(self.p['JVIP']*r_vs[0:2, t] - self.p['I_SOM_rh'] + self.p['I_SOM_bg'])
            r_vs[2:4,t] = np.where(rate_SOM<0, 0, rate_SOM)

            # Total synaptic input
            
            I_dend[0,t] = A[0,0]*S[0,t] + A[0,1]*S[1,t] + Istim1[t]
            I_dend[1,t] = A[1,0]*S[0,t] + A[1,1]*S[1,t] + Istim2[t]

            VD[0,t], IDtoS[0,t] = self.I_DS(I_dend[0,t], r_vs[2,t])
            VD[1,t], IDtoS[1,t] = self.I_DS(I_dend[1,t], r_vs[3,t])

            I1 = IDtoS[0,t] + A[0,2]*S[2,t] + A[0,3]*S[3,t] + self.I0[0] + Ieta[0,t]
            I2 = IDtoS[1,t] + A[1,2]*S[2,t] + A[1,3]*S[3,t] + self.I0[1] + Ieta[1,t]

            I3 = np.dot(A[2,:],S[:,t]) + self.I_GRIK[2] + self.I0[2] + Ieta[2,t]
            I4 = np.dot(A[3,:],S[:,t]) + self.I_GRIK[3] + self.I0[3] + Ieta[3,t]
            I[:,t] = np.array([I1,I2,I3,I4])

            # I[:,t] = np.matmul(A, s_array) + self.I0 + Istim + self.I_GRIK + I_SOM + Ieta[:,t]

            # Transfer function to get firing rate
            r[0:2, t] = F(I[0:2,t], self.p['aE'], self.p['bE'], self.p['dE'])
            r[2:4, t] = F(I[2:4,t], self.p['aI'], self.p['bI'], self.p['dI'])
        
            #---- Dynamical equations -------------------------------------------

            # Mean synaptic dynamics updating
            S[0:2, t+1] = S[0:2,t] + dt*(-S[0:2,t]/self.p['tauN'] + (1-S[0:2,t])*self.p['gamma']*r[0:2,t])
            S[2:4,t+1] = S[2:4,t] + dt*(-S[2:4,t]/self.p['tauG'] + r[2:4,t])

            # Ornstein-Uhlenbeck generation of noise in pop1 and 2
            Ieta[:, t+1] = Ieta[:,t] + (dt/self.p['tauA']) * (-Ieta[:,t]) + np.sqrt(self.p['tauA'])*self.p['sigma']*gwn[:,t]

        return {'r': r, 'S': S, 'rvs': r_vs, 'I':I,'gE':gE_arr,'Idend':I_dend,'VD':VD, 'IDtoS':IDtoS,
                't':self.timerange}
    
    def dsdt(self, s, ifstim=True):
        I_stim = np.zeros([4])
        x      = np.zeros([4])
        r_vs    = np.zeros([4])
        H      = np.zeros([4])
        dS     = np.zeros([4])

        # stim
        if ifstim:
            I_stim[0] = self.p['Jampa_ext'] * self.p['mu'] * (1+self.p['coh']/100) 
            I_stim[1] = self.p['Jampa_ext'] * self.p['mu'] * (1-self.p['coh']/100)

        # Interneuron output
        r_vs[0:2] = 5 * self.D2
        rate_SOM = self.p['betaSOM']*(self.p['JVIP']*r_vs[0:2] - self.p['I_SOM_rh'] + self.p['I_SOM_bg'])
        r_vs[2:4] = np.where(rate_SOM<0, 0, rate_SOM)
        I_SOM = np.zeros(4)
        I_SOM[0:2] = self.p['JSOM'] * r_vs[2:4]

        # input
        x[:] = np.matmul(self.A,s) + self.I0 + I_stim + self.I_GRIK + I_SOM

        # freq
        H[0:2]  = (self.p['aE']*x[0:2] - self.p['bE'])/(1 - np.exp(-self.p['dE']*(self.p['aE']*x[0:2] - self.p['bE']))) 
        H[2:4]  = (self.p['aI']*x[2:4] - self.p['bI'])/(1 - np.exp(-self.p['dI']*(self.p['aI']*x[2:4] - self.p['bI']))) 

        # s
        dS[0:2] = -1*(self.p['tauN']**-1)*s[0:2] + (1-s[0:2])*self.p['gamma']*H[0:2]
        dS[2:4] = -1*(self.p['tauG']**-1)*s[2:4] + H[2:4]

        return dS
    
    def fourvar_jac(self, S, ifstim=True):
        # jacobian
        aE = self.p['aE']
        bE = self.p['bE']
        dE = self.p['dE']
        aI = self.p['aI']
        bI = self.p['bI']
        dI = self.p['dI']

        taus = np.array([self.p['tauN'], self.p['tauN'], self.p['tauG'], self.p['tauG']])

        I_stim = np.zeros([4])
        H      = np.zeros([4])
        r_vs   = np.zeros([4])

        dHds   = np.zeros([4,4])
        dsds   = np.zeros([4,4])

        # Interneuron output
        r_vs[0:2] = 5 * self.D2
        rate_SOM = self.p['betaSOM']*(self.p['JVIP']*r_vs[0:2] - self.p['I_SOM_rh'] + self.p['I_SOM_bg'])
        r_vs[2:4] = np.where(rate_SOM<0, 0, rate_SOM)
        I_SOM = np.zeros(4)
        I_SOM[0:2] = self.p['JSOM'] * r_vs[2:4]

        if ifstim:
            I_stim[0] = self.p['Jampa_ext'] * self.p['mu'] * (1+self.p['coh']/100) 
            I_stim[1] = self.p['Jampa_ext'] * self.p['mu'] * (1-self.p['coh']/100)

        dxds = self.A
        
        x = np.matmul(self.A,S) + self.I0 + I_stim + self.I_GRIK + I_SOM
        H[0:2] = (aE*x[0:2] - bE)/(1 - np.exp(-dE*(aE*x[0:2] - bE))) 
        H[2:4] = (aI*x[2:4] - bI)/(1 - np.exp(-dI*(aI*x[2:4] - bI)))

        for i in range(4):
            dHds[0,i] = (aE*dxds[0,i]*(1 - np.exp(-dE*(aE*x[0]-bE))) - aE*dE*np.exp(-dE*(aE*x[0]-bE))*dxds[0,i]*(aE*x[0]-bE))/ ((1 - np.exp(-dE*(aE*x[0] - bE))))**2
            dHds[1,i] = (aE*dxds[1,i]*(1 - np.exp(-dE*(aE*x[1]-bE))) - aE*dE*np.exp(-dE*(aE*x[1]-bE))*dxds[1,i]*(aE*x[1]-bE))/ ((1 - np.exp(-dE*(aE*x[1] - bE))))**2
            dHds[2,i] = (aI*dxds[2,i]*(1 - np.exp(-dI*(aI*x[2]-bI))) - aI*dI*np.exp(-dI*(aI*x[2]-bI))*dxds[2,i]*(aI*x[2]-bI))/ ((1 - np.exp(-dI*(aI*x[2] - bI))))**2
            dHds[3,i] = (aI*dxds[3,i]*(1 - np.exp(-dI*(aI*x[3]-bI))) - aI*dI*np.exp(-dI*(aI*x[3]-bI))*dxds[3,i]*(aI*x[3]-bI))/ ((1 - np.exp(-dI*(aI*x[3] - bI))))**2

        dsds[0,0] = (-1/taus[0]) + self.p['gamma']*(-H[0] + dHds[0,0]*(1-S[0]))
        dsds[1,1] = (-1/taus[1]) + self.p['gamma']*(-H[1] + dHds[1,1]*(1-S[1]))
        dsds[2,2] = (-1/taus[2]) + dHds[2,2]
        dsds[3,3] = (-1/taus[3]) + dHds[3,3]

        for i in [0,1]:
            for j in range(4):
                if i != j:
                    dsds[i,j] = self.p['gamma']*(1-S[i])*dHds[i,j]
        for i in [2,3]:
            for j in range(4):
                if i != j:
                    dsds[i,j] = dHds[i,j]
        return dsds

    def get_fixed_points(self, ifstim):
        s = np.arange(0,1,0.1) # trial s values
        
        pts = []
        ddS  = []
        tol  = 1e-6     # tolerance
        stol = 1e-6

        for i in range(len(s)):
            for j in range(len(s)):
                # solve for solutions where dsdt = 0
                # Requires the corresponding jacobian.
                
                sol = optimize.fsolve(lambda s, ifstim: self.dsdt(s, ifstim), np.array([s[i],0.5,0.5,s[j]]), args=(ifstim), fprime=lambda s, ifstim: self.fourvar_jac(s, ifstim)) # returns the roots
                sol_ds = np.sqrt(np.sum(self.dsdt(sol, ifstim)**2)) # returns the magnitude of all the roots(?)

                if sol_ds < tol:
                    if (len(pts)>0):
                        if (np.any(np.sqrt(np.sum((pts-sol)**2,axis=1))<stol)):
                            for q,w in enumerate(np.where(np.sqrt(np.sum((pts-sol)**2,axis=1))<stol)[0]):
                                if sol_ds < ddS[w]:
                                    pts[w] = sol 
                                    ddS[w] = sol_ds
                        else:
                            pts.append(sol)
                            ddS.append(sol_ds)
                    else:
                        pts.append(sol)
                        ddS.append(sol_ds)

                sol = optimize.fsolve(lambda s, ifstim: self.dsdt(s, ifstim), np.array([0.5,s[i],s[j],0.5]), args=(ifstim), fprime=lambda s, ifstim: self.fourvar_jac(s, ifstim))
                sol_ds = np.sqrt(np.sum(self.dsdt(sol, ifstim)**2))
                
                if sol_ds < tol:
                    if (len(pts)>0):
                        if (np.any(np.sqrt(np.sum((pts-sol)**2,axis=1))<stol)):
                            for q,w in enumerate(np.where(np.sqrt(np.sum((pts-sol)**2,axis=1))<stol)[0]):
                                if sol_ds < ddS[w]:
                                    pts[w] = sol 
                                    ddS[w] = sol_ds
                        else:
                            pts.append(sol)
                            ddS.append(sol_ds)
                    else:
                        pts.append(sol)
                        ddS.append(sol_ds)
        pts   = np.array(pts)
        fps   = pts[np.isfinite(pts).all(axis=1),:]
        fps   = fps[np.argsort(fps[:,0]),:]
        stab  = np.zeros([fps.shape[0]])
        evals = [] 
        for i in range(fps.shape[0]):
            jac = self.fourvar_jac(fps[i], ifstim)
            if np.isfinite(jac).all():
                egs = np.linalg.eig(jac)
                evals.append(egs[0])
                if (egs[0]<0).all():
                    stab[i] = 1
            else:
                stab[i] = np.nan
                evals.append(np.array([np.nan]))
                
        return {'points':fps,'stable':stab,'eiganvalues':evals, 'dS':ddS}  

