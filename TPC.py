import numpy as np 
import matplotlib.pyplot as plt 
import sys 



class TPC:
	#xenon mass: kg
	#assumes cylindrical vol: xe_r, xe_h (length and height in cm)
	def __init__(self, xe_mass, xe_r, xe_h):
		#some "global" like constants
		self.xe_density_L = 2.942 #kg/L
		self.xe_density_cm = 2.942e-3 #kg/cm^3
		rate_const_ko = 1.05e11# #1/M 1/s for field of 380 V/cm, 400 V/cm is same, +- 0.02e11
		self.tau_conv = 1/(rate_const_ko*self.xe_density_L/131) #131 is g/mol. 
		self.tau_conv = self.tau_conv*1e6*1e6 #in microseconds * ppb

		self.xe_m = xe_mass #kg
		self.xe_r = xe_r #cm
		self.xe_h = xe_h #cm
		self.xe_vol_mass = self.xe_m/self.xe_density_cm #cm^3 
		self.xe_vol_chamber = 2*np.pi*self.xe_r*self.xe_r*self.xe_h #cm^3

		#total moles of xenon in chamber
		self.n_tot = self.xe_m*1000/131 


	#get mole fraction conc of oxygen in ppb and us
	def getX(self, tau):
		return (self.tau_conv/tau)

	#get tau for mole fraction conc of o2, ppb and us
	def getTau(self, X):
		return (self.tau_conv/X)

	#return tau_c, the effective time 
	#to recirc full volume of xenon
	def get_recirc_time(self, recirc_rate):
		return self.n_tot/(recirc_rate*60/131) #hours (60 is for mins to hours), 131 molar mass



	#literally makes figures 5 from https://arxiv.org/pdf/2205.07336.pdf
	#and returns all x, y, lists. for looking at shape and dynamics 
	#for fixed periods of purification. 
	#recirc_rate: in g/min
	#eps: getter efficiency
	#ogas: outgassing rate in moles/hour
	#f: mixing coefficient
	#x0: oxygen mole fraction in units of ppb initially
	#recirc_t: time that recirculation is active (hours)
	#total_t: total time of run (so plot scales are the same) (hours)
	def get_purity_curve(self, recirc_rate, ogas, eps, f, x0, recirc_t, total_t, npoints=100):
		#calculate tau_c (total time to recirculate full mass)
		#based on the recirc rate. 
		tau_c = self.n_tot/(recirc_rate*60/131) #hours (60 is for mins to hours), 131 molar mass

		#solution to diff eq. with purifier on
		def soln_on(t):
			a = ogas/self.n_tot #mole concentration per unit time (1/hour) 
			b = eps*f/tau_c 
			c = x0 - (a/b)*1e9
			return (a/b + c*np.exp(-b*t))

		#starts with a new initial concentration
		def soln_off(t, x00):
			return (ogas/self.n_tot)*t + x00


		ts = np.array(np.linspace(0, total_t, npoints))
		#determine the index in t's where purification turns off
		off_idx = np.abs(ts - recirc_t).argmin()
		#set all values after this to 0 for efficiency
		ts_on = ts[:off_idx]
		ts_off = ts[off_idx:]

		xs_on = soln_on(ts_on)
		xs_off = soln_off(ts_off - ts_off[0], xs_on[-1])
		xs_full = np.concatenate((xs_on, xs_off))*1e9 #ppb
		taus_full = self.getTau(xs_full)
		return ts, xs_full, taus_full
	
	#a simplified version of the above for single piecewise sections
	#x0 in units of ppb
	def get_purity_snippet(self, recirc_rate, ogas, eps, f, x0, total_t, npoints=100):
		#discrete differential equation
		def dx_on(dt, tau_c, cur_x):
			a = ogas*1e9/self.n_tot #mole concentration ppb per unit time (1/hour) 
			b = eps*f/tau_c 
			return (a - b*cur_x)*dt #differential change in x

		#starts with a new initial concentration
		def dx_off(dt):
			return (ogas*1e9/self.n_tot)*dt
		
		
		dT = total_t/npoints
		ts = [0]
		xs_full = [x0]
		while True:
			if(ts[-1] >= total_t):
				break
			ts.append(ts[-1] + dT)
			x00 = xs_full[-1]
			if(recirc_rate == 0 or eps == 0):
				xs_full.append(x00 + dx_off(dT))
			else:
				#perform state machine for duty cycled purification
				tau_c = self.n_tot/(recirc_rate*60/131) #hours (60 is for mins to hours), 131 molar mass, for when on
				xs_full.append(x00 + dx_on(dT, tau_c, x00))
				
		xs_full = np.array(xs_full)
		taus_full = self.getTau(xs_full)
		ts = np.array(ts)
		return ts, xs_full, taus_full
		
	#a simplified version of the above for single piecewise sections
	#x0 in units of ppb. This is a special case as our "fast recirc"
	#mode is duty cycled due to too high pressures. period in hours, duty is on fraction
	def get_purity_snippet_duty_cycled(self, recirc_rate, ogas, eps, f, x0, total_t, period, duty, npoints=100):
		#discrete differential equation
		def dx_on(dt, tau_c, cur_x):
			a = ogas*1e9/self.n_tot #mole concentration ppb per unit time (1/hour) 
			b = eps*f/tau_c 
			return (a - b*cur_x)*dt #differential change in x

		#starts with a new initial concentration
		def dx_off(dt):
			return (ogas*1e9/self.n_tot)*dt
		

		t_on = period*duty
		t_off = period*(1 - duty)
		
		dT = total_t/npoints
		ts = [0]
		xs_full = [x0]
		on = True
		total_time_in_state = 0
		while True:
			if(ts[-1] >= total_t):
				break
			ts.append(ts[-1] + dT)
			x00 = xs_full[-1]
			if(recirc_rate == 0 or eps == 0):
				xs_full.append(x00 + dx_off(dT))
			else:
				#perform state machine for duty cycled purification
				tau_c = self.n_tot/(recirc_rate*60/131) #hours (60 is for mins to hours), 131 molar mass, for when on
				
				#see if its time to switch states
				if(on and total_time_in_state >= t_on):
					on = False
					total_time_in_state = 0
				if(on == False and total_time_in_state >= t_off):
					on = True
					total_time_in_state = 0

				if(on):
					xs_full.append(x00 + dx_on(dT, tau_c, x00))
				else:
					xs_full.append(x00 + dx_off(dT))
					
				total_time_in_state += dT

		xs_full = np.array(xs_full)
		taus_full = self.getTau(xs_full)
		ts = np.array(ts)
		return ts, xs_full, taus_full








