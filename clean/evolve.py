import os
import sys
import getopt
import time
import numpy as np


######################
#
#  Setting variable params
#       These are more commonly changed and can be set with command-line arguments
#
######################


# Defaults of possible argument parameters
#    Physical setup
e0 = 0.1         # center of eccentricity kernel
e0_std = 0.001   # standard deviation of eccentricity kernel
m0 = 10          # initial mass in scatter belt in Earth masses
alpha = 11/6     # size distribution power (Dohnanyi Law, 1969)

#    Integration
t_tot = 1e5             # total time of integration
dt_min = 10             # minimum timestep length
rate_cutoff = -10       # log_10(minimum collision rate tracked)
badr_cutoff = 99.5      # percentage of cells allowed to be underresolved

#    Numbers of gridpoints
gps = 80    # gridpoints in sizd
gpe = 100   # gridpoints in ecc. 

#    Condition to track at timestep or at set times
track_at_step = False     # if True, tracking will be done when N_i, N_o are updated; if False, at set, evenly-spaced times
num_track = 1000          # number of times at which to track N, N_i, N_o

# this script takes the following command line arguments:
#    -e --e0 (float; default 0.1; can also be a tuple if multiple ecc. peaks desired)
#    -m --m0 (float; default 1e2)
#    -a --alpha (float; default 11/6)
#    -t --t_tot (float; default 1e6)
#    -d --dt_min (float; default 10)
#    -p --badr_cutoff (float; default 99.5)
#    -r --rate_cutoff (float; default -10)
#    -ge --gpe (int; default 100)
#    -gs --gps (int; default 80)
#    -tr --track_at_step (bool; default False)
#    -tn --track_number (int; default 1000)

# set arguments
argv = sys.argv  
# set help message
arg_help = '{0} -e <e0> -es <e0_std> -t <time> -d <dt_min> -a <alpha> -m <m0> -p <badr_cutoff> -r <rate_cutoff> -ge <gpe> -gs <gps> -tr <track_at_step> -tn <track_number>'.format(argv[0])

try:  # see if arguments are real, accounted for
    opts, args = getopt.getopt(argv[1:], "h:e:es:t:d:a:m:p:r:ge:gs:tr:tn",
                                ["help","e0=","e0_std=","t_tot=","dt_min=","alpha=","m0=","badr_cutoff=","rate_cutoff=","gpe=","gps=","track_at_step=","track_number="])
except:  # if not, print help message and break
    print(arg_help)
    sys.exit(2)

for opt, arg in opts:  # go through args, setting the appropriate parameter using each 
    if opt in ("-h","--help"):
        print(arg_help)
        sys.exit(2)
    elif opt in ("-e","--e0"):
        if ',' in arg:
            e0 = tuple(map(float, arg.split(',')))
        else:
            e0 = float(arg)
    elif opt in ("-t","--t_tot"):
        t_tot = float(arg)
    elif opt in ("-d","--dt_min"):
        dt_min = float(arg)
    elif opt in ("-a","--alpha"):
        alpha = float(arg)
    elif opt in ("-m","--m0"):
        m0 = float(arg)
    elif opt in ("-p","--badr_cutoff"):
        badr_cutoff = float(arg)
    elif opt in ("-r","--rate_cutoff"):
        rate_cutoff = float(arg)
    elif opt in ("-ge","--gpe"):
        gpe = int(arg)
    elif opt in ("-gs","--gps"):
        gps = int(arg)
    elif opt in ("-tr","--track_at_step"):
        track_at_step = bool(arg)
    elif opt in ("-tn","--track_number"):
        num_track = int(arg)


######################
#
#  Setting other parameters/arrays
#       These should be less-frequently changed
#
######################

# Parameters of grids
sMin = -4         # log sizes [cm]
sMax = 8

eMin = 0          # eccentricities 
if isinstance(e0,tuple) == True:
    eMax = max(e0)+.05
else:
    eMax = e0+.05

num_colls = gps*gpe*gpe    # number of total possible collisions

# Grid creation
size = np.linspace(sMin,sMax,gps)      
eccs = np.linspace(eMin,eMax,gpe)  

asp_rat = (sMax-sMin)/(eMax-eMin)

# Arrays of integration
ts_max = int(1e5)           # maximum number of timesteps
dtp = np.zeros(ts_max)      # array for tracking proposed timestep length
dt = np.zeros(ts_max)       # array for tracking actual timestep length
ts = np.zeros(ts_max)       # array for tracking actual time
fu = np.zeros(ts_max)       # array for tracking fraction of bins underresolved

nrem_all = np.zeros((ts_max,num_colls))   # tracking the rate of each collision at all times

if track_at_step == True:
    N = np.zeros((ts_max,gpe,gps))            # tracking number in each cell
    N_i = np.zeros((ts_max,gpe,gps))          # tracking number going into each cell
    N_o = np.zeros((ts_max,gpe,gps))          # tracking number going out of each cell

else:
    ts_track = np.logspace(0,np.log10(t_tot),num_track)
    N_track = np.zeros((num_track,gpe,gps))
    Nitrack = np.zeros((num_track,gpe,gps))
    Notrack = np.zeros((num_track,gpe,gps))

rho = 1

def logamplitude(sizes,mtot,a):    
    """
    Calculating the amplitude of the number distribution

    In: sizes -- array; log sizes in cm of bins to distribute into
        mtot -- float; total mass in earth masses
        a -- float; power law parameter alpha

    Out: amp -- float; amplitude of number distribution
    """
    scale = mtot*5.972e27 * (2-a) / ((4/3)*np.pi*rho)
    size_part = np.power(10,sizes[-1]*(6-3*a)) - np.power(10,sizes[0]*(6-3*a))
    
    amp = np.log10(scale) - np.log10(size_part)
    return amp

def size_dist(s,mt,a):
    """
    Calculate the number of bodies to be distributed across a given range of size bins
       - note: if only one bin given (in this system, *always* only the smallest bin), mass **removed**

    In: s -- array; log sizes in cm
        mt -- float; total mass in earth masses
        a -- float; power law parameter alpha
        
    Out: s_scale -- array; log number of bodies in each size bin
    """
    if len(s) > 1:
        amp = logamplitude(s,mt,a)    
        s_scale = s*(3-3*a) + amp
    else:                              # if only one bin available (i.e. coll between two smallest-bin bodies)
        s_scale = -np.inf              #      get rid of the mass -- add nothing back to system
        
    return s_scale

def save_files(N, N_i, N_o, ts, dt, fu, ts_end, it):
    """
    Saves arrays, with some total time and iteration number

    In: depends on track_at_step...
        if track_at_step == True:
            N -- T-by-gpe-by-gps array; numbers in each cell at up to T >= it timesteps
            N_i -- ditto; incoming rate at each step 
            N_o -- ditto; incoming rate at each step
            ts -- len(T) array; time at each step
        else:
            N -- num_track-by-gpe-by-gps array; numbers in each cell at num_track tracking steps
            N_i -- ditto; incoming rate
            N_o -- ditto; outgoing rate
            ts -- len(num_track) array; time at each tracking step

            (note: if KeyboardInterrupt broke up integration, many of the steps will have N, N_i, N_o = 0)
        ts_end -- float; total time of integration
        it -- int; number of timesteps taken 

    During: uses pandas to save the arrays of interest as dfs

    Out: none
    """
    if track_at_step == True:
        N = N[:it+1]
        N_i = N_i[:it+1]
        N_o = N_o[:it+1]
        ts = ts[:it+1] 

    fu = fu[:it+1]
    dt = dt[:it+1]
    
    # set up folder to which to save results
    cwd = os.getcwd()
    rwd = os.path.join(cwd,r'results')
    newdir = os.path.join(rwd, 
                        r'e0-%1.2f_t-%.1fMyr_m0-%.0fMe_p-%i_r%i' %(e0,ts_end/1e6,m0,badr_cutoff,rate_cutoff))
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    # save results as npy files

    if track_at_step == True:
        len_of_tracked = it+1
    else:
        len_of_tracked = num_track
        
    np.save('%s/N.npy' %newdir, N.reshape(len_of_tracked,gpe*gps))
    np.save('%s/N_i.csv', N_i.reshape(len_of_tracked,gpe*gps))
    np.save('%s/N_o.csv', N_o.reshape(len_of_tracked,gpe*gps))
    np.save('%s/ts.csv' %newdir, ts)
    np.save('%s/dt.csv' %newdir, dt)
    np.save('%s/fu.csv' %newdir, fu)
    #np.save('%s/nrem.csv' %newdir, nrem_all)

    return

##################
#
#  Setting initial distribution
#
##################

# set initial number distribution in size bins
s_num = size_dist(size,m0,alpha)

# set initial relative distribution in ecc bins
if isinstance(e0,float) == True:   # if there's just one peak,
    # create unnormed number distribution in eccentricity as a gaussian around that peak
    e_num_unnorm = 1/np.sqrt(2*np.pi*e0_std**2) * np.exp(-0.5*(eccs-e0)**2/e0_std**2)
    # normalize distribution and switch to log n
    e_num = np.log10(e_num_unnorm / np.sum(e_num_unnorm))
else:
    # if there's more than one peak eccentricity
    e_num_unnorm = np.zeros(gpe)  # created empty array for number distribution
    for i in range(len(e0)):      # for each peak, add gaussian based on that peak
        e_num_unnorm += 1/np.sqrt(2*np.pi*e0_std**2) * np.exp(-0.5*(eccs-e0[i])**2/e0_std**2)

    # add some baseline number of particles between the gaussians (default 0)    
    e_num_unnorm[np.argmin(abs(min(e0) - eccs)):np.argmin(abs(max(e0) - eccs))] += 0
    # normalize distribution and switch to log n
    e_num = np.log10(e_num_unnorm / np.sum(e_num_unnorm))

# convolve these distributions into 2D initial number distribution 
N_start = np.log10(np.outer(10**e_num,10**s_num))

        
#####################
#
# Importing lookup tables
#
#####################


# if there is no directory with lookup tables, we need to generate them
cwd = os.getcwd()
lookupdir = os.path.join(cwd,r'lookup')
if not os.path.exists(lookupdir):
    print("Generating lookup tables... this can be done beforehand by running lookup.py")
    import lookup
    lookup.generate_tables()

# if pre-existing (or once generated) can import CSVs
print("Importing lookup tables...")

idx = np.load('lookup/idx.npy')
RNS = np.load('lookup/RNS.npy')
R_flat = np.load('lookup/R_flat.npy')
esucc_idx = np.load('lookup/esucc_idx.npy')


##############
#
# Integration
#   & Saving
#
##############


if track_at_step == True:    # if we're tracking at every integration (integ.) step
    N[0] = N_start           #     set the first integ. step based on the calculated starting distribution 
else:                        # else
    N_track[0] = N_start     #     set the first tracking step based on the same

N_prev = np.zeros((gpe,gps))
it = 0
print("Starting integration!")

try:  
    
    # try to do the full integration and save files
    # if taking too long, can KeyboardInterrupt -- will save files wherever you are in integration

    while it < ts_max - 1:
        
        #####
        # Update arrays
        #####
        if np.count_nonzero(np.isnan(N_prev)) > 0:     # if there are nans in the 2D number distribution, break
            print('nans in N[%.i]:'%it,np.count_nonzero(np.isnan(N[it])))
            break
        
        if (it - 1) % 10 == 0:    # start a timer if iteration number ends in 1
            start_time = time.time()   # this will run over next ten steps
        
        if track_at_step == True:   # if we are tracking at every integ. step
            N_now = N[it]           #     set N dist for this step based on recorded previous step
        elif it == 0:               # else if it is the first step
            N_now = N_start         #     set N dist for this step based on calculated starting distribution
        else:                       # else if we are beyond the first step
            N_now = N_prev          #     set N dist for this step based on (un-recorded) previous step

        N_incoming = np.zeros((gpe,gps))           # create empty 2D array for incoming rates for this step
        N_outgoing = np.zeros((gpe,gps))           # create empty 2D array for incoming rates for this step
            
        n1_all = N_now[idx[:,1],idx[:,0]]          # number of bodies for every collision
        n2_all = N_now[idx[:,3],idx[:,2]]          # number of bullets for every collision
        nrem_all[it] = R_flat + (n1_all + n2_all)  # current rate of each collision

        #####
        # Fill incoming/outgoing rate arrays
        #####

        for i in np.argwhere(nrem_all[it] > rate_cutoff):  # for each collision,    
            i=i[0]
        
            s1_id, e1_id = idx[i,0], idx[i,1]        # indices of 2D bin of body
            s2_id, e2_id = idx[i,2], idx[i,3]        # indices of 2D bin of bullet

            n1 = N_now[e1_id,s1_id]                  # number of bodies
            n2 = N_now[e2_id,s2_id]                  # number of bullets

            nins_collision = RNS[i] + (n1 + n2)      # rate of insertion of bodies into each size bin from this collision
            nrem_collision = R_flat[i] + (n1 + n2)   # rate of occurrence of this collision       

            j = esucc_idx[i].astype(int)             # index of successor e bin
            N_incoming[j,:] += 10**nins_collision    # add successor number rate to successor e row

            N_outgoing[e1_id,s1_id] += 10**nrem_collision   # add outflow rate to 2D bin of body
            N_outgoing[e2_id,s2_id] += 10**nrem_collision   # add outflow rate to 2D bin of bullet
        
        #####
        # Set timestep
        #####
        
        net_loss_bins = np.argwhere(N_incoming < N_outgoing)     
        rateN_ratio = np.log10(np.abs(N_incoming - N_outgoing)) - N_now   # calculate log(rate / number) for each 2D bin
        rateN_ratio_cut = rateN_ratio
        # COMMENT OUT NEXT LINE TO CONSIDER ALL BINS for dt_proposal
        #rateN_ratio_cut = rateN_ratio[net_loss_bins]
        rateN_ratio_cut[np.abs(rateN_ratio_cut) == np.inf] = -10          # set infinities to some value 
        rateN_cut = np.nanpercentile(rateN_ratio_cut,badr_cutoff)         # we only want some percentage of cells to be underresolved
        dt_proposal = 10**(-rateN_cut)                                    # propose a timestep based on this cutoff
        
        dt_it = max(dt_min,dt_proposal)                                   # choose whichever timestep is bigger: proposal or floor
        
        #####
        # Integrate forward
        #####
        
        N_current = 10**N_now                                             # go from log number to number
        N_next = (N_current + (N_incoming - N_outgoing) * dt_it)          # add and subtract numbers with timestep
        N_next = N_next.clip(min=0)                                       # cannot have < 0 bodies in any bin
        N_next[np.isnan(N_next)] = 0                                      # set any nans to 0
        
        #####
        # Update arrays
        #####

        #dtp[it] = dt_proposal           # record timestep proposed at this step
        dt[it] = dt_it                  # record timestep used at this step
        ts[it+1] = ts[it] + dt_it       # record time at start of next step
        fu[it] = np.count_nonzero(rateN_ratio > rateN_cut)/(gpe*gps)

        # if we want to track evolution at every integration step, do so (note that this has to be set at start of script)
        if track_at_step == True:
            N_i[it] = np.log10(N_incoming)  # record incoming rates at this step
            N_o[it] = np.log10(N_outgoing)  # record outgoing rates at this step
            N[it+1] = np.log10(N_next)      # record log numbers at start of next step
        
        # if we want to track evolution at specific tracking steps rather than at each integration step, things are trickier
        else:
            tprev = ts[it-1]            # time at previous integ. step
            tnext = ts[it]              # time at this integ. step

            # determine which tracking steps fall between last integ. step and this one
            tracked_ts_instep = np.argwhere((ts_track > tprev) & (ts_track <= tnext))
            # for each of them, 
            for i in tracked_ts_instep:
                t_since_int = ts_track[i] - tprev                                     # dt from last integ. step to tracking step 
                N_intermed = (N_current + (N_incoming - N_outgoing) * t_since_int)    # calculate new N at tracking step
                N_intermed = N_intermed.clip(min=0)                                   # cannot have < 0 bodies in any bin
                N_intermed[np.isnan(N_intermed)] = 0                                  # set any nans to 0

                N_track[i] = np.log10(N_intermed)      # record N at tracking step
                Nitrack[i] = np.log10(N_incoming)      # record current integ. step inflow rates 
                Notrack[i] = np.log10(N_outgoing)      # record current integ. step outflow rates

        N_prev = np.log10(N_next)
        
        if ts[it] > t_tot:              # if this step was beyond the planned total time of integration, 
            print("Integration complete. Total t = %1.2E" %ts[it])
            break                       #    stop

        if it % 10 == 0:                # if this was first or n*10th step,
            if it == 0:                 #    print message for 1st step
                print("Those RuntimeWarnings were expected! First step complete.")
            else:                       #    print message for n*10th step
                step_avg_time = (time.time() - start_time)/10    # calculate average time of last 10 int steps
                print("Step",it,"complete. t = %1.2E. average step time %1.2f s" %(ts[it],step_avg_time))

        it += 1     # if ts[it] > t_tot, move to next step

    print("Saving files...")
    if track_at_step == True:
        save_files(N, N_i, N_o, ts, dt, fu, t_tot, it)
    else:
        save_files(N_track, Nitrack, Notrack, ts_track, dt, fu, t_tot, it)
    print("Done.")

except KeyboardInterrupt:

    # in case of KeyboardInterrupt, want to save what we have integrated so far

    t_tot = ts[it]
    print()
    print("KeyboardInterrupt registered. Integration ended.")
    print("Saving files...")
    if track_at_step == True:
        save_files(N, N_i, N_o, ts, dt, fu, t_tot, it)
    else:
        save_files(N_track, Nitrack, Notrack, ts_track, dt, fu, t_tot, it)
    print("Done.")