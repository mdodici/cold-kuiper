import os
import sys
import getopt
import numpy as np

######################
#
#  Setting variable params
#
######################


# Defaults of possible argument parameters
#    Physical setup
eMax = 0.20      # maximum eccentricity for grid
alpha = 11/6     # size distribution power (Dohnanyi Law, 1969)
q = 0

#    Numbers of gridpoints
gps = 80    # gridpoints in sizd
gpe = 100   # gridpoints in ecc. 

# this script takes the following command line arguments:
#    -e --emax (default 0.25)
#    -q --break (default 0)
#    -a --alpha (default 11/6)
#    -ge --gpe (default 100)
#    -gs --gps (default 80)

# set arguments
argv = sys.argv  
# set help message
arg_help = '{0} -e <emax> -q <break> -a <alpha> -ge <gpe> -gs <gps>'.format(argv[0])

try:  # see if arguments are real, accounted for
    opts, args = getopt.getopt(argv[1:], "h:e:q:a:ge:gs",
                                ["help","eMax=","q=","alpha=","gpe=","gps="])
except:  # if not, print help message and break
    print(arg_help)
    sys.exit(2)

for opt, arg in opts:  # go through args, setting the appropriate parameter using each 
    if opt in ("-h","--help"):
        print(arg_help)
        sys.exit(2)
    elif opt in ("-e","--emax"):
        eMax = float(arg)
    elif opt in ("-q","--break"):
        q = int(arg)
    elif opt in ("-a","--alpha"):
        alpha = float(arg)
    elif opt in ("-ge","--gpe"):
        gpe = int(arg)
    elif opt in ("-gs","--gps"):
        gps = int(arg)


######################
#
#  Setting other parameters/arrays
#
######################


# Physical parameters
G = 4*np.pi**2           # grav constant in au^3 solar mass^-1 yr^-2
sma = 45                 # semimajor axis in au
dlt = 4                  # delta_a, width of debris disk in au
per = np.sqrt(sma**3)    # period in yr
frq = 1/per              # frequency in yr^-1
rho = 1                  # density of objects in g cm^-3
r_s = 7e10               # solar radius in cm
sma_cm = sma*1.5e13      # semimajor axis in cm
dlt_cm = dlt*1.5e13      # delta_a in m
alp_sun = r_s/(sma_cm)   # alpha, defined in Goldreich et al as solar radius / SMA
v_kep = np.sqrt(G/sma) * (1.5e13/3.15e7)     # keplerian velocity in cm s^-1

# Breaking energy law constants
A_stg = 5e7       # strength constant A in erg g^-1
B_stg = 3.3e-8    # strength constant B
s0 = 0            # size_0 of strength eq (1 cm in log cm)
alp_stg = -0.3    # strength constant alpha
bet_stg = 2       # strength constant beta

# Parameters of grids
sMin = -4         # log sizes [cm]
sMax = 8

eMin = 0          # eccentricities 

num_colls = gps*gpe*gpe    # number of total possible collisions

# Grid creation
size = np.linspace(sMin,sMax,gps)      
eccs = np.linspace(eMin,eMax,gpe)  

# Lookup tables
R = np.zeros((gps,gpe,gpe))               # one-to-one collision rates of each collision
G = np.zeros((gps,gpe,gpe))               # bullet size for each collision (s_min(s_1,e_1,e_2))
e_succ = np.zeros((gps,gpe,gpe))          # eccentricity of results of each collision   
successor = np.zeros((gps,gpe,gpe,gps))   # number of bodies of each size from each collision


######################
#
#  Defining functions
#
######################


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

def qstar(s1):
    """
    Energy required to break a body of size s1

    In: s1 -- float; log size cm

    Out: Q -- float; energy to break body in erg g^-1
    """
    term1 = A_stg * np.power(10,(s1-s0)*alp_stg)
    term2 = B_stg * 2.5 * 10**((s1-s0)*bet_stg)
    Q = term1 + term2
    return Q

qsize = qstar(size)

def s2_break(s1, e1, e2):
    """
    Size of smallest possible catastrophic bullet for s1, e1, e2 based on given strength law

    In: s1 -- float; log size of body cm
        e1 -- float; eccentricity of body 
        e2 -- float; eccentricity of bullet

    Out: logs2 -- float; log size of bullet in cm
    """
    Q_ = qstar(s1)               # breaking energy   
    omega = frq / (3.154e7)      # frq in 1/s
    vcol = vcollision(e1,e2)     # collisional velocity
    denm = vcol**2 - 2*Q_        # denominator of s2 factor
    s_ = 2*Q_ / denm             # s2 factor in cm
    
    s_ = s_.clip(min=0)                   # s2 factor cannot be less than 0
    size2 = 10**(s1) * np.power(s_,1/3)   # multiplying factor by s1
    logs2 = np.log10(size2)               # taking log of s2
    return logs2

def vcollision(e1,e2):
    """
    Typical relative velocity of bodies with eccentricity e1 and e2

    In: e1, e2 -- floats; eccentricities

    Out: vcoll -- float; typical collisional velocity in cm s^-1
    """
    vcoll = v_kep * np.sqrt(e1**2 + e2**2)
    return vcoll

def ecc_com(s1,s2,e1,e2):
    """
    Typical center-of-mass eccentricity of two bodies in collision -- i.e. the eccentricity their successors will take on

    In: s1, s2 -- floats; log sizes [cm] of body and bullet in collision
        e1, e2 -- floats; eccentricities of body and bullet
    
    Out: ecc -- float; center-of-mass eccentricity of collision
    """
    s1 = 10**s1; s2 = 10**s2
    numer = np.sqrt(s1**6 * e1**2 + s2**6 * e2**2)
    denom = s1**3 + s2**3
    
    ecc = numer/denom
    return ecc

def rate(s1,s2):
    """
    One-to-one collision rate of two bodies with log size s1 and s2

    In: s1, s2 -- float; log sizes of colliding bodies in m
         
    Out: lf -- 1D array; log rates of collisions for all bodies
    """
    numer = np.pi * (10**s1 + 10**s2)**2
    denom = 2 * np.pi * sma_cm * per * dlt_cm
    f = numer / denom
    lf = np.log10(f)
    return lf

def size_succ(s1,s2,e1,e2):
    """
    Calculating size distribution of successors of a given collision based on work in ?????

    In: s1, e1 -- floats; log size [cm] of body, eccentricity of body
        s2, e2 -- len=N array; log sizes [cm] of bullets, eccentricity of bullets
        
    Out: number -- N-by-gps array; log number distribution across all sizes for each collision 
    """
    m1 = (4/3) * np.pi * rho * 10**(s1*3)
    m2 = (4/3) * np.pi * rho * 10**(s2*3)

    vcol = vcollision(e1,e2)           # len=N array; collision velos
    Q_ = qstar(s1)                     # float; energy to break star

    C = 0.5 * (m2/m1) * (vcol**2/Q_)   # array; constant from ?????
    m_X = 0.5 * m1 * C**(-1.24)        # array; maximum succsessor MASS [g]

    s_X = np.log10( np.power( m_X * 3/(4*np.pi*rho) , 1/3) )     # array; maximum successor log SIZE [cm]

    m_redist = (m1 + m2)/5.972e27      # mass to be redistributed [g]
    
    number = np.zeros((len(e2),gps))   # empty N-gps array
    for i in range(len(e2)):           # for each collision

        s_X_bin = ((np.abs(s_X[i] - size)).argmin()).astype(int)     # maximum size bin based on s_X

        dist = size_dist(size[:s_X_bin+1], m_redist[i], alpha)       # number distribution from minimum size bin up to s_X_bin
        extra = np.full(gps - s_X_bin - 1, -np.inf)                  # for excess size bins, want log N = -inf

        number[i] = np.append(dist, extra)                           # append these to get log N for each size bin 
    return number

def generate_tables():
    ##################
    #
    #  Filling things
    #
    ##################

    for n in range(gps):
        s1 = size[n]      # focus on a single target body size
        for i in range(gpe):     
            e1 = eccs[i]        # focus on a single target eccentricity
            e2 = eccs           # but consider the whole range of bullet eccs at once

            G[n,i,:] = s2_break(s1,e1,e2)         # fill G with the size of bullet with e2 that will break s1, e1 
            R[n,i,:] = rate(s1,G[n,i,:])          # fill R with the one-to-one rate of collisions between body and bullet
            e_succ[n,i,:] = ecc_com(s1,G[n,i,:],e1,e2)           # fill e_succ with typical COM ecc. for this collision
            successor[n,i,:] = size_succ(s1,G[n,i,:],e1,e2)      # fill successor with log # of fragments in each size bin

            
    #####################
    #
    # Flattening/indexes
    #
    #####################


    succdist_flat = successor.reshape((num_colls,gps))    # reshape successor number dist to num_colls-by-gps array
    G_flat = G.transpose(0,2,1).ravel()                   # flatten G to 1D array of len=num_colls (note transposition to keep relations between cells)
    R_flat = R.transpose(0,2,1).ravel()                   # flatten R to 1D array of len=num_colls (note transposition to keep relations between cells)
    e_succ_flat = e_succ.ravel()                          # flatten successor eccentricity array (note transposition is NOT necessary here. Don't know why :))

    esucc_idx = np.zeros_like(e_succ_flat)                # empty array len=num_colls
    for i in range(len(e_succ_flat)):                     # for each collision
        esucc_idx[i] = ((np.abs(eccs - e_succ_flat[i])).argmin()).astype(int)     # record index of successor eccentricity bin
        
    idx = np.zeros((num_colls,4))     # empty array for storing indices of body/bullet for each collision

    s1_idx, e1_idx, e2_idx = np.unravel_index(range(num_colls),(gps,gpe,gpe))    # indices of s1, e1, e2 for each collision

    s2_idx = np.zeros(num_colls) 
    for i in range(num_colls):
        s2_idx[i] = (np.abs(size - G_flat[i])).argmin()    # indices of s2 for each collision based on closest bin    
        
    # filling idx array with the index arrays we just made
    idx[:,0] = s1_idx 
    idx[:,1] = e1_idx
    idx[:,2] = s2_idx
    idx[:,3] = e2_idx

    idx = idx.astype('int')   # make sure they're all integers

    R_flat[idx[:,0] < idx[:,2]] = -np.inf                    # say that collisions where s2 > s1 don't exist
    R_flat[G_flat < -4] = -np.inf                            # say that collisions where s2 < min size don't exist

    RNS = succdist_flat + R_flat.reshape((num_colls,1))      # create RNS -- lookup table w/ rate at which each collision adds new numbers to each size bin
    RNS = np.nan_to_num(RNS, nan=-np.inf, neginf=-np.inf)    # make sure there are no nans in RNS lookup table


    # set up folder to which to save results
    cwd = os.getcwd()
    lookupdir = os.path.join(cwd,r'lookup')
    if not os.path.exists(lookupdir):
        os.makedirs(lookupdir)

    np.save('%s/idx.npy' %lookupdir, idx)
    np.save('%s/RNS.npy' %lookupdir, RNS)
    np.save('%s/R_flat.npy' %lookupdir, R_flat)
    np.save('%s/esucc_idx.npy' %lookupdir, esucc_idx)

    print("Tables generated")
    return

generate_tables()