# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:33:01 2025

Contributors:
Slightly modified script from PicoQuant read in routine from GitHub
https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/tree/master/PTU/Python

Modified by Ilaria on Wed 21/08/2024

Modified by Xinqiu on Fri 17/10/2025, contributing the following:
    - Uncorrelated algorithm for re-aligning Macrotimes
    - Using minima of the Microtime differences as time windows for photon classification
    - Reconstruction verification through Scatter plot

"""

import time
import sys
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pickle
from scipy.signal import find_peaks
from scipy.stats import binned_statistic_2d
from scipy.io import savemat

# =============================================================================
# File information
# =============================================================================
    # ================================================= #
    # Must check: data/measurement info
    # ================================================= #
num_det = 2     # number of detectors
TDC_Sig = 0     # TDC channel of the signal, usually 0
TDC_Ref = 7     # TDC channel of the reference, usually 6
# TDC_Ref = 6
delay = 500     # theoretical delay between each channel in [ps]

# only for files with Ground Truth data
TDC_D1 = 1        # True signal from Det1
TDC_D2 = 6        # True signal from Det2
GND_True = 1      # If the GroundTrue signal is present

    # ================================================= #
    # Must check: data paths
    # ================================================= #
IRF = 0             # if the current PTU file is IRF
file_number = 430

if (IRF == 0) and (GND_True == 0):
    file = r"/MEDm_" + str(file_number)     # filename
else:
    if GND_True == 1:
        file = r"/MH160m_" + str(file_number)
    else:
        file = r"/MEDs_" + str(file_number)     # filename
        num_det = 1
    
ext = r".ptu"         # file extension
# input .ptu file path
# path =r"C:\Users\ye\Documents\TTTR\DataIn\8det"       
path = r"C:\Users\Xinqiu\Documents\Python Scripts"                                          

# path and name for the generated (exported) data file (.pckl)
file_out = file + r"_Out" # output file name
path_out = r"C:\Users\Xinqiu\Documents\Python Scripts\Output"   

    # ================================================= #
    # Optional check: disabling some functions to run faster if not needed
    # ================================================= #
PlotHist = 0        # if = 1 it shows the measured DToF
LinearCheck = 0     # A simple linearity check of Macrotimes
Save = 0            # if = 1 it saves the processed data (all reconstructed DToFs)
Select = 'SIG'      # The channel from which we want to "split"/decompose the DToFs of each detector

#%%
# =============================================================================
# From PicoQuant GitHub: Reading of the input .PTU file
# =============================================================================
# Modified by Ilaria. Do not modify this cell section
# =============================================================================
# Tag & Record Types
# =============================================================================
tyEmpty8      = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
tyBool8       = struct.unpack(">i", bytes.fromhex("00000008"))[0]
tyInt8        = struct.unpack(">i", bytes.fromhex("10000008"))[0]
tyBitSet64    = struct.unpack(">i", bytes.fromhex("11000008"))[0]
tyColor8      = struct.unpack(">i", bytes.fromhex("12000008"))[0]
tyFloat8      = struct.unpack(">i", bytes.fromhex("20000008"))[0]
tyTDateTime   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
tyAnsiString  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
tyWideString  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
tyBinaryBlob  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

# Record types
rtPicoHarpT3     = struct.unpack(">i", bytes.fromhex('00010303'))[0]
rtPicoHarpT2     = struct.unpack(">i", bytes.fromhex('00010203'))[0]
rtHydraHarpT3    = struct.unpack(">i", bytes.fromhex('00010304'))[0]
rtHydraHarpT2    = struct.unpack(">i", bytes.fromhex('00010204'))[0]
rtHydraHarp2T3   = struct.unpack(">i", bytes.fromhex('01010304'))[0]
rtHydraHarp2T2   = struct.unpack(">i", bytes.fromhex('01010204'))[0]
rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
rtMultiHarpNT3   = struct.unpack(">i", bytes.fromhex('00010307'))[0]
rtMultiHarpNT2   = struct.unpack(">i", bytes.fromhex('00010207'))[0]

# =============================================================================
# Global Variables
# =============================================================================
global inputfile
global outputfile
global recNum
global oflcorrection
global truensync
global dlen
global isT2
global globRes
global numRecords

# =============================================================================
# Read Header
# =============================================================================
inputfile = open(path + file + ext, "rb") # opens the .ptu file in binary read mode

# Check if inputfile is a valid PTU file (all ptu files start with PQTTTR)
# Python strings don't have terminating NULL characters, so they're stripped
magic = inputfile.read(8).decode("utf-8").strip('\0')
if magic != "PQTTTR":
    print("ERROR: Magic invalid, this is not a PTU file.")
    inputfile.close()
    exit(0)

version = inputfile.read(8).decode("utf-8").strip('\0')


# Write the header data to outputfile and also save it in memory.
# There's no do ... while in Python, so an if statement inside the while loop
# breaks out of it
tagDataList = []    # Contains tuples of (tagName, tagValue)
while True:
    tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
    tagIdx = struct.unpack("<i", inputfile.read(4))[0]
    tagTyp = struct.unpack("<i", inputfile.read(4))[0]
    if tagIdx > -1:
        evalName = tagIdent + '(' + str(tagIdx) + ')'
    else:
        evalName = tagIdent
    if tagTyp == tyEmpty8:
        inputfile.read(8)
        tagDataList.append((evalName, "<empty Tag>"))
    elif tagTyp == tyBool8:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        if tagInt == 0:
            tagDataList.append((evalName, "False"))
        else:
            tagDataList.append((evalName, "True"))
    elif tagTyp == tyInt8:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        tagDataList.append((evalName, tagInt))
    elif tagTyp == tyBitSet64:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        tagDataList.append((evalName, tagInt))
    elif tagTyp == tyColor8:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        tagDataList.append((evalName, tagInt))
    elif tagTyp == tyFloat8:
        tagFloat = struct.unpack("<d", inputfile.read(8))[0]
        tagDataList.append((evalName, tagFloat))
    elif tagTyp == tyFloat8Array:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        tagDataList.append((evalName, tagInt))
    elif tagTyp == tyTDateTime:
        tagFloat = struct.unpack("<d", inputfile.read(8))[0]
        tagTime = int((tagFloat - 25569) * 86400)
        tagTime = time.gmtime(tagTime)
        tagDataList.append((evalName, tagTime))
    elif tagTyp == tyAnsiString:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
        tagDataList.append((evalName, tagString))
    elif tagTyp == tyWideString:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        tagString = inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
        tagDataList.append((evalName, tagString))
    elif tagTyp == tyBinaryBlob:
        tagInt = struct.unpack("<q", inputfile.read(8))[0]
        tagDataList.append((evalName, tagInt))
    else:
        print("ERROR: Unknown tag type")
        exit(0)
    if tagIdent == "Header_End":
        break

# Reformat the saved data for easier access
tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]

# get important variables from headers
numRecords = tagValues[tagNames.index("TTResult_NumberOfRecords")]
globRes = tagValues[tagNames.index("MeasDesc_GlobalResolution")] 
Res = tagValues[tagNames.index("MeasDesc_Resolution")]*1e9 # in ns
Res_ps = tagValues[tagNames.index("MeasDesc_Resolution")]*1e12 # in ps
print("Found: %d records" % numRecords)


# =============================================================================
# Read Data
# =============================================================================
          
mtime = []  #microtime information
trace = []  #macrotime information
ch = []     #channel information

    # ================================================= #
    # Function to read in T3 mode
    # ================================================= #
def readHT3(version):
    global inputfile, outputfile, recNum, oflcorrection, numRecords
    T3WRAPAROUND = 1024
    for recNum in range(0, numRecords):
        try:
            recordData = "{0:0{1}b}".format(struct.unpack("<I", inputfile.read(4))[0], 32)
        except:
            print("The file ended earlier than expected, at record %d/%d."\
                  % (recNum, numRecords))
            exit(0)
        
        special = int(recordData[0:1], base=2)
        channel = int(recordData[1:7], base=2)
        dtime = int(recordData[7:22], base=2)
        nsync = int(recordData[22:32], base=2)
        if special == 1:
            if channel == 0x3F: # Overflow
                # Number of overflows in nsync. If 0 or old version, it's an
                # old style single overflow
                if nsync == 0 or version == 1:
                    oflcorrection += T3WRAPAROUND
                    #gotOverflow(1)
                else:
                    oflcorrection += T3WRAPAROUND * nsync
                    #gotOverflow(nsync)
            if channel >= 1 and channel <= 15: # markers
                truensync = oflcorrection + nsync
                #gotMarker(truensync, channel)
        else: # regular input channel
            truensync = oflcorrection + nsync
            #gotPhoton(truensync, channel, dtime)
            
        """ Here all valid data from each photon event is recorded into a list
        """
        trace.append(truensync) # macrotimes: the true number of syncs           
        mtime.append(dtime)  # The list for microtimes
        ch.append(channel)
            

        if recNum % 100000 == 0:
            sys.stdout.write("\rProgress: %.1f%%" % (float(recNum)*100/float(numRecords)))
            sys.stdout.flush()


    # ================================================= #
    # Check the data read type (T2 or T3) and read the data
    # ================================================= #
oflcorrection = 0
dlen = 0
recordType = tagValues[tagNames.index("TTResultFormat_TTTRRecType")]

if recordType == rtMultiHarpNT3:
     isT2 = False
     print("MultiHarp150N T3 data")
     # call in T3 function to read the file
     readHT3(2)
     
# elif recordType == rtMultiHarpNT2:
#      isT2 = True
#      print("MultiHarp150N T2 data")
#      readHT2(2)
     
else:
    print("ERROR: Unknown record type")
    exit(0)

inputfile.close()

# =============================================================================
# make a pd dataframe
# all units are in bins!!!
# =============================================================================
df = pd.DataFrame({"Channel" : ch,
                   "MicroTime" : mtime,
                   "MacroTime" : trace})

MicroRebinning = 1  #rebins the microtime tags

# Making the binning edges from -period to period, with a step of bin_width
# Ex: for 80MHz -> period = 12.5ns, bin_width = 5ps -> [-12500, -12495, ..., 0, 5, 10, ..., 12500]
period = round(globRes*1e+12)   # in ps
binning = np.arange(-period, period + Res_ps, MicroRebinning * Res_ps)

print("\rTypes of Channel: {}".format(df['Channel'].unique()))

#%%
# =============================================================================
# Extract Macrotime (still in #), Microtime (converted to ps)
# =============================================================================
# from dataframe locate the microtime data of corresponding channel, 
# and convert this data into a list of float

Micro_Sig = df.loc[df['Channel'] == TDC_Sig, "MicroTime"].to_numpy()
Micro_Sig = Micro_Sig * int(Res_ps) # The microtimes from #bins to ps
# Micro_Sig = Micro_Sig.astype(int)

Micro_Ref = df.loc[df['Channel'] == TDC_Ref, "MicroTime"].to_numpy()
Micro_Ref = Micro_Ref * int(Res_ps)
# Micro_Ref = Micro_Ref.astype(int)

Macro_Sig = df.loc[df['Channel'] == TDC_Sig, "MacroTime"].to_numpy()
Macro_Ref = df.loc[df['Channel'] == TDC_Ref, "MacroTime"].to_numpy()

if GND_True == 1:
    Micro_D1 = df.loc[df['Channel'] == TDC_D1, "MicroTime"].to_numpy()
    Micro_D1 = Micro_D1 * int(Res_ps)

    Micro_D2 = df.loc[df['Channel'] == TDC_D2, "MicroTime"].to_numpy()
    Micro_D2 = Micro_D1 * int(Res_ps)
    
    Macro_D1 = df.loc[df['Channel'] == TDC_D1, "MacroTime"].to_numpy()
    Macro_D2 = df.loc[df['Channel'] == TDC_D2, "MacroTime"].to_numpy()


#%%   
# =============================================================================
# Plot Histograms
# =============================================================================

if PlotHist == 1:    
    plt.figure('Histogram Raw Data TDCs', figsize=(6, 5))   
    Hist_Sig, _, _ = plt.hist(Micro_Sig, bins = binning, histtype='step', log = True, label = 'TDC 1 - Signal')
    Hist_Ref, _, _ = plt.hist(Micro_Ref, bins = binning, histtype='step', log = True, color = 'orange', label = 'TDC 2 - Reference')
    
    if GND_True == 1:
        Hist_D1, _, _ = plt.hist(Micro_D1, bins = binning, histtype='step', color='green', log = True, label = 'True Det1')
        Hist_D2, _, _ = plt.hist(Micro_D2, bins = binning, histtype='step', color = 'grey', log = True, label = 'True Det2')
       
    plt.legend(loc='upper right')
    plt.ylim(100,)
    plt.xlim(0,period)
    plt.title('Histogram Raw Data TDCs')
    plt.ylabel('Number of counts')
    plt.xlabel('time [ps]')
    ax = plt.gca()
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(which='major', direction = 'in')
    ax.tick_params(which='minor', direction = 'in')
    ax.grid(True, which = 'major', color = 'darkgrey', linestyle = '-')
    ax.grid(True, which = 'minor', color = 'lightgrey', linestyle = '--')
    
    plt.show()

#%%
# =============================================================================
# Golden Truth check 1: sync check
# =============================================================================

if LinearCheck == 1:
    x_TDC1 = range(0, len(Macro_Sig))
    x_TDC2 = range(0, len(Macro_Ref))
    
    
    plt.figure('Linearity Macrotime Signal vs Reference', figsize=(6, 5))
    plt.plot(x_TDC1, Macro_Sig, label='TDC1 - Signal', color='blue', linewidth=3)
    plt.plot(x_TDC2, Macro_Ref, label='TDC2 - Reference (1 ns)', color='orange')
    
    plt.legend(loc='upper left')
    plt.title('Linearity Macrotime TDC1 vs TDC2')
    plt.ylabel('Macrotime [Sync number]')
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(which='major', direction = 'in')
    ax.tick_params(which='minor', direction = 'in')
    ax.grid(True, which = 'major', color = 'darkgrey', linestyle = '-')
    ax.grid(True, which = 'minor', color = 'lightgrey', linestyle = '--')
    
    plt.show()




#%%

# =============================================================================
# Step 1: Generic/uncorrelated algorithom to Re-aligning the Macrotimes
# =============================================================================

i_Sig = 0       # index for advancing in the Macro_Sig array
i_Ref = 0       # index for advancing in the Macro_Ref array
k = 0           # index for writing the aligned_Sig and Ref arrays
o_Sig = 0       # temporal var for jumps in the array due to an orphan
o_Ref = 0
n_delyedPhotons = 0
twins_Sig = 0
twins_Ref = 0
Orphans_Sig = []
Orphans_Ref = []

# BREAK = False

max_len = min(len(Macro_Sig), len(Macro_Ref))
# limit = 1000
limit = max_len

# Pre-allocated np arrays for a more efficient data management
aligned_Sig = np.zeros((int(limit*1.2), 2), dtype=np.int64)
aligned_Ref = np.zeros((int(limit*1.2), 2), dtype=np.int64)

    # =============================================================================
    # Iteration through all Signal Macrotime events
    # =============================================================================

while i_Sig < limit:
    # Progress reader to not lose patience...
    if i_Sig % 100000 == 0:
        sys.stdout.write("\rProgress: %.1f%%" % (float(i_Sig)*100/float(limit)))
        sys.stdout.flush()
    
    if (i_Ref + o_Ref) == len(Macro_Ref):
        break
    
    # First case: the most common/desired case is that two events match
    if Macro_Sig[i_Sig] == Macro_Ref[i_Ref]:
        aligned_Sig[k] = [Macro_Sig[i_Sig], Micro_Sig[i_Sig]]
        aligned_Ref[k] = [Macro_Ref[i_Ref], Micro_Ref[i_Ref]]
        k += 1
        i_Sig += 1
        i_Ref += 1
    else:
        # ====================
        # Unmatched event at index 0 - (no previous events to be look at)
        # ====================        
        if i_Sig == 0:
            Orphans_Ref.append([i_Ref, Macro_Ref[i_Ref], Micro_Ref[i_Ref]])
            Orphans_Sig.append([i_Sig, Macro_Sig[i_Sig], Micro_Sig[i_Sig]])
            i_Sig += 1
            i_Ref += 1   
            continue
        
        # ====================
        # Delayed Ref photons - arrived at next Macrotime
        # ====================
        if (Macro_Ref[i_Ref] - Macro_Sig[i_Sig]) == 1:
            aligned_Sig[k] = [Macro_Sig[i_Sig], Micro_Sig[i_Sig]]
            aligned_Ref[k] = [Macro_Ref[i_Ref] -1, Micro_Ref[i_Ref]+period]
            k += 1
            i_Sig += 1
            n_delyedPhotons += 1
            if i_Sig == limit:
                break
            else:
                i_Ref += 1
            continue
        
        # ====================
        # Twins at Sig and Ref
        # ====================
        elif Macro_Sig[i_Sig] == Macro_Sig[i_Sig-1]:
            aligned_Sig[k] = [Macro_Sig[i_Sig], Micro_Sig[i_Sig]]
            aligned_Ref[k] = aligned_Ref[k-1]
            k += 1
            i_Sig += 1
            twins_Sig += 1
            continue
        elif Macro_Ref[i_Ref] == Macro_Ref[i_Ref-1]:            
            aligned_Ref[k] = [Macro_Ref[i_Ref], Micro_Ref[i_Ref]]
            aligned_Sig[k] = aligned_Sig[k-1]
            k += 1
            i_Ref += 1
            twins_Ref += 1
            continue
        
            # ====================
            # Last pair of Orphans - to avoid IndexError
            # ====================
        elif ((i_Sig + 1) == len(Macro_Sig)) or ((i_Ref + 1) == len(Macro_Ref)):
            Orphans_Ref.append([i_Ref + o_Ref, Macro_Ref[i_Ref+o_Ref], Micro_Ref[i_Ref+o_Ref]])
            Orphans_Sig.append([i_Sig + o_Sig, Macro_Sig[i_Sig + o_Sig], Micro_Sig[i_Sig + o_Sig]])
            break
        
        else:
            # ====================
            # Orphans in Ref
            # ====================
            while Macro_Sig[i_Sig] > Macro_Ref[i_Ref + o_Ref]:
                Orphans_Ref.append([i_Ref + o_Ref, Macro_Ref[i_Ref+o_Ref], Micro_Ref[i_Ref+o_Ref]])
                o_Ref += 1
            
                    
            if o_Ref > 0:
                if Macro_Sig[i_Sig] == Macro_Ref[i_Ref + o_Ref]:
                    aligned_Sig[k] = [Macro_Sig[i_Sig], Micro_Sig[i_Sig]]
                    aligned_Ref[k] = [Macro_Ref[i_Ref + o_Ref], Micro_Ref[i_Ref + o_Ref]]
                    k += 1
                    i_Ref += (o_Ref + 1)
                    i_Sig += 1 
                else:
                    i_Ref += o_Ref
                    
                               
                o_Ref = 0
                continue
            
            # ====================
            # Orphans in Sig
            # ====================
            while Macro_Ref[i_Ref] > Macro_Sig[i_Sig + o_Sig]:
                Orphans_Sig.append([i_Sig + o_Sig, Macro_Sig[i_Sig + o_Sig], Micro_Sig[i_Sig + o_Sig]])
                o_Sig += 1
                    
            if o_Sig > 0:
                if Macro_Ref[i_Ref] == Macro_Sig[i_Sig + o_Sig]:
                    aligned_Ref[k] = [Macro_Ref[i_Ref], Micro_Ref[i_Ref]]
                    aligned_Sig[k] = [Macro_Sig[i_Sig + o_Sig], Micro_Sig[i_Sig + o_Sig]]
                    k += 1
                    i_Sig += (o_Sig + 1)
                    i_Ref += 1    
                else:
                    i_Sig += o_Sig
                
                            
                o_Sig = 0
                
# Truncate the non-zero values                
aligned_Sig = aligned_Sig[:k]
aligned_Ref = aligned_Ref[:k]


#%%
# =========================================================================
# Step2: compute the microtime differences at each matching macrotime (sync)
# =========================================================================

    # ================================================= #
    # Define the array for computing the microtime difference
    # ================================================= #
aMicro_Sig = aligned_Sig[:, 1]
aMicro_Ref = aligned_Ref[:, 1]
Micro_diff = aMicro_Ref - aMicro_Sig

aligned_diff = np.stack((aMicro_Sig, aMicro_Ref, Micro_diff), axis=1)

    # ================================================= #
    # Plotting the DToF after discarding the orphans
    # ================================================= #
if LinearCheck == 1:
    plt.figure('Histogram aligned Data', figsize=(6, 5))
    plt.hist(aMicro_Sig, bins = binning, histtype='step', log = True, label = 'Sig - Aligned Microtime')
    plt.hist(aMicro_Ref, bins = binning, histtype='step', log = True, label = 'Ref - Aligned Microtime')
    
        
    plt.ylim(100,)
    plt.xlim(0,period)
    plt.ylabel('Number of counts')
    plt.xlabel('Time [ps]')
    plt.title('Aligned Data (orphans discarted)')
    ax = plt.gca()
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(which='major', direction = 'in')
    ax.tick_params(which='minor', direction = 'in')
    ax.grid(True, which = 'major', color = 'darkgrey', linestyle = '-')
    ax.grid(True, which = 'minor', color = 'lightgrey', linestyle = '--')       
    plt.legend() 
    plt.show() 


    # ================================================= #
    # Plotting the difference in microtimes
    # ================================================= #

plt.figure('Histogram Micro Time Difference', figsize=(6, 5))
Counts_Micro_diff, _, _, = plt.hist(Micro_diff, bins = binning, histtype='step', log = True, label = r'$\Delta Micro Time$ (Ref - Sig)')
      
plt.ylim(0,)
plt.ylabel('Number of counts')
plt.xlabel('MicroTime difference [ps]')
plt.title('Histogram Micro Time Difference')
ax = plt.gca()
ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax.tick_params(which='major', direction = 'in')
ax.tick_params(which='minor', direction = 'in')
ax.grid(True, which = 'major', color = 'darkgrey', linestyle = '-')
ax.grid(True, which = 'minor', color = 'lightgrey', linestyle = '--')
plt.legend()
plt.show()  

#%%
# =========================================================================
# Step 3: photon classification
# =========================================================================
    # ================================================= #
    # Step 3-1: Looking for difference minimums
    # ================================================= #

# 1. Defining the interested region
bin_centers = (binning[:-1] + binning[1:]) / 2          # bin centers instead of bin edges (Ex: 2.5ps is the bin center of 0-5ps)
x_min, x_max = -delay*1.3, num_det*delay*1.3            # time limits [ps] for the interested region
mask = (bin_centers >= x_min) & (bin_centers <= x_max)  # bins (boolean values) between the defined region

x_data = bin_centers[mask]          # bin centers (ps) between the defined region
y_data = Counts_Micro_diff[mask]    # number of counts at these bin centers
inverted_y_data = -y_data

# 2. Find peaks in the inverted signal
if IRF == 0:
    # Multi-Detectors case
    valley_indices, _ = find_peaks(
        inverted_y_data, 
        # distance: Minimum horizontal distance between neighboring peaks (in samples)
        distance = int(delay/5),  # applied delay between two min peaks
        height = -2e4
    )
else:   
    # IRF case
    valley_indices, _ = find_peaks(
        inverted_y_data, 
        distance = 300,     # 300*5=1500 ps between two min peaks
        height = -2e4       # the minima peaks are below -2e4 counts
    )

 
# 3. Get the (x, y) coordinates of the valleys
valley_x_coords = x_data[valley_indices]
valley_y_coords = y_data[valley_indices]


    # ================================================= #
    # Plotting the found minima peaks
    # ================================================= #

fig, ax = plt.subplots(figsize=(8, 6))
# Plot the region of histogram from Micro_diff
ax.plot(
    x_data, y_data,
    label='Hist Micro_diff (range of interest)',
    linewidth=2,
    color='tab:blue')

# Plot the Detected Valleys (Minima) as markers
ax.plot(
    valley_x_coords, valley_y_coords,
    'x',  # Marker style: 'x' cross
    color='red',
    markersize=8,
    label='Detected Minima')

ax.set_yscale('log')
ax.set_title('Histogram with Detected Minima (linear plot)')
ax.set_xlabel('MicroTime difference [ps]')
ax.set_ylabel('Number of counts')
ax.legend()
ax.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.show()

    # ================================================= #
    # Check minima
    # ================================================= #

if len(valley_indices) > (num_det + 1):
    print("Number of minima found is higher than needed")
elif len(valley_indices) < (num_det + 1):
    print("Number of minima found is lower than needed")
else:
    print('Minima found at {} ps'.format(valley_x_coords))
    
index_first = np.where(binning == int(valley_x_coords[0] + 2.5))[0][0]
index_last = np.where(binning == int(valley_x_coords[num_det] + 2.5))[0][0]

counts_ignored_before = np.sum(Counts_Micro_diff[:index_first])
counts_ignored_after = np.sum(Counts_Micro_diff[index_last + 1:])
total_ignored_counts = counts_ignored_before + counts_ignored_after
per_ignored = total_ignored_counts * 100 / np.sum(Counts_Micro_diff)
print("Total counts ignored outside the first and last peak: %d, %.2f%% of the total."
      % (total_ignored_counts, per_ignored))


#%%
    # =========================================================================
    # Step 3-2: Time windows applied to the Micro_diff for the photon classification
    # =========================================================================
if IRF == 0:
    # Variable W for time window
    W = np.zeros((num_det, 2))
    
    for i in range(num_det):
        W[i,0] = valley_x_coords[i]
        W[i,1] = valley_x_coords[i+1]
else:
    # Set the time window depending on the IRF
    for i in range(0, len(valley_y_coords)):
        if valley_y_coords[i] == 0:
            W = np.array([
                [valley_x_coords[i-1], valley_x_coords[i]]])
            break
        else:
            W = np.array([
                [valley_x_coords[-2], valley_x_coords[-1]]])

    if W.ndim == 1:
        if W.shape[0] == 2:
            W = W.reshape(1, 2)
        else:
            raise ValueError("Window file does not contain valid [start, stop] format.")
        
print("Window shape:", W.shape) 

#%%
    # ==========================================
    # Step 3-3: Reconstruct Detectors' Signals
    # ==========================================
# Select TDC channel
if Select == 'REF':
    ch = 1
elif Select == 'SIG':
    ch = 0

# aligned_diff = np.stack((aMicro_Sig, aMicro_Ref, Micro_diff), axis=1)
Detectors = [
    aligned_diff[((aligned_diff[:,2])>=W[n,0]) & ((aligned_diff[:,2])<W[n,1]),ch]
    for n in range(num_det)
    ]


#%%
# =============================================================================
# Plotting the Reconstructed Detectors' Signals
# ============================================================================= 

num_bins = len(binning) 

Hist_Det = np.zeros((num_det, num_bins-1))
bins_Det = np.zeros((num_det, num_bins))


plt.figure('Histogram of Reconstructed Data', figsize=(8, 7))
for n in range(num_det):
   counts, bin_edges, _ = plt.hist(Detectors[n], bins = binning, histtype='step', log = True, label = 'Reconstructed: Det ' + str(n+1))
   Hist_Det[n, :] = counts
   bins_Det[n, :] = bin_edges
   
# Add also raw data to compare
plt.hist(Micro_Ref, bins = binning, histtype='step', log = True, label = 'TDC Ref - Raw Data')
plt.hist(Micro_Sig, bins = binning, histtype='step', log = True, label = 'TDC Sig - Raw Data')

plt.ylim(100,)
plt.xlim(0, period)
plt.title('Histogram Reconstructed DToFs')
plt.ylabel('Number of counts')
plt.xlabel('Time [ps]')

ax = plt.gca()
ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax.tick_params(which='major', direction='in')
ax.tick_params(which='minor', direction='in')
ax.grid(True, which='major', color='darkgrey', linestyle='-')
ax.grid(True, which='minor', color='lightgrey', linestyle='--')


plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2)
    # Legend: loc for anchor point, bbox_to_anchor for anchor position relative to the chart
    # ncol = number of columns

plt.tight_layout()
plt.show()

#%%
# =============================================================================
# Section C: Comparison Ground Truth vs Reconstructed Signal
# ============================================================================= 
if GND_True == 1:
    # From the aligned Sig events, extract only the events (Macro, Micro) belonging to Det1
    Reconst_D1_Micro = np.array(Detectors[0])
    D1_indexes = ((aligned_diff[:,2])>=W[0,0]) & ((aligned_diff[:,2])<W[0,1])       # Indexes belonging to D1
    Reconst_D1 = np.stack((aligned_Sig[D1_indexes, 0], Reconst_D1_Micro), axis=1)
    
    # Final arrays for the Scatter plot: resizing the two arrays
    Scatter_ReconsD1 = np.zeros((int(len(Micro_D1)*1.3), 2), dtype=np.int64)
    Scatter_TrueD1 = np.zeros((int(len(Micro_D1)*1.3), 2), dtype=np.int64)
    r = 0   # index for checking Scatter_ReconsD1 (the reconstructed D1)
    tr = 0   # index for checking Micro_D1 (True D1)
    s = 0   # index to iterate the loop, also it will be the final length of both arrays
    
    # Array for the difference between the True and the Reconstructed
    Reconst_diff = np.zeros((int(len(Micro_D1)*1.3), 1), dtype=np.int64)
    
    # Assigning the microtime of -1000 to all missing events of both arrays
    while tr < len(Macro_D1):
        # Matching events: note down the value for Scatter and their difference
        if Macro_D1[tr] == Reconst_D1[r, 0]:
            Scatter_ReconsD1[s] = Reconst_D1[r]
            Scatter_TrueD1[s] = [Macro_D1[tr], Micro_D1[tr]]
            Reconst_diff[s] = Micro_D1[tr] - Reconst_D1[r, 1]
            # update the three indexes for next comparison
            tr += 1
            r += 1
            s += 1
            continue
        # Missing Reconstructed event
        elif Macro_D1[tr] < Reconst_D1[r, 0]:
            Scatter_ReconsD1[s] = [Macro_D1[tr], -1000]
            Scatter_TrueD1[s] = [Macro_D1[tr], Micro_D1[tr]]
            Reconst_diff[s] = 12500
            # Move to next True event, stay on current Reconstructed event
            tr += 1
            s += 1
            continue
        else:
            # Missing True event
            Scatter_ReconsD1[s] = Reconst_D1[r]
            Scatter_TrueD1[s] = [Reconst_D1[r, 0], -1000]
            Reconst_diff[s] = -12500
            r += 1
            s += 1

    
    # Truncate the finalized arrays for Scatter and Histogram
    Scatter_ReconsD1 = Scatter_ReconsD1[:s]  
    Scatter_TrueD1 = Scatter_TrueD1[:s]
    Reconst_diff = Reconst_diff[:s]
    
    # Take only the microtime
    GroundTruth = Scatter_TrueD1[:,1]
    Reconstructed = Scatter_ReconsD1[:, 1]

    #%%
    # ==========================================
    # Plotting the DToFs of GroundTrue and Reconstructed
    # ==========================================
    
    plt.figure('Histogram of Reconstructed Data & GroundTruth', figsize=(8, 7))
    plt.hist(Detectors[0], bins = binning, histtype='step', log = True, label = 'Det1 - Reconstr. Data')
    plt.hist(Micro_D1, bins = binning, histtype='step', log = True, label = 'Det1 - True Data')
    plt.ylim(0,)
    plt.xlim(0, period)
    plt.title('Histogram Reconstructed vs GND Truth DToFs')
    plt.ylabel('Number of counts')
    plt.xlabel('Time [ps]')
    ax = plt.gca()
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(which='major', direction='in')
    ax.tick_params(which='minor', direction='in')
    ax.grid(True, which='major', color='darkgrey', linestyle='-')
    ax.grid(True, which='minor', color='lightgrey', linestyle='--')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2)
    plt.tight_layout()
    plt.show()
    
    # ==========================================
    # Plotting the colored scatter plot
    # ==========================================
    # Compute density grid
    stat, xedges, yedges, _ = binned_statistic_2d(
        GroundTruth, Reconstructed, None, statistic='count', bins=1000
    )
    # if bins = 1000, it means that the axis are divided into 12500/1000=12.5ps
    # so for each 12.5x12.5ps region, stat gives its number of counts
    # In our case, we have actually 12500+1000 (the outliners)=13500ps, so 13.5ps/division
    
    # Assign each point its bin’s count
    x_bin = np.digitize(GroundTruth, xedges) - 1
    y_bin = np.digitize(Reconstructed, yedges) - 1
    x_bin = np.clip(x_bin, 0, stat.shape[0] - 1)
    y_bin = np.clip(y_bin, 0, stat.shape[1] - 1)
    z = stat[x_bin, y_bin]
    
    # Scatter with color by density
    plt.figure(figsize=(8,7))
    plt.scatter(GroundTruth, Reconstructed, c=z, s=1, cmap='inferno', norm='log')
        # x-axis and y-axis being GroundTruth, Reconstructed
        # Changing x-axis and y-axis won't change the scatter plot, its value is given by c=z
    plt.xlabel("GroundTruth Data (ps)")
    plt.ylabel("Reconstructed Data (ps)")
    plt.title("Scatter Colored by Local Density")
    plt.colorbar(label="Counts per bin")
    plt.axvline(x=4500, linestyle='--', linewidth=2)
    plt.axvline(x=5600, linestyle='--', linewidth=2)
    # plt.plot([-1000, 12500], [-1000, 12500], 'g--', lw=1)
    plt.show()
    
    # ==========================================
    # Exporting the scatter plot to .mat file
    # ==========================================    
    
    path_mat = r"C:\Users\Xinqiu\Documents\Python Scripts\Output\\"   # output path for .mat file, double \\ for non scape symbol
    savemat(path_mat+"scatter_data.mat", {
        "x": GroundTruth,       # x-axis
        "y": Reconstructed,     # y-axis
        "color": z              # the colormap info
    })

    
#%%
  
    # ==========================================
    # Plotting the differences (in Microtimes) between GNDTruth and Reconstruction
    # ==========================================
    # Reconst_diff is computed when re-sizing the True and Recons arrays for Scatter plot
    plt.figure(figsize=(8, 7))
    Rec_diff_DToF, _, _ = plt.hist(Reconst_diff, bins = binning, histtype='step', log = True, label = r'$\Delta Micro Time$ (GNDTruth - Reconstruction)')
    plt.ylim(0,1e6)
    plt.title('Hist Microtime differences of (GND Truth - Reconstructed)')
    plt.ylabel('Number of counts')
    plt.xlabel('Time [ps]')
    plt.xticks(range(-12500, 12501, 2500))
    ax = plt.gca()
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(which='major', direction='in')
    ax.tick_params(which='minor', direction='in')
    ax.grid(True, which='major', color='darkgrey', linestyle='-')
    ax.grid(True, which='minor', color='lightgrey', linestyle='--')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
    plt.tight_layout()
    plt.show()


    # ==========================================
    # To give some data about the reconstruction
    # ==========================================
    # Number of events before -1000 ps (by naked eye, where the noise starts) and after 1000ps
    index_start = np.where(binning == -1000)[0][0]
    index_end = np.where(binning == 1000)[0][0]
    n_upper = Rec_diff_DToF[1:index_start].sum()    # Total number of events before -1000ps excluding at -12500 (missing TrueDet1 events)
    n_lower = Rec_diff_DToF[index_end:-1].sum()     # Total number of events after 1000ps excluding at 12500 (missing reconstruted events)
    
    # Percentage of orphans and misplaced events
    per_orphans = (Rec_diff_DToF[0] + Rec_diff_DToF[-1])*100/len(Reconstructed)
    per_misplaced = (n_lower + n_upper)*100/len(Reconstructed)
    
    # Print out these data
    print('''The {:.0f} orphans are {:.2f}% of the total while the misplaced events are {:.0f} and {:.0f}.\r
The total number of misplaced events are {:.0f}, which is {:.2f} of the total.'''.format(
          Rec_diff_DToF[0] + Rec_diff_DToF[-1], per_orphans, n_upper, n_lower, n_lower + n_upper, per_misplaced))


#%%
    # ==========================================
    # Slicing the Reconstruction DToF when GNDTruth is at its peak
    # ==========================================
    
    # Record all the Macrotimes when the microtimes of GNDTruth are at its peak
    GND_peak_start = 4500
    GND_peak_stop = 5600
    # Pre-allocate array to optiminize memory, estimated length
    Reconst_slice = np.zeros((int(len(GroundTruth)), 1), dtype=np.int64)
    i_slice = 0     # index of the macrotimes where the peaks at GNDTruth happens
    
    # For all Micro_True that is in its peak region (yellow region in the Scatter plot)
    # take the microtime of the Reconstruction that happens at the same index i (or Macrotime).
    for i in range(len(Scatter_TrueD1[:, 1])):
        micro_True = Scatter_TrueD1[i, 1]
        if (micro_True >= GND_peak_start) and (micro_True <= GND_peak_stop):
            Reconst_slice[i_slice] = Scatter_ReconsD1[i, 1]
            i_slice += 1
    
    # Truncate the zero values
    Reconst_slice = Reconst_slice[:i_slice]
    
    # Plot the sliced data into DToF
    plt.figure(figsize=(8, 7))
    plt.hist(Reconst_slice, bins = binning, histtype='step', log = True)
    # plt.ylim(0,1e6)
    plt.xlim(-2500, period)
    plt.title('DToF of Sliced Reconstructed Data for when GNDTruth are [{}, {}]'.format(GND_peak_start, GND_peak_stop))
    plt.ylabel('Number of counts')
    plt.xlabel('Time [ps]')
    plt.xticks(range(-2000, 12501, 2000))
    ax = plt.gca()
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(which='major', direction='in')
    ax.tick_params(which='minor', direction='in')
    ax.grid(True, which='major', color='darkgrey', linestyle='-')
    ax.grid(True, which='minor', color='lightgrey', linestyle='--')
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
    plt.tight_layout()
    plt.show()
    

#%%
# =============================================================================
# Save Info
# =============================================================================
if Save == 1:
    f = open(path_out + file_out + '_7MultiDet.pckl', 'wb')
    pickle.dump([Detectors, binning, aligned_diff, Hist_Det], f)
    f.close()