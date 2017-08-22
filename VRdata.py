
'''                           #######################
#--------------------------   ## ---   Set up  --- ##     ----------------------
                              #######################
'''
"""import every package you'll need"""
import os 
import scipy.io as scio #scipy
import numpy as np #to use arrays
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.animation as animation #animate plots, unused
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import matplotlib.image as image                      
import csv
import sympy as sp #solves equations
import pandas as pd  #data frames
import matplotlib 
import random
import math
import tkinter as Tk #opening files
from tkinter import filedialog
import time
import re #used for natural sort function
import sys
os.getcwd()
pd.set_option('display.max_columns',0)
plt.ion() # allows closing plots without freezing ipython
from ggplot import *
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from PIL import Image
from ggplot import *
import plotly.plotly as py
import plotly.graph_objs as go


'''                              #######################
#-----------------------------   ## ---   Notes   --- ##     -------------------
                                 #######################
'''
'''
Note: for sniff, True and False are actually ' True' and ' False'


Pandas
    # data_frame.iloc[#:#,#:#] for indexing by number
    # data_frame["header"].iloc[#] for a single specific column
    # data_frame.loc['a', 'b'] for indexing by header, can use range
    # list(data_frame.columns.values) for header names
'''






'''                           #######################
#--------------------------   ## --- Functions --- ##     ----------------------
                              #######################
'''

def open_file():
    ''' Open file using GUI

    Args:
        None

    Returns: 
        data:   pandas df; a subject's trial with all data
    '''
    
    root = Tk.Tk()
    
    #stop extra root window from Tk opening
    root.withdraw()
    
    #close Tk file open window after selecting file and allow matplotlib to plot.
    root.update()
    
    # #Get path
    file_path = Tk.filedialog.askopenfilename()

    # #open the file
    data = pd.read_csv(file_path, sep=',', header=None)
    
    #destroys root window and allows python to quit process
    root.quit() 
    
    return data
    

def open_folder():
    ''' Open folder using GUI

    Args:
        None

    Returns: 
        file_path:    str; path to folder
    '''
    
    root = Tk.Tk()
    
    #stop extra root window from Tk opening
    root.withdraw()
    
    #close Tk file open window after selecting file and allow matplotlib to plot.
    root.update()
    
    file_path = Tk.filedialog.askdirectory()
    
    root.quit() 
    
    return file_path

def natural_sort(l): 
    ''' Sorts data alphabetically
    
    Args:
        l: list of strings
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def org_data(folder, csv_names, df_names, setup):
    ''' Reads all CSV names for a single subject's trials
    
    Args:
        folder:     str; path to folder
        csv_names:  list of str; contains all file names in folder
        df_names:   list of str; contains the trial number
        setup:      pd df; contains information about the trial 
    
    Returns:
        userdata:   dict; contains raw data of the user's trial
        
    '''
    
    # Zip the csv names and data frames into a tuple so they can both be read iteratively at once
    # read out the file names and dataframe names for each index
    # Create empty dictionary
    
    header_names = ['time', 'frames', 'x', 'z', 'density_l', 'density_r', 
        'density_both','angle', 'sniff', 'date_time',
        'codename', 'trial', 'file_path1', 'file_path2'
        ]
    userdata = {}
    
    for fname, dfname in zip(csv_names, df_names):
        # temp dataframe is created by reading the path formed by our folder pathway from before and the file name
        df = pd.read_csv(folder+"/"+fname, sep=",", header=None)

        # enter column headers for dataframe
        df.columns = header_names

        # convert sniff to categeorical data
        # note to self: ' False' and ' True' have spaces before them
        df['sniff'] = df['sniff'].astype('category')
        
        # some usernames are all numbers, convert to str
        df['codename']= df['codename'].astype(str)

        # Correct the angles for the program because they're off by 90 degrees
        df["angle"]=df["angle"]*(-1) + 90

        # # Replace the hh:mm:ss.ms time format to just milliseconds
        ms = get_msec(df.time)
        df["time_ms"]= ms
        
        condition = df['file_path2'][0][-6]
        df['condition']=setup[condition][0]
        
        # # for pilot data
        # if (hint=='a'):
        #     df['hint'] = 'no'
        # elif (hint=='b'):
        #     df['hint'] = 'yes'
    
        # trial = df['trial'][0]
        # row = setup[setup['trial']==trial].index.tolist()
        # df['condition']= setup['audio'][row[0]]
        # df['cue']= setup['cue'][row[0]]
        
    
        # store our new dataframe with the dfname as the key name and the dataframe as the values
        userdata[dfname] = df
    # Finally, add an extra key to store the participant codename for easier access.
    userdata['codename']=userdata['1'].codename[0]
    userdata['codename'] = userdata['codename'].replace(" ", "")  
    
    return userdata
    
    
def get_msec(hhmmss):
    ''' Transforms values from minutes, seconds and milliseconds into seconds

    Args:
        hhmmss:      str; form hh:mm:ss:ff

    Returns: 
        ms:    int; time in milliseconds
    '''
    
    ms=()
    num_time = (len(hhmmss[0].split(":")))-1
    
    # some of the time formats are only in hhmmss, others in mmss only
    # check for which type before calculating ms
    if (num_time == 2):
        for i in range(0,len(hhmmss)):
            # print(hhmmss.iloc[i])
            h, m, s = hhmmss.iloc[i].split(':')
            ms_i = round((int(h)*3600) + (int(m) * 60*1000) + float(s)*1000)
            ms = np.append(ms, ms_i)
    elif (num_time == 1):
        for i in range(0,len(hhmmss)):
            m, s = hhmmss.iloc[i].split(':')
            ms_i = round((int(m) * 60*1000) + float(s)*1000)
            ms = np.append(ms, ms_i)
    return ms

def get_time(ms):
    ''' Transforms values from milliseconds to minutes, seconds 

    Args:
        ms:      int; 

    Returns: 
        time:    str; time in minutes and seconds
    '''
    
    m = int(ms/1000.0/60.0)
    sec = int(ms/1000.0%60.0)
    time = str(m) + " minutes, " + str(sec) + " seconds."
    return time

def animate_walk(userdata, setup, start_trial, end_trial, speed, folder):
    ''' Creates an animated plot of a user's trial

    Args:
        userdata:       pd df; all trials from a user
        setup:          pd df; data about trial set ups
        start_trial:    int; trial to start graphing (1-8)
        end_trial:      int; trial to end graphing (1-8)
        speed:          int; speed of graphing animation (0-1000)
        folder:         unused

    Returns: 
        None
    '''
    
    codename = userdata['codename']
    
    for i in range(start_trial, end_trial+1):
        trial = str(i)
        
        row = setup[setup['trial']==i].index.tolist()

        loc_x = userdata[trial]['x']
        loc_z = userdata[trial]['z']
        angle = userdata[trial]['angle']
        angle_acc = userdata[trial]['angle_acc']
        angle2source = userdata[trial]['angle2source']
        
        # for funsies, green always will always point at the big arrow.
        # Dc = userdata[trial]['Dc']
        # Ds = userdata[trial]['Ds']

        plume_x = (setup['plume_x'].iloc[row[0]])
        plume_z = (setup['plume_z'].iloc[row[0]])
        plume_angle_x = (setup['plume_angle_x'].iloc[row[0]])
        plume_angle_z = (setup['plume_angle_z'].iloc[row[0]])

        user_angle_x = (setup['user_angle_x'].iloc[row[0]])
        user_angle_z = (setup['user_angle_z'].iloc[row[0]])
        
        final_x = userdata[trial].x.iloc[-1]
        final_z = userdata[trial].z.iloc[-1]
        final_rad = np.deg2rad(userdata[trial].angle.iloc[-1])
        final_angle_x, final_angle_z = np.cos(final_rad), np.sin(final_rad)
        
        plume_found_frame = userdata[trial]['plume_found_frame'][0]

        step = 1.0/float(len(loc_x))
        
        # Create figure
        fig = plt.figure()    
        ax = fig.gca()
        
        # # put in a plume image
        # img = Image.open('/Users/judyjinn/Desktop/plume_8.png')
        # fig.figimage(img, plume_x, plume_z)
        #
        # xy = [plume_x, plume_z]
        # ax.plot(xy[0], xy[1], ".r")
        # imagebox = OffsetImage(img, zoom=0.2)
        # imagebox.image.axes = ax
        #
        # ab = AnnotationBbox(imagebox, xy,
        #                     xybox=(120., -80.),
        #                     xycoords='data',
        #                     boxcoords="offset points",
        #                     pad=0.5,
        #                     arrowprops=dict(
        #                         arrowstyle="->",
        #                         connectionstyle="angle,angleA=0,angleB=90,rad=3")
        #                     )
        #
        # ax.add_artist(ab)


        
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        plt.plot(0,0, marker='o', color='lime', zorder=1)
        # plt.quiver(0, 0, user_angle_x, user_angle_z, color='lime',width=0.01, scale=10, zorder=2);
        plt.title(codename+" "+trial)
        # plt.plot(plume_x, plume_z, marker='o', color = 'lime', zorder=4, alpha=0.5)
        # plt.quiver(plume_x, plume_z, plume_angle_x, plume_angle_z, color='lime',width=0.01, scale=10, zorder=4, alpha = 0.5)
        # plt.quiver(final_x, final_z, final_angle_x, final_angle_z, color='black',width=0.005, zorder=3)
        
        patch = plt.Arrow(0, 0, 0.1, 0.1, color='black')

        def data_gen():
            # This is my attempt to graph xy
            for i in range(0, len(loc_x)):
                x = loc_x[i]
                y = loc_z[i]
                r = 1
                u = r*math.cos(angle[i]*math.pi/180.0)
                v = r*math.sin(angle[i]*math.pi/180.0)
                
                # used to specify arrow vectors for which way to correct plume direction
                # change betwen angle_acc, angle2source, Dc, Ds (last two need some code tweaking to pass in data)
                acc_u = r*math.cos(angle2source[i])
                acc_v = r*math.sin(angle2source[i])

                sniff = userdata[trial]['sniff'].iloc[i]
                
                yield x,y,i,u,v,acc_u,acc_v,sniff


        def get_colour(i, plume_found_frame):
            # cmap = cm.jet
#             c = cmap(int(np.rint(i*step * 255)))
#             cmap = matplotlib.cm.get_cmap('hot')

            # c = cmap(int(np.rint(i*step * 255)))
            
            # color graph
            if i < plume_found_frame:
                c = 'lightgray'
            elif  i >= plume_found_frame:
                c = 'gray'
            return c

        def init():
            ax.add_patch(patch)
            
            if trial=='1':        
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume1.png')
                plt.imshow(img, zorder=0, extent=[plume_x,2,-11,plume_z])
            elif trial=='2':
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume2.png')
                plt.imshow(img, zorder=0, extent=[plume_x,12,plume_z,-5])
            elif trial=='3':
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume3.png')
                plt.imshow(img, zorder=0, extent=[plume_x-2,plume_x+2,plume_z-12,plume_z])
            elif trial=='4':
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume4.png')
                plt.imshow(img, zorder=0, extent=[plume_x,13,plume_z,15])
            elif trial=='5':
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume5.png')
                plt.imshow(img, zorder=0, extent=[-14,plume_x,plume_z,5])
            elif trial=='6':
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume6.png')
                plt.imshow(img, zorder=0, extent=[-15,plume_x,2,plume_z])
            elif trial=='7':
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume7.png')
                plt.imshow(img, zorder=0, extent=[-9,plume_x,6,plume_z])
            elif trial=='8':
                img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume8.png')
                plt.imshow(img, zorder=0, extent=[-6,plume_x,-1,plume_z])
            
            return patch, 

            
        def animate(data):
            x,y,i,u,v,acc_u,acc_v,sniff = data
            plt.scatter(x, y, c=get_colour(i, plume_found_frame), edgecolors='none', zorder=1) 
            
            # # Turn this on and comment other half of animate if you want to see all the direcitonal arrows
            # plt.arrow(x, y, u, v, color='lightgray', zorder=3, alpha=0.5)   # user direction
            # plt.arrow(x, y, acc_u, acc_v, color='green', zorder=1, alpha=0.5) # magnitude of direction from correct
            #     # note that this second line does not point towards the plume origin, only shows magnitude
            #     # of correctness from plume origin (eg. if user is facing )

            
            # note you don't need to store as a patch or return patch,. This is an attempt to add axes as patches to graph arrows that disappear on top of the points.

            ax.patches.pop(0)
            
            if (sniff==' True'):
                patch = plt.Arrow(x, y, u, v, color='coral', zorder=3, alpha=0.5)
            else:
                patch = plt.Arrow(x, y, u, v, color='black', zorder=3, alpha=0.5)
            ax.add_patch(patch)
            

            return patch, 

        ani = animation.FuncAnimation(fig, animate, data_gen, blit=False, interval=speed, repeat=False,init_func=init)

        plt.show()
        # animation.save(codename+'mp4', fps=30)
        plt.close()


    return
    
def just_graph(folder, graph_dict, setup, start_trial, end_trial):
    ''' Graph the trajectory data for a customizable range of trials.
        Does not animate. Saves the graphs into unique user folders

    Args:
        folder:         str; path where main script is located
        graph_dict:     dict; dictionary that contains all user's trials
        setup:          pd df; data about trial set ups
        start_trial:    int; trial to start graphing (1-8)
        end_trial:      int; trial to end graphing (1-8)


    Returns: 
        None
    '''
    
    # Final graphs, no animation. Used for saving    
    date_folders = list(graph_dict.keys()) 
    for date in date_folders:
        user_folders = graph_dict[date]        
        name_temp = []
        for name in user_folders:
            print(name)
            
            name_temp.append(name)
            csv_folder = folder+"/"+date+"/data/"+name
            csv_names = get_csv_names(csv_folder)
            df_names = get_df_names(csv_names)
            userdata = org_data(csv_folder, csv_names, df_names, setup)
            
            results_folder = folder+"/"+date+"/data/"+userdata['codename']+"_results"
            if not os.path.exists(results_folder):
                os.makedirs(folder+"/"+date+"/data/"+userdata['codename']+"_results")
    
            # grab codename and create a new folder for graphs in the results pathway
            codename = userdata['codename']
            graph_folder = results_folder+"/"+codename+"_graphs/"
    
            # check to see if folder already exists, if yes, ignore.
            if not os.path.exists(graph_folder):
                os.makedirs(results_folder+"/"+codename+"_graphs/")
    
            # Only graph designated trials.
            for i in range(start_trial, end_trial+1):
                # Create string for trial number
                trial = str(i)
        
                # Get row of the trial of interest to find the correct information about plume location and orientation
                row = setup[setup['trial']==i].index.tolist()
                
                # Get data for where participant moved during trial and angle of view
                loc_x = userdata[trial]['x']
                loc_z = userdata[trial]['z']
                angle = userdata[trial]['angle']
        
                # From setup files get the plume location, direction
                plume_x = (setup['plume_x'].iloc[row[0]])
                plume_z = (setup['plume_z'].iloc[row[0]])
                plume_angle_x = (setup['plume_angle_x'].iloc[row[0]])
                plume_angle_z = (setup['plume_angle_z'].iloc[row[0]])
                
                userdata[trial]['inplume'] = (userdata[trial].density_both>0.0) 
                userdata[trial]['detect'] = (userdata[trial].inplume) & \
                    (userdata[trial].sniff==' True')
                detect_rows = userdata[trial][userdata[trial]['detect']==True]
                detect_diff = detect_rows.time_ms.diff()
                detect_diff.iloc[0]= detect_rows.iloc[0].time_ms
                plume_found_frame = detect_diff.sort_values(inplace=False).index[-1]        
                

                # Also from setup file get the starting angle direction
                user_angle_x = (setup['user_angle_x'].iloc[row[0]])
                user_angle_z = (setup['user_angle_z'].iloc[row[0]])
        
                # Get final location and direction of participant. Ued to graph the final vector angle
                final_x = userdata[trial].x.iloc[-1]
                final_z = userdata[trial].z.iloc[-1]
                final_rad = np.deg2rad(userdata[trial].angle.iloc[-1]) #convert to radians
        
                # get x and z to specify angle of vector for plt.quiver
                final_angle_x, final_angle_z = np.cos(final_rad), np.sin(final_rad)

                # Calculate percentage of path completed for a single step
                step = 1.0/float(len(loc_x))
        
                # Specific color palette
                cmap = cm.jet
        
                # Set up figure
                # Create figure
                fig = plt.figure()
                ax = fig.gca()

                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)
    
                # # put in a plume image
                if trial=='1':        
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume1.png')
                    plt.imshow(img, zorder=0, extent=[plume_x,2,-11,plume_z])
                elif trial=='2':
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume2.png')
                    plt.imshow(img, zorder=0, extent=[plume_x,12,plume_z,-5])
                elif trial=='3':
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume3.png')
                    plt.imshow(img, zorder=0, extent=[plume_x-2,plume_x+2,plume_z-12,plume_z])
                elif trial=='4':
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume4.png')
                    plt.imshow(img, zorder=0, extent=[plume_x,13,plume_z,15])
                elif trial=='5':
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume5.png')
                    plt.imshow(img, zorder=0, extent=[-14,plume_x,plume_z,5])
                elif trial=='6':
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume6.png')
                    plt.imshow(img, zorder=0, extent=[-15,plume_x,2,plume_z])
                elif trial=='7':
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume7.png')
                    plt.imshow(img, zorder=0, extent=[-9,plume_x,6,plume_z])
                elif trial=='8':
                    img = Image.open('/Users/judyjinn/Python/Human Olfaction/VR/plumeimages/plume8.png')
                    plt.imshow(img, zorder=0, extent=[-6,plume_x,-1,plume_z])
        
                plt.plot(0,0, marker='o', color='lime', zorder=5, alpha=0.5)
                plt.plot(final_x,final_z, marker='o', color='black', zorder=4)
                plt.arrow(final_x, final_z, final_angle_x, final_angle_z, color='black', zorder=3)

                # Split colors by before and after finding plume

                plt.plot(loc_x[0:plume_found_frame], loc_z[0:plume_found_frame], linestyle='-', c='lightgray',  zorder=1)
                plt.plot(loc_x[plume_found_frame:], loc_z[plume_found_frame:], linestyle='-', c='gray', zorder=2)
        
        
                # # For each step, specify a specific gray arrow for directionality and a color for the step.
                # for i in range(0,len(loc_x)):
                #
                #     # Used to specify arrow vectors for angle faced
                #     r = 1;
                #     u = r*math.cos(angle[i]*math.pi/180.0);
                #     v = r*math.sin(angle[i]*math.pi/180.0);
                #
                #     # used to specify arrow vectors for which way to correct plume direction
                #     acc_u = r*math.cos(angle[i]*math.pi/180.0);
                #     acc_v = r*math.sin(angle[i]*math.pi/180.0);
                #
                #     # Gives the color of the plot point. basically it's % of trajectory complete * 255 = color on 0-255 scale
                #     # c = cmap(int(np.rint(i*step * 255)))
                #     if i < plume_found_frame:
                #         c = 'lightgray'
                #     elif  i >= plume_found_frame:
                #         c = 'gray'
                #     # Plor the points and the arrow for direction
                #
                #     plt.scatter(loc_x[i], loc_z[i], c=c, edgecolors='none', zorder=2, alpha=0.5)
                #     # plt.arrow(loc_x[i], loc_z[i], u, v, color='lightgray', zorder=1, alpha=0.5)
                #     # plt.arrow(loc_x[i], loc_z[i], u, v, color='green', zorder=1, alpha=0.5)
            
            
                # Plot a final black arrow for final direction faced by user
                # plt.quiver(final_x, final_z, final_angle_x, final_angle_z, color='black',width=0.005, zorder=3)
        
                # Save it to the directory created earlier
                plt.savefig(graph_folder+"/"+codename+"_"+trial, transparent=True)
                plt.close('all')
                plt.close(fig)

    return
    

def final_stats(userdata, df_names, setup, subj_info):
    ''' Takes all the data for a user and performs the statistics to analyze
        how that person did for finding the plume in terms of final distance,
        time to complete, and angle accuracy. Also parses sniffing rates

    Args:
        userdata:       pd df; all trials from a user
        df_names:       list of str; trial names (1-8)
        setup:          pd df; data about trial set ups
        subj_info:      pd df; all survey answers from subjects


    Returns: 
        finals_df:          pd df; final locations and angles of participant
        results_df:         pd df; accuracy of final location, angle, and time
        results_trials_df:  pd df; easy to read data of results and some "scores" for fun
        results_overall_df: pd df; avg of all trials for location accuracy time, and scores
        sniffpre_all_df:    pd df; location of sniffs before finding plume
        sniffpost_all_df:   pd df; location of sniffs before after plume
        binned_sniff_df:    pd df; sniffs binned by distance from plume origin
    '''

    # Create empty dictionaries and some variables to store data in and be returned
    finals = {}
    results = {}
    results_trials = {}
    results_overall = {}
    sniffpre_all = {}
    sniffpost_all = {}
    binned_sniff = {}
    
    tot_loc = 0
    tot_angle = 0
    tot_time = 0
    
    # Iterate over all trials in dataframe names
    for dfname in df_names:
        # print(dfname)
        
        # Store each trial as a temp dictionary to be saved into the main dictionary then rewritten for the next trial
        temp_finals = {}
        temp_results = {}
        temp_results_trials = {}
        temp_binned_sniff = {}
        
        # Get the correct trial number
        trial = userdata[dfname].trial.iloc[0]
        # Match it with the correct row in the setup file to get map data
        row = setup[setup['trial']==trial].index.tolist()
        row = row[0]
        
        # print(dfname, trial, row)

        # Find plume location and angle values
        plume_x = (setup['plume_x'].iloc[row])
        plume_z = (setup['plume_z'].iloc[row])
        plume_angle_x = (setup['plume_angle_x'].iloc[row])
        plume_angle_z = (setup['plume_angle_z'].iloc[row])
        plume_rad = (setup['plume_rad'].iloc[row])
        
        # Get the codename for the participant. Make sure to eliminate any accidental spaces first and lowercase all.
        codename = codename=userdata[dfname].codename[0].replace(" ","") 
        codename = codename.lower()

        ## ------------- Get values for final location, time to complete trial, and final angle -------------
        # Get final user angle values
        user_angle_x = (setup['user_angle_x'].iloc[row])
        user_angle_z = (setup['user_angle_z'].iloc[row])
        
        # Get final location of user, store into the temp dictionary
        final_x = userdata[dfname].x.iloc[-1]
        final_z = userdata[dfname].z.iloc[-1]
        temp_finals['final_x'] = final_x
        temp_finals['final_z'] = userdata[dfname].z.iloc[-1]
        
                # Use very last row to find the time it took to complete trial and store information
        temp_finals['final_time'] = userdata[dfname].time.iloc[-1]

        # Convert the final angle of direction to radians, store the information
        final_rad = np.deg2rad(userdata[dfname].angle.iloc[-1])
        temp_finals['final_rad'] = math.acos(user_angle_x)
        pd.Series(temp_finals) # I can't remember why this is here...
    
        
        # Store this trial into the finals dictionary to be returned
        finals[dfname] = temp_finals       
        

        
        ## ----------- This section used for classifying sniffing data  -------------
        
        # ---- Short functions in here are being used for pandas.apply to apply a function to a column of data.
        def find_distance(data):
            # Must have plume_x/y in the function, cannot be passed in otherwise pandas.apply() won't work
            plume_x = (full_setup['plume_x'].iloc[row])
            plume_z = (full_setup['plume_z'].iloc[row])
            d = math.hypot((plume_x - data.x), (plume_z - data.z)) 

            return d

        def make_rad(data):
            rad = np.deg2rad(data)
            return rad


        def find_angleacc_dist(data):
            plume_angle_x = (full_setup['plume_angle_x'].iloc[row])
            plume_angle_z = (full_setup['plume_angle_z'].iloc[row])
            rad = np.deg2rad(data.angle)
            c, s = np.cos(rad), np.sin(rad) 
            d = math.hypot((plume_angle_x-c), (plume_angle_z-s))
            d = round(d,2)
            return d

        def acos(distances):
            acos = math.acos((1-(distances**2)/(2.0*(1.0**2))))
            return acos

        def find_angleacc(data):   
            d =  data.apply(find_angleacc_dist, axis=1)
            angle_accuracy = np.pi-d.apply(acos) # convert back to degrees

            return angle_accuracy
        
        def trans2orig(data):
            plume_x = (setup['plume_x'].iloc[7])        
            plume_z = (setup['plume_z'].iloc[7])
        
            # vector from user to plume source
            Dx = plume_x - data['x']
            Dz = plume_z - data['z']
            Drad = np.arctan2(Dz,Dx)
            Dc, Ds = np.cos(Drad), np.sin(Drad) 
        
            # user direction
            rad = np.deg2rad(data.angle)
            c, s = np.cos(rad), np.sin(rad)
        
            # distance between the user vector and the Dvector
            d = math.hypot((Dc-c), (Ds-s))
            d = round(d,2)
            return d
        
        def find_angle2source(data):
            d =  data.apply(trans2orig, axis=1)
            angle2source = d.apply(acos) 
        
            return angle2source
            
        # This is just a note on how to find the difference in angle betwen two vectors. It's what is being used by
        # the functions find_angleacc_dist, acos, and find_angleacc
        # Assess accuracy of angle faced is hard. Logic is to calculate distance arc length between the two points
        # First treat the participant direction and plume direction as their vector components on the unit circle
            # c, s = np.cos(final_rad), np.sin(final_rad) for the user
        
        # Then using this equation:
        # http://math.stackexchange.com/questions/830413/calculating-the-arc-length-of-a-circle-segment 
        
        # find the arc length of the distance between the points as if on a unit circle first
        # d = math.hypot((plume_angle_x-c), (plume_angle_z-s))
        
        # Round this number to the hundredths, because if you don't, 
        # if the distance is close to 180 (facing opposite ways). The distance (d) is near 2 
        # (because diameter of unit circle). And when we take the arccos of the value for d, d can be too high and will
        # cause an error for arccos where the value is impossible to return
            # d=round(d,2)
        
        # Use the equation given to find the angle formed between the plume diretion and the user direction.
        # This mathematically outputs a value in radians between 0 and pi. 
            # angle_accuracy = math.acos((1-(d**2)/(2.0*(1.0**2))))
    
        # ----
        
        userdata[dfname]['distance'] = userdata[dfname].apply(find_distance, axis=1)

        # bins for labeling how close individual is to the scent origin
        # bin label means that individual is within "bin label-0.5 meters" of the goal. 
        # eg bin label 1.5 means individual is 1-1.5 m from goal
        bins = pd.DataFrame({'Bins': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 
                                      5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
                                      10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14,
                                      14.5, 15],
                             'Max': [0.49, 0.99, 1.49, 1.99, 2.49, 2.99, 3.49, 3.99, 4.49, 4.99, 5.49,
                                     5.99, 6.49, 6.99, 7.49, 7.99, 8.49, 8.99, 9.49, 9.99, 10.49, 10.99, 
                                     11.49, 11.99, 12.49, 12.99, 13.49, 13.99, 14.49, 100],
                             'Min': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 
                                     5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
                                     10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14,
                                     14.5] })
        cutoff = np.hstack([np.array(bins['Min'][0]), bins['Max'].values])
        labels = bins['Bins']

        userdata[dfname]['bin_dist'] = pd.cut(userdata[dfname].distance, bins=cutoff, labels=labels, include_lowest=True)

        # Finds the magnitude of the angle that the user is facing into the plume direction (lower is better)
        userdata[dfname]['angle_acc'] = find_angleacc(userdata[dfname]) 
        
        # Finds the magnitude of the angle that the user is to the plume source point(lower is better)
        userdata[dfname]['angle2source'] = find_angle2source(userdata[dfname])
    
        # finding times when the plume was possibly detected/detectable and was missed
        # label values as true if there was a density value greater than 0
        userdata[dfname]['inplume'] = (userdata[dfname].density_both>0.0) 
        # compare when density values were above 0 and a sniff was taken, which means the plume should have been detected
        userdata[dfname]['detect'] = (userdata[dfname].inplume) & (userdata[dfname].sniff==' True')
        detect_rows = userdata[dfname][userdata[dfname]['detect']==True]
        detect_diff = detect_rows.time_ms.diff()
        # # fill in the first time as time elapsed
        detect_diff.iloc[0]= detect_rows.iloc[0].time_ms
    
        # get the row/index number associated with the frame that signals the end of the longest time of not detecting anything.
        plume_found_frame = detect_diff.sort_values(inplace=False).index[-1]
        plume_found_percent = plume_found_frame/len(userdata[dfname])
        userdata[dfname]['plume_found_frame'] = plume_found_frame
        
        
        # print(userdata[dfname].tail(10))
        
        # find all rows where sniff == true
        sniff_rows = userdata[dfname][userdata[dfname]['sniff'] == ' True']

        # get the milisecond time different between sniffs
        sniff_diff = sniff_rows['time_ms'].diff()
        # concatenate in the bin label for the sniffs
        sniff_diff= pd.concat([sniff_diff,  sniff_rows['x'], sniff_rows['z'], sniff_rows['distance'],  
                                sniff_rows['bin_dist'], sniff_rows['angle'], sniff_rows['density_l'],  
                                sniff_rows['density_r'], sniff_rows['density_both'], sniff_rows['angle_acc'],  
                                sniff_rows['angle2source']], axis=1)
        
        # pop out the first row which is a NaN
        sniff_diff = sniff_diff.drop(sniff_diff.index[0])
        # overall mean
        sniff_mean_all = sniff_diff.mean()
        
        sniff_diff['trial']=dfname
        sniff_diff['codename']=codename
        sniff_diff['plume_found_frame'] = plume_found_frame
        sniff_diff['plume_found_percent'] =  plume_found_percent
        sniff_diff['condition'] = userdata[dfname]['condition'][0]
        
        # These diff calculations must be located after sniff_diff is separated here because:
        # 1) It shouldn't look at all movement data because .diff() would calculate changes of 
        #    movement not related to sniffs. EG person sniffs, then walks chaotically and then last dist_change/angle
        #    change is backwards. Does not reflect change between the two sniffs directly
        # 2) If we don't put here in the final_stats loop, the diff() will take the diff of lines between trial #s
        #    Eg. dist of 2.3 for end of trial 1, and then the next dist is 17. The distdiff between those sniff is not true.
        
        # Find changes between sounds, left and right, angles, and distances
        sniff_diff['angle2source_deg'] = np.rad2deg(sniff_diff['angle2source'])
        sniff_diff['l-r'] = sniff_diff['density_l']-sniff_diff['density_r']
        sniff_diff['sound_change']= sniff_diff['density_both'].diff() # positive means louder than previous
        
        # Can't just take difference to find angle change because 1-359 degrees != a left turn of -270 degrees. it means right turn of just 2 degree
        # but 4-6 degrees does mean a left turn of just two degrees
        # therefore, we must use find_angacc again to determine changes 
        # sniff_diff['ang2sor_next']= sniffpost_all['angle2source'].shift(1) 
    

        # shift the angle change and dist change all down one row because the way a person moves should be based on
        # the difference in sound change of the previous time step, thus to line up a change in sound with the resulting action (angle/distance), we need to pull down the row number so it's not 1 ahead
        sniff_diff['angle_change']= sniff_diff['angle2source_deg'].diff()
        sniff_diff['angle_change']= sniff_diff['angle_change'].shift(-1) # negative means getting more accurate
        sniff_diff['dist_change']=sniff_diff['distance'].diff()
        sniff_diff['dist_change']=sniff_diff['dist_change'].shift(-1)      # negative means getting closer

        sniff_diff['x_diff'] = sniff_diff['x'].diff()
        sniff_diff['x_diff'] = sniff_diff['x_diff'].shift(-1)

        sniff_diff['z_diff'] = sniff_diff['z'].diff()
        sniff_diff['z_diff'] = sniff_diff['z_diff'].shift(-1)
        
        # Use .loc to go to the frame number(row value) and not the index.
        sniff_pre = sniff_diff.loc[:plume_found_frame]
        sniff_post = sniff_diff.loc[plume_found_frame:]
                
        # store the information into a temporary dict. Sniff all contains data only after plume is found
        sniffpre_all[dfname]=sniff_pre
        sniffpost_all[dfname]=sniff_post       
    
        # Find the number of sniffs per bin distance
        post_num_sniffs = sniff_post.groupby(['bin_dist']).size()
    
        # Find the average sniff rate per bin distance
        post_avg_sniffrate = sniff_post.groupby(['bin_dist']).time_ms.mean()
    
        # merge the sniff number and average sniff rate together
        temp_binned_sniff = pd.concat([post_num_sniffs, post_avg_sniffrate], axis=1)
        #reset index so bin values are a column
        temp_binned_sniff = temp_binned_sniff.reset_index()
        # rename the columns. First convert the nameless column to a str and then rename.
        temp_binned_sniff.columns = temp_binned_sniff.columns.astype(str)
        temp_binned_sniff.rename(columns={'0':'num_sniff','time_ms': 'sniff_rate'}, inplace=True)
        
        temp_binned_sniff['trial']=dfname
        temp_binned_sniff['codename']=codename
        temp_binned_sniff['plume_found_frame'] = plume_found_frame
        temp_binned_sniff['plume_found_percent'] =  plume_found_percent
        temp_binned_sniff['condition'] = userdata[dfname]['condition'][0]

        # place the temprorary dict into the larger dict sorted by trial names        
        binned_sniff[dfname]= temp_binned_sniff
        
        
        
        ## ------------- Calculate the accuracy of the final location and viewing angle -------------
        temp_results['trial'] = dfname
        # simple distance calculation using hypotenuse function to find distance from plume. Store the accuracy value
        loc_accuracy = userdata[dfname].distance.iloc[-1]
        temp_results['loc_accuracy'] = loc_accuracy
        
        angle_accuracy = userdata[dfname].angle_acc.iloc[-1]
        temp_results['angle_accuracy_rad']= angle_accuracy

        
        temp_results['final_time_ms'] = userdata[dfname].time_ms.iloc[-1]
        
        temp_results['codename'] = codename
        
        # Also get the condition that the participant was in.
        temp_results['condition'] = userdata[dfname]['condition'][0]
        
        # Participant information
        # Use a boolean to find row where with the codename of interest. Then look under column name and get first value
        subj = subj_info[subj_info['Codename']==codename].index.tolist()
        
        if (bool(subj) == False):
            print("codename ", codename, " does not match any participant information")
            sys.exit(1)
            
        subj = subj[0]
        temp_results['sex'] = subj_info['Sex'].iloc[subj]
        temp_results['self_game_xp'] = subj_info['PC Exp'].iloc[subj]
        temp_results['RA_game_xp'] = subj_info['RA Eval'].iloc[subj]
        temp_results['SOD'] = subj_info['SOD'].iloc[subj]
        

        
        # Store the trial values into the results dictionary to be returned
        results[dfname] = temp_results

        ## This section for plotting final user location & angle and plume angle. 
        ## Used as sanity check to make sure results made sense
        # plt.axis([-15, 15, -15, 15])
        # plt.title(dfname)
        # plt.plot(plume_x, plume_z, marker='o', color = 'lime', zorder=4, alpha=0.5)
        # plt.quiver(plume_x, plume_z, plume_angle_x, plume_angle_z, color='lime',width=0.01, scale=10, zorder=4, alpha = 0.5);
        # plt.scatter(final_x, final_z, marker='o', s=15, zorder=3, edgecolors='none')
        # plt.quiver(final_x, final_z, c, s, color='black',width=0.005);
        # plt.show(block=True)
        # plt.clf()

          
        ## ------------- This section mostly used for fast assessment to view results in intuitive way -------------
        # Find the angle accuracy in degrees intead of radians
        temp_results_trials['angle_accuracy_deg']=round(abs(180.0-angle_accuracy*180/math.pi),2)
        # Get the final time and location accuracy as well
        # Time is passed through function that converts the miliseconds back to easy to read min/sec string.
        temp_results_trials['final_time'] = get_time(userdata[dfname].time_ms.iloc[-1])
        temp_results_trials['loc_accuracy'] = round(loc_accuracy,2)
        
        # Calculate some scores for fun. Perfect score =10, worst = 0
        # Basic logic is that I binned values for location accuracy, time, and angle accuracy. 
        # User final results are given a score based on what bin they land in.
        
        # Seperate the time into 30 second bins (recorded in ms, thus 30k = 30 sec)
        # 10 = Trial ended within 30 sec. -1 for every 30 sec after
        # Create the bins of values to use for scores. Iterate through the bins to find the correct score for user.
        score_time = np.arange(30000,330000,30000)
        for t in np.arange(0,len(score_time)):
            # If the results of the user are less than the current bin time, then assign the score
            if (temp_results['final_time_ms'] < score_time[t]):            
                temp_results_trials['score_time'] = (10-t) #assign score
                # also to find an average overall time, add this time to a total time counter
                tot_time = tot_time + temp_results_trials['score_time']
                # junp out of loop once score is assigned.
                break 
            # If the time is higher than 330000 then just give it a score of 0 and stop.
            elif (temp_results['final_time_ms'] > score_time[-1]):
                temp_results_trials['score_time'] = 0
                break
        
        # Binned b5 intervals of 15 degrees.
        score_angle = np.arange(15,165,15)        
        for a in np.arange(0,len(score_angle)):
            if (temp_results_trials['angle_accuracy_deg'] < score_angle[a]):               
                temp_results_trials['score_angle'] = (10-a)
                tot_angle = tot_angle + temp_results_trials['score_angle']
                break
            elif (temp_results_trials['angle_accuracy_deg'] > score_angle[-1]):               
                temp_results_trials['score_angle'] = 0
                break
        
        # Binned by intervals of 1 meter
        score_loc = np.arange(1,11)        
        for l in np.arange(0,len(score_loc)):
            if (temp_results_trials['loc_accuracy'] < score_loc[l]):               
                temp_results_trials['score_loc'] = (10-l)
                tot_loc = tot_loc + temp_results_trials['score_loc']
                break
            elif (temp_results_trials['loc_accuracy'] > score_loc[-1]):
                temp_results_trials['score_loc'] = 0
                break
        
        # store the easy to read results into the main dictionary
        results_trials[dfname] = temp_results_trials
        
    '''___ end of dfname in dfnames for loop ___'''
    
    # Concat the sniffing data into a single data frame
    binned_sniff_df = binned_sniff[df_names[0]]
    sniffpre_all_df = sniffpre_all[df_names[0]]
    sniffpost_all_df = sniffpost_all[df_names[0]]
    for dfname in df_names[1:]:
        binned_sniff_df = pd.concat([binned_sniff_df, binned_sniff[dfname]])
        sniffpre_all_df = pd.concat([sniffpre_all_df, sniffpre_all[dfname]])
        sniffpost_all_df = pd.concat([sniffpost_all_df, sniffpost_all[dfname]])
        
    
    # Find how users performed overall by finding an average result, the overall score, and the total score.
    # Perfect score is 160 each for loc, angle, time
    # Total score also found out of 480
    results_overall['codename']=userdata['1'].codename[0].replace(" ","") 
    results_overall['overall_score_loc'] = tot_loc
    results_overall['overall_score_angle'] = tot_angle
    results_overall['overall_score_time'] = tot_time
    results_overall['overall_score'] = tot_time+tot_angle+tot_loc
    results_overall['avg_score_loc'] = tot_loc/len(df_names)
    results_overall['avg_score_angle'] = tot_angle/len(df_names)
    results_overall['avg_score_time'] = tot_time/len(df_names)
    results_overall_df = pd.DataFrame.from_dict(results_overall,  orient='index')
    results_overall_df = results_overall_df.reindex(['codename','avg_score_loc','avg_score_angle','avg_score_time',
                            'overall_score_loc','overall_score_angle','overall_score_time','overall_score'])
    
    # Convert all the dictionaries into data frames for easier manipulation and return    
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.transpose()
    results_df = results_df.reindex(df_names)
    results_df = results_df[['trial','angle_accuracy_rad', 'final_time_ms', 'loc_accuracy','codename', 'condition',
                            'sex', 'self_game_xp', 'RA_game_xp', 'SOD']]
    
    # code names sometimes have weird spaces in them. This code removes spaces.
    finals['codename']=userdata['1'].codename[0].replace(" ","")
    finals_df = pd.DataFrame.from_dict(finals)
    finals_df = finals_df.transpose()
    finals_df = finals_df.reindex(df_names)
    
    df_names = df_names.append('codename')
    results_trials['codename']=userdata['1'].codename[0].replace(" ","")   
    results_trials_df = pd.DataFrame.from_dict(results_trials)
    # Transpose for legibility
    results_trials_df = results_trials_df.transpose()
    # Reorder the rows so the trials are in increasing order
    results_trials_df = results_trials_df.reindex(df_names)
        
    return finals_df, results_df, results_trials_df, results_overall_df, sniffpre_all_df, sniffpost_all_df, binned_sniff_df

def get_csv_names(folder):
    ''' Pull all CSV names from a folder

    Args:
        folder:     str; path of folder

    Returns: 
        csv_names:  list of str; all file names in folder
    '''
    
    # Read all the names of the files and store as list
    csv_names_temp = os.listdir(folder)
    csv_names = os.listdir(folder)

    for f in csv_names_temp:
        if (f.startswith('.')==True):
            csv_names.remove(f)
        elif '_' in f:
            trial_num = f.split("_")[1]
            if (int(trial_num) < 1):
                csv_names.remove(f)
        elif '_' not in f:
            csv_names.remove(f)
            
    return csv_names
    
def get_df_names(csv_names):
    ''' Pull all trial names from list of file names

    Args:
        csv_names:     list of strs; contains all CSV names for a user

    Returns: 
        df_names:  list of str; trial numbers 1-8
    
    Note: Really don't need this function to loop so many times since we know
    There are always 8 trials.
    '''
    df_names = []
    for i in range(0,len(csv_names)):
        # split the csv_names by "_". The 2nd index holds the trial number
        trial_num = csv_names[i].split("_")[1]
        # temporarily hold the name of the trial#
        name = str(trial_num)
        # append it to the list of all names.
        df_names.append(name)
    df_names = natural_sort(df_names)

    
    return df_names

def single_user(setup, subj_info):
    ''' Opens a user's folder, takes all the trials, sorts them into a dictionary
        Calculates basic statistics of how well the user performed.

    Args:
        setup:     pd df; contains information about all trial set ups
        subj_info: pd df; answers from survey given to subject

    Returns: 
        finals:          pd df; final locations and angles of participant
        results:         pd df; accuracy of final location, angle, and time
        results_trials:  pd df; easy to read data of results and some "scores" for fun
        results_overall: pd df; avg of all trials for location accuracy time, and scores
        sniffpre_all:    pd df; location of sniffs before finding plume
        sniffpost_all:   pd df; location of sniffs before after plume
        sniff_df:       pd df; sniffs binned by distance from plume origin
        
    '''

    
     # Open folder of a single participant pathway from a GUI
    folder = open_folder()

    # Open folder from a pathway
    # folder = os.getcwd()+"/data/2016-11-01/data/eajo63"

    # # -- For testing purposes, this section opens a single file --
    # # Open from a GUI
    # trial = open_file()
    # # Open specific file
    # trial = pd.read_csv('/Users/judyjinn/Python/Human Olfaction/VR/data/2016-10-24/data/48JO40_6_2016-10-24_1151.csv', sep=',')


    # Read all the names of the files and store as list
    csv_names = get_csv_names(folder)

    # -- Create a list of names for trial numbers and store as a list to be used in the dataframes --
    # This was done because the csv_names list does not list the trials in exactly incrementing numbers
    # Thus the future dataframe names must be made in the same order to match up with the correct file data.

    # Create an empty list to store dataframe names for each trial
    df_names = get_df_names(csv_names)
    # print(df_names)
    
    # -- Create a dictionary of dataframes to store all the data for a single participant --

    # Create a list of header names to be used for all the dataframes.
    
    userdata = org_data(folder, csv_names, df_names, setup)

    results_folder = folder[:-6]+userdata['codename']+"_results"
    if not os.path.exists(results_folder):
        os.makedirs(folder[:-6]+userdata['codename']+"_results")

    # -- Get some  stats --
    finals, results, results_trials, results_overall, sniffpre_all, sniffpost_all, sniff_df = final_stats(userdata, df_names, setup, subj_info)

    results_combined = pd.concat([results_overall, results_trials])
    results_combined.to_csv(results_folder+"/"+results['codename'][0]+"_results_combined.csv", sep=',')

    results_overall = results_overall.transpose()
    
    

    #animate_walk
    # animate_walk(userdata, full_setup, 3,3, 1, results_folder)

    #Final graphs, no animation. Used for saving
    # just_graph(folder, userdata, full_setup, 8,8)
    return finals, results, results_trials, results_overall, userdata, folder, sniffpre_all, sniffpost_all, sniff_df
    
def check_missing(folder): 
    ''' To save time, the program checks only for subjects where their final
        stats have not been calculated or have missing graphs. This function 
        makes a list of all users which need to be graphed or analyzed

    Args:
        folder:     str; path to folder containing all users

    Returns: 
        all_files:      list of str; names of all users
        missing_graphs: list of str; names of users missing graphs
        missing_files:  list of str; names of users who have not been analyzed
    
    '''
    
    all_files = {}
    missing_graphs = {}
    missing_files = {}
    
    # Always iterate through the temp file and remove from the other same named variable.
    # Otherwise when iterating it will tend to skip ahead because suddenly an index was removed so all index shifted -1
    # But the loop itself is still counting +1 from where it was before.
    date_folders_temp = os.listdir(folder)
    date_folders = os.listdir(folder)
    
    if os.path.isfile(os.getcwd()+'/final_results.csv'):
        final_results = pd.read_csv(os.getcwd()+'/final_results.csv', sep=',')         
        existing_names = final_results['codename'].unique()
        existing_names = [n.lower() for n in existing_names]
    else:
        existing_names=[]

    for f in date_folders_temp:
        if (f.startswith('.')==True):
            date_folders.remove(f)
        elif '_' not in f:
            date_folders.remove(f)
            

    for date in date_folders:
        all_files[date] = {}
    
    # final_results = pd.read_csv(os.getcwd()+'/final_results.csv', sep=',')
    # old_names = final_results['codename'].unique()

    user_names = []  
    missing_g = [] 
    for date in date_folders:
        # print(date)
        
        user_names = (os.listdir(folder+"/"+date+"/data"))
        user_names_temp = user_names   
            
        # Only get the user names
        for f in user_names_temp:
            if ("_" in f):
                user_names.remove(f)
     
        all_files[date] = user_names
        
        # Now go back and check if there were some files which were not analyzed for this date
        missing_f_temp = (os.listdir(folder+"/"+date+"/data"))
        missing_f = missing_f_temp
        missing_g_temp = (os.listdir(folder+"/"+date+"/data"))
        missing_g = missing_g_temp
        
        # Remove the hidden folder first
        for f in missing_f_temp:
            if (f.startswith('.')==True):
                missing_f.remove(f)
        for g in missing_g_temp:
            if (g.startswith('.')==True):
                missing_g.remove(g)
        
        # print(missing_f)

        # Now check each name for an existing results folder which contains data
        for name in user_names:
            # print(name)
            
            results_folder = folder+"/"+date+"/data/"+name+"_results"
            graph_folder = results_folder+"/"+name+"_graphs/"
            
            # remove the file names which already exist in the old data frame
            if ((os.path.isfile(results_folder+'/'+name+'_results_combined.csv')==True) & \
                (name.lower() in existing_names)
                ):
                
                missing_f.remove(name+"_results")
                missing_f.remove(name)
            # If the csv file does not exist but for some reason the results folder exists, mark as missing
            elif (os.path.exists(results_folder) & \
                (os.path.isfile(results_folder + '/' + name + \
                    '_results_combined.csv')==False)
                ):
                
                missing_f.remove(name+"_results")
            # If the csv file exists but has not been concatenated into the existing final_results.csv also mark as missing
            elif ((os.path.isfile(results_folder + '/' + name + '_results_combined.csv')==True) & \
                (name.lower() not in existing_names)
                ):
                missing_f.remove(name+"_results")
            # If the entire folder is missing, then do nothing, and the name should be retained.
                
            # for graphs, remove the names which have a graph folder, or only remove the results name if graphs are missing.
            if (os.path.exists(graph_folder)):
                missing_g.remove(name+"_results")
                missing_g.remove(name)
            elif (os.path.exists(results_folder)):
                missing_g.remove(name+"_results")
                    
                
        # After checking all the names, see if missing_g contains anything. If it does, store the missing file names
        # under their respective date

        if (bool(missing_g)==True):
            missing_graphs[date] = missing_g
            # print("MISSING GRAPHS " , missing_g)
        if (bool(missing_f)==True):
            missing_files[date] = missing_f
            # print("MISSING FILES ", missing_f)
    
    return all_files, missing_graphs, missing_files

def concat_all(setup, subj_info, used_files):
    ''' Takes all users, analyzes their data, graphs their paths then concatenates
        all data from all users in a huge data frame. Saves final data as CSVs

    Args:
        setup:      pd df; contains informatiion about each trial's set up
        subj_info:  pd df; survey answers from all users
        use_files:  str; 'all' or 'missing'. To tell function whether to reanalyze
                    all users or to only look for users missing graphs or anlyses

    Returns:
        all_files:          list of str; names of all users
        allresults_dict:    dict;  all users accuracy of final location, angle, and time
        missing_graphs:     list of str; names of users missing graphs
        missing_files:      list of str; names of users who have not been analyzed
        alldata_dict:       dict; contains all users and all trial raw data
        final_results:      pd df; accuracy of final location, angle, and time
        folder:             str; path to main folder
        sniffpre_all:       pd df; location of sniffs before finding plume
        sniffpost_all:      pd df; location of sniffs before after plume
        binned_sniff_dict:  dict; sniffs binned by distance from plume origin
        binned_sniff:       pd df; sniffs binned by distance from plume origin

    '''
    
    folder = open_folder()
    
    all_files, missing_graphs, missing_files = check_missing(folder) 
    
    if (used_files=='all'):
        date_folders = list(all_files.keys()) # use if you want to redo all data
        print("looking through all files")
    elif (used_files == 'missing'):
        date_folders = list(missing_files.keys())     # use if you want to work on the missing data
        print("Only looking for missing files")
        if (bool(missing_files)==False):
            print('No missing files!')
            allresults_dict = {}
            alldata_dict = {}
            sniffpre_all={}
            sniffpost_all={}
            sniff_df={}
            final_results = pd.read_csv(os.getcwd()+'/final_results.csv', sep=',')
            sniffpre_all = pd.read_csv(os.getcwd()+'/sniffpre_all.csv', sep=',')
            sniffpost_all = pd.read_csv(os.getcwd()+'/sniffpost_all.csv', sep=',')
            binned_sniff = pd.read_csv(os.getcwd()+'/binned_sniff.csv', sep=',')
            
            return all_files, allresults_dict, missing_graphs, missing_files, alldata_dict, final_results, \
            folder, sniffpre_all, sniffpost_all, sniff_df, binned_sniff
    
    # Create giant dict containing all data
    alldata_dict = {}
    allresults_dict = {}
    sniffpre_all_dict = {}
    sniffpost_all_dict = {}
    binned_sniff_dict = {}
    for date in date_folders:
        print(date)
        if (used_files=='all'):
            user_folders = all_files[date]    # use if you want to redo all data
        elif (used_files == 'missing'):
            user_folders = missing_files[date]        # use if you want to work on the missing data
        name_temp = []
        for name in user_folders:
            name_temp.append(name)
            csv_folder = folder+"/"+date+"/data/"+name
            csv_names = get_csv_names(csv_folder)
            df_names = get_df_names(csv_names)
            userdata = org_data(csv_folder, csv_names, df_names, setup)

            results_folder = folder+"/"+date+"/data/"+userdata['codename']+"_results"
            if not os.path.exists(results_folder):
                os.makedirs(folder+"/"+date+"/data/"+userdata['codename']+"_results")

            finals, results, results_trials, results_overall, sniffpre_all_df, sniffpost_all_df, binned_sniff_df, \
            = final_stats(userdata, df_names, setup, subj_info)
            
            alldata_dict[name] = userdata
            allresults_dict[name] = results
            sniffpre_all_dict[name] = sniffpre_all_df
            sniffpost_all_dict[name] = sniffpost_all_df
            binned_sniff_dict[name] = binned_sniff_df

            results_combined = pd.concat([results_overall, results_trials])
            results_combined.to_csv(results_folder+"/"+results['codename'][0]+"_results_combined.csv", sep=',')
            results_overall = results_overall.transpose()

    all_names = list(allresults_dict.keys())

    if bool(all_names)==True:
        temp_allresults_df = pd.DataFrame.from_dict(allresults_dict[all_names[0]])
        temp_binned_sniff_df = binned_sniff_dict[all_names[0]]
        temp_sniffpre_all_df = sniffpre_all_dict[all_names[0]]
        temp_sniffpost_all_df = sniffpost_all_dict[all_names[0]]

        for name in all_names[1:]:
            temp_allresults_df = pd.concat([temp_allresults_df, pd.DataFrame.from_dict(allresults_dict[name])])
            temp_binned_sniff_df = pd.concat([temp_binned_sniff_df, binned_sniff_dict[name]])
            temp_sniffpre_all_df = pd.concat([temp_sniffpre_all_df, sniffpre_all_dict[name]])
            temp_sniffpost_all_df = pd.concat([temp_sniffpost_all_df, sniffpost_all_dict[name]])

            temp_allresults_df.to_csv("temp_allresults.csv", sep=',',index=False)
            temp_binned_sniff_df.to_csv("temp_binned_sniff.csv", sep=',', index=False)
            temp_sniffpre_all_df.to_csv("temp_sniffpre_all.csv", sep=',', index=False)
            temp_sniffpost_all_df.to_csv("temp_sniffpost_all.csv", sep=',', index=False)

        # Check for an existing file called final_results which is the concatenated dataset.
        if os.path.isfile(os.getcwd()+'/final_results.csv'):
            final_results = pd.read_csv(os.getcwd()+'/final_results.csv', sep=',')
            final_results = final_results.append(temp_allresults_df)
            final_results.to_csv("final_results.csv", sep=',',index=False)
        else:
            final_results = temp_allresults_df
            temp_allresults_df.to_csv("final_results.csv", sep=',',index=False)
            
        if os.path.isfile(os.getcwd()+'/sniffpre_all.csv'):
            sniffpre_all = pd.read_csv(os.getcwd()+'/sniffpre_all.csv', sep=',')
            sniffpre_all = sniffpre_all.append(temp_sniffpre_all_df)
            sniffpre_all.to_csv("sniffpre_all.csv", sep=',',index=True)
        else:
            sniffpre_all= temp_sniffpre_all_df
            temp_sniffpre_all_df.to_csv("sniffpre_all.csv", sep=',',index=True)
        
        if os.path.isfile(os.getcwd()+'/sniffpost_all.csv'):
            sniffpost_all = pd.read_csv(os.getcwd()+'/sniffpost_all.csv', sep=',')
            sniffpost_all = sniffpost_all.append(temp_sniffpost_all_df)
            sniffpost_all.to_csv("sniffpost_all.csv", sep=',',index=True)
        else:
            sniffpost_all= temp_sniffpost_all_df
            temp_sniffpost_all_df.to_csv("sniffpost_all.csv", sep=',',index=True)


        if os.path.isfile(os.getcwd()+'/binned_sniff.csv'):
            binned_sniff = pd.read_csv(os.getcwd()+'/binned_sniff.csv', sep=',')
            binned_sniff = binned_sniff.append(temp_binned_sniff_df)
            binned_sniff.to_csv("binned_sniff.csv", sep=',',index=False)
        else:
            binned_sniff= temp_binned_sniff_df
            temp_binned_sniff_df.to_csv("binned_sniff.csv", sep=',',index=False)
        

        return all_files, allresults_dict, missing_graphs, missing_files, alldata_dict, final_results, folder, \
        sniffpre_all,  sniffpost_all, binned_sniff_dict, binned_sniff
    else:
        final_results = pd.read_csv(os.getcwd()+'/final_results.csv', sep=',')

        return all_files, allresults_dict, missing_graphs, missing_files, alldata_dict, final_results, folder, \
        sniffpre_all, sniffpost_all, binned_sniff_dict, binned_sniff

'''
For all participants
(mono-stereo)/std
Angles
        effect size = 0.015002728057676001
        power = 0.05
Time
        -0.1304888940603742
        power = 0.114
Location
         0.0888528729415939
        power = 0.17
'''

def sniff_stats(allresults_df, sniffpost_all, binned_sniff):
    ''' Calculates basic statistics from the processed data and prints results

    Args:
        allresults_df:      pd df; accuracy of final location, angle, and time
        sniffpost_all:      pd df; location of sniffs before after plume
        binned_sniff:       pd df; sniffs binned by distance from plume origin

    Returns: 
        None    
    '''
    
    overall_finals = allresults_df.groupby(['condition']).describe()
    all_mono = allresults_df[allresults_df['condition']=='mono']
    all_stereo = allresults_df[allresults_df['condition']=='stereo']
    print('Number of participants in mono condition', len(all_mono['codename'].unique()))
    print('Number of participants in stereo condition', len(all_stereo['codename'].unique()))
    
    
    ## Separated by "gamers" those who had no difficulty moving in game
    gamers = allresults_df[allresults_df['RA_game_xp']==3]
    gamer_finals = gamers.groupby(['condition']).describe().reset_index()
    print(gamer_finals)
    
    gamer_mono = gamers[gamers['condition']=='mono']
    gamer_stereo = gamers[gamers['condition']=='stereo']
    print('Number of gamers in mono condition', len(gamer_mono['codename'].unique()))
    print('Number of gamers in stereo condition', len(gamer_stereo['codename'].unique()))
    f_val, p_val = stats.f_oneway(gamer_mono['angle_accuracy_rad'], gamer_stereo['angle_accuracy_rad'])
    print('Gamers Final Angle Accuracy ANOVA, mono vs stereo, F:', f_val, ' p:', p_val)
    f_val, p_val = stats.f_oneway(gamer_mono['final_time_ms'], gamer_stereo['final_time_ms'])
    print('Gamers Final Time ANOVA, mono vs stereo, F:', f_val, ' p:', p_val)
    f_val, p_val = stats.f_oneway(gamer_mono['loc_accuracy'], gamer_stereo['loc_accuracy'])
    print('Gamers Distance from Source ANOVA, mono vs stereo, F:', f_val, ' p:', p_val)
    
    
    mod = ols('loc_accuracy ~ condition', data=gamers).fit()
    sm.stats.anova_lm(mod, typ=1) 
    
    
    ## Separated by high SOD participants
    SOD_hi = allresults_df[allresults_df['SOD']>=5]
    SOD_lo = allresults_df[allresults_df['SOD']<5]
    # gamer_finals = SOD_hi.groupby(['condition']).describe().reset_index()
    # print(gamer_finals)
    
    # gamer_mono = gamers[gamers['condition']=='mono']
    # gamer_stereo = gamers[gamers['condition']=='stereo']
    # print('Number of gamers in mono condition', len(gamer_mono['codename'].unique()))
    # print('Number of gamers in stereo condition', len(gamer_stereo['codename'].unique()))
    # f_val, p_val = stats.f_oneway(gamer_mono['angle_accuracy_rad'], gamer_stereo['angle_accuracy_rad'])
    # print('Gamers Final Angle Accuracy ANOVA, mono vs stereo, F:', f_val, ' p:', p_val)
    # f_val, p_val = stats.f_oneway(gamer_mono['final_time_ms'], gamer_stereo['final_time_ms'])
    # print('Gamers Final Time ANOVA, mono vs stereo, F:', f_val, ' p:', p_val)
    # f_val, p_val = stats.f_oneway(gamer_mono['loc_accuracy'], gamer_stereo['loc_accuracy'])
    # print('Gamers Distance from Source ANOVA, mono vs stereo, F:', f_val, ' p:', p_val)
    
    
    
    # Turning angle based on distance/R/L side
    stereo=sniffpost_all[sniffpost_all['condition']=='stereo']
    mono=sniffpost_all[sniffpost_all['condition']=='mono']
    
    stereo_angle = pd.concat([stereo['density_l'],stereo['density_r'], stereo['angle2source']],axis=1)
    stereo_angle['angle2source'] = np.rad2deg(stereo_angle['angle2source'])
    stereo_angle['l-r']= np.abs(stereo_angle['density_l']-stereo_angle['density_r'])
    stereo_angle['angle_change']=stereo_angle['angle2source'].diff()
    
    ggplot(aes(x=stereo_angle['l-r'], y=stereo_angle['angle_change']), data=stereo_angle) + \
            geom_point() +stat_smooth(method='nls', span=0.2)  +\
            ggtitle('Male Mass by Generation and Environment Effects') +\
            xlab('Magnitude of Difference between Left and Right Ear') +\
            ylab("Magnitude of Angle Change (degrees)")
    

    return
    
def plots(allresults_df, sniffpost_all, binned_sniff):
    ''' Graphs and saves

    Args:
        allresults_df:      pd df; accuracy of final location, angle, and time
        sniffpost_all:      pd df; location of sniffs before after plume
        binned_sniff:       pd df; sniffs binned by distance from plume origin

    Returns: 
        None    
    '''

    sniff_rm = sniffpost_all[
        ['codename','trial', 'condition', 'x', 'z', 
        'angle','distance', 'density_l', 'density_r', 'density_both',
        'l-r'], 'sound_change', 'angle_change'], 'dist_change'],
        'x_diff','z_diff']
        ]
    stereo_sniff=sniff_rm[sniff_rm['condition']=='stereo']
    mono_sniff=sniff_rm[sniff_rm['condition']=='mono']
    # #
    #
    # # Graphing the spread of angle changes that occur as function of distance to the odor source
    # dist_angle = ggplot(aes(x='distance', y='angle_change'), data=sniffpost_all) + \
    #        geom_point()+ theme_bw() +\
    #        ggtitle('Angle Corrections by Distance from Source') +\
    #        xlab("Distance to Odor Source", size=18) +\
    #        ylab("Angle Change (degrees)") +\
    #        scale_x_continuous(limits=(0,30)) #+ theme(axis_title_x  = element_text(size = 18))
    # # dist_angle.save('dist_angle.png')
    #
    
    # right = sniffpost_all[sniffpost_all['angle_change']<0]
    # left = sniffpost_all[sniffpost_all['angle_change']>0]
    # fig = plt.figure()
    # ax = fig.gca()
    # plt.style.use('ggplot')
    # ax.scatter(right['distance'], right['angle_change'], marker='>', color='crimson',label='Right Turn')
    # ax.scatter(left['distance'], left['angle_change'], marker='<', color='gray',label='Left Turn')
    # # fig.suptitle('Angle Corrections by Distance from Source', fontsize=20)
    # plt.xlabel('Distance to Odor Source', fontsize=20)
    # plt.ylabel('Angle Change (degrees)', fontsize=20)
    # plt.tick_params(axis='both', which='major', labelsize=15)
    # # plt.tick_params(axis='both', which='minor', labelsize=8)
    # ax.set_axis_bgcolor('white')
    # ax.grid(color='lightgray', linestyle='-', linewidth=1)
    #
    # plt.legend(loc='lower right').draggable()
    #
    # fig.savefig('dist_angle.png')
    # plt.show()

    #

    
    #

    #
    #
    #
    # ##  Graphing the distance of the user to the odor source in relation to the changing levels of sound.
    # def concentration(data):
    #    if sniff_rm['sound_change'] == 0:
    #        val = 'No Change'
    #    elif sniff_rm['sound_change'] > 0:
    #        val = 'Increase'
    #    else:
    #        val = 'Decrease'
    #    return val


    sound_dist_sub = pd.concat(
        [sniffpost_all['density_both'], 
        sniff_rm[['sound_change', 'distance', 'dist_change']], axis=1
        )

    sound_dist_sub = sound_dist_sub[sound_dist_sub['sound_change']!=0]
    sound_dist_sub['sound_change']  = sound_dist_sub['sound_change']*100
    sound_dist_sub['Concentration'] = np.sign(sound_dist_sub['sound_change'])
    sound_dist_sub['Concentration'] = sound_dist_sub['Concentration'].apply(str)
    sound_dist_sub['Concentration'] = \
        sound_dist_sub['Concentration'].replace(
            ['-1.0', '0.0', '1.0'], ['Decrease', 'No Change', 'Increase']
            )
    #
    # sound_distance = ggplot(aes(x='sound_change', y='distance', color='Concentration'), data=sound_dist_sub) + \
    #        geom_point()+ theme_bw()  +\
    #        ggtitle('Participant Distance to Goal In Response to Sampling') +\
    #        xlab("Percent Change of Concentration") +\
    #        ylab("Distance to Odor Source") +\
    #        scale_x_continuous(limits=(-100,100)) +\
    #        scale_y_continuous(limits=(0,30))
    # sound_distance.save('sound_distance.png')
    #
    sound_dist_sub['Distance'] = np.sign(sound_dist_sub['dist_change'])
    sound_dist_sub['Distance'] = sound_dist_sub['Distance'].apply(str)
    sound_dist_sub['Distance'] = \
        sound_dist_sub['Distance'].replace(
            ['-1.0', '0.0', '1.0'], ['Closer', 'No Change', 'Farther']
            )
    sound_dist_sub = sound_dist_sub[sound_dist_sub['dist_change']!=0]
    
    sound_dist_sub.dropna(axis=0, inplace=True)
    #
    # sound_dis_change=ggplot(aes(x='sound_change', y='dist_change', shape= 'Concentration', color='Distance'), data=sound_dist_sub) + \
    #     geom_point() + theme_bw()  +\
    #     ggtitle('Participant Changes in Distance to Goal In Response to Sampling') +\
    #     xlab("Percent Change of Concentration") +\
    #     ylab("Distance Change to Odor Source") +\
    #     scale_x_continuous(limits=(-100,100)) +\
    #     scale_y_continuous(limits=(-12,12))
    # sound_dis_change.save('sound_dis_change.png')
    

    neg_far= sound_dist_sub[sound_dist_sub['sound_change']<0]
    neg_far = neg_far[neg_far['dist_change']>0]
    neg_close = sound_dist_sub[sound_dist_sub['sound_change']<0]
    neg_close = neg_close[neg_close['dist_change']<0]
    pos_far= sound_dist_sub[sound_dist_sub['sound_change']>0]
    pos_far = pos_far[pos_far['dist_change']>0]
    pos_close = sound_dist_sub[sound_dist_sub['sound_change']>0]
    pos_close = pos_close[pos_close['dist_change']<0]

    fig = plt.figure()
    ax = fig.gca()
    plt.style.use('ggplot')
    ax.set_xlim(-100,100)
    ax.set_ylim(-12,12)
    ax.set_axis_bgcolor('white')
    plt.subplots_adjust(top=0.95)
    plt.axes().set_aspect(5)

    ax.scatter(
        neg_far['sound_change'], neg_far['dist_change'], 
        marker="v", facecolors='white', edgecolors='gray', 
        label='Concentration Decrease, Moved Away'
        )
    ax.scatter(
        neg_close['sound_change'], neg_close['dist_change'], 
        marker="^", facecolors='white', edgecolors='crimson', 
        label='Concentration Decrease, Moved Closer'
        )
    ax.scatter(
        pos_far['sound_change'], pos_far['dist_change'], 
        marker="v", color='gray', label='Concentration Increased, Moved Away'
        )
    ax.scatter(
        pos_close['sound_change'], pos_close['dist_change'], 
        marker="^", color='crimson', 
        label='Concentration Increased, Moved Closer'
        )
        
        
    fig.suptitle('Participant Changes in Distance to Goal In Response to Sampling', fontsize=20)
    plt.xlabel('Percent Change of Concentration', fontsize=20)
    plt.ylabel('Distance Change to Odor Source', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    # plt.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_axis_bgcolor('white')
    ax.grid(color='lightgray', linestyle='-', linewidth=1)

    plt.legend(loc='lower right').draggable()
    fig.tight_layout()
    fig.savefig('dist_angle.png')
    plt.show()
    # plt.close('all')

    
    
    
    

    #
    # sound_dis_change=ggplot(aes(x='l-r', y='angle_change'), data=sniff_rm) + \
    #     geom_point() + theme_bw()  +\
    #     ggtitle('Participant Changes in Distance to Goal In Response to Sampling') +\
    #     xlab("Percent Change of Concentration") +\
    #     ylab("Distance Change to Odor Source") +\
    #     scale_x_continuous(limits=(-100,100)) +\
    #     scale_y_continuous(limits=(-12,12))
    # sound_dis_change.save('sound_dis_change.png')
    #
    

    # Sampling rate
    avg = binned_sniff.groupby('bin_dist').mean().reset_index()
    std = binned_sniff.groupby(['bin_dist']).std().reset_index()
    std.columns = ['bin_dist_std', 'num_sniff_std', 'sniff_rate_std', 'trial_std',
       'plume_found_frame_std', 'plume_found_percent_std']
    binned_avg = pd.concat([avg,std], axis=1)

    # low = binned_avg['sniff_rate']-binned_avg['sniff_rate_std']
    # hi = binned_avg['sniff_rate']+binned_avg['sniff_rate_std']
    #
    # sample_rate_all = ggplot(aes(x='bin_dist', y='sniff_rate'), data=binned_avg) + \
    #        geom_line()+ theme_bw()   +\
    #        ggtitle('Average Sampling Rate at Distance from Odor Source') +\
    #        xlab("Distance (m)") +\
    #        ylab("Sample rate (ms)") +\
    #        scale_x_continuous(breaks=(range(0,11,1)), limits=(0,11)) +\
    #        scale_y_continuous(limits=(400,900))
    # sample_rate_all.save('sample_rate_all.png')

    avg=  binned_sniff.groupby(['bin_dist', 'condition']).mean().reset_index()
    std = binned_sniff.groupby(['bin_dist', 'condition']).std().reset_index()
    std.columns = [
        'bin_dist_std', 'condition_std', 'num_sniff_std', 
        'sniff_rate_std', 'trial_std',
       'plume_found_frame_std', 'plume_found_percent_std'
       ]
    bin_avg_cond = pd.concat([avg,std], axis=1)

    #
    # sample_rate_condition = ggplot(aes(x='bin_dist', y='sniff_rate', color='condition'), data=bin_avg_cond) + \
    #        geom_line()+ theme_bw()   +\
    #        ggtitle('Average Sampling Rate at Distance from Odor Source') +\
    #        xlab("Distance (m)") +\
    #        ylab("Sample rate (ms)") +\
    #        scale_x_continuous(breaks=(range(0,11,1)), limits=(0,11)) +\
    #        scale_y_continuous(limits=(400,900))
    # # sample_rate_condition.save('sample_rate_condition.png')

    #
    # bin_mono=bin_avg_cond[bin_avg_cond['condition']=='mono']
    # bin_stereo=bin_avg_cond[bin_avg_cond['condition']=='stereo']
    # #
    # fig = plt.figure()
    # ax = fig.gca()
    # plt.style.use('ggplot')
    # # plt.hold(True)
    # ax.plot( bin_mono['bin_dist'],bin_mono['sniff_rate'], color='crimson', linewidth=2, label='mono')
    # ax.plot( bin_stereo['bin_dist'], bin_stereo['sniff_rate'],color='gray', linewidth=2, label='stereo')
    # plt.xlabel('Distance to Odor Source', fontsize=20)
    # plt.ylabel('Sample Rate (ms)', fontsize=20)
    # plt.tick_params(axis='both', which='major', labelsize=15)
    # # plt.tick_params(axis='both', which='minor', labelsize=8)
    # ax.set_axis_bgcolor('white')
    # ax.grid(color='lightgray', linestyle='-', linewidth=1)
    # ax.legend(loc='lower left').draggable()
    #
    # fig.savefig('sample_rate_condition.png')
    # plt.show()
    # plt.close('all')


    #
    # # Number of Sniffs by Distance
    # sample_num_condition = ggplot(aes(x='bin_dist', y='num_sniff', color='condition'), data=bin_avg_cond) + \
    #        geom_line()+ theme_bw()   +\
    #        ggtitle('Average Sample Number at Distance from Odor Source') +\
    #        xlab("Distance (m)") +\
    #        ylab("Number of Samples") +\
    #        scale_x_continuous(breaks=(range(0,11,1)), limits=(0,11)) +\
    #        scale_y_continuous(limits=(0,25))
    # sample_num_condition.save('sample_num_condition.png')
    #
    #
    #
    # final_avg=final_results.groupby(['trial', 'condition']).mean().reset_index()
    # final_dist = ggplot(aes(x='trial', y='loc_accuracy', color='condition'), data=final_avg) + \
    #        geom_line()+ theme_bw()    +\
    #        ggtitle('Final Distance to Odor Source') +\
    #        xlab("Distance (m)") +\
    #        ylab("Distance to Odor Source (m)")  +\
    #        scale_x_continuous(breaks=(range(1,8,1)), limits=(1,8)) +\
    #        scale_y_continuous(limits=(0.0,4.0))
    # final_dist.save('final_dist.png')
    
    
    # final_mono = final_results[final_results['condition']=='mono']
    # final_mono_8 = final_mono[final_mono['trial']==8]
    # final_stereo = final_results[final_results['condition']=='stereo']
    # final_stereo_8 = final_stereo[final_stereo['trial']==8]
    # print('Number of gamers in mono condition', len(final_mono_8['codename'].unique()))
    # print('Number of gamers in stereo condition', len(final_stereo_8['codename'].unique()))
    # stats.ttest_ind(final_mono_8['loc_accuracy'], final_stereo_8['loc_accuracy'], equal_var=True)
    
    
    
    
    

    #
    # plt.show()
    # plt.close()
    # # root.destroy()



    
    return


'''                     #######################
#--------------------   ## ---    MAIN   --- ##     ----------------------------
                        #######################
'''

if __name__ == '__main__':
    # If you want to move this entire section up to the top above all the other functions for readability
    # Move return to the end of the script and move the etnire MAIN section.
    # Also uncomment main class
    def main():
        return


    # -- Open the set up data so the program knows where to find start location, plume location, etc. --
    # pilot_setup =  pd.read_csv('/Users/judyjinn/Python/Human Olfaction/VR/full_setup.csv', sep=',')
    full_setup =  pd.read_csv('/Users/judyjinn/Google Drive/Labs/Jacobs/Olfaction/VR/VR Analysis/full_setup.csv', sep=',')
    subj_info = pd.read_csv('/Users/judyjinn/Google Drive/Labs/Jacobs/Olfaction/VR/VR Analysis/VRdata_information2017_05_08.csv', sep=',')

    # Flip the likert scale for the negative sense of direction questions
    # This is because a higher number should mean better sense of direction. Thus, all the negative SODs should be low #s.
    subj_info.ix[:,['q2','q6','q8','q10','q11','q12','q13','q15']] = \
         8-subj_info.ix[:,['q2','q6','q8','q10','q11','q12','q13','q15']]
    # Get mean rating of SOD (higher = better)
    subj_info['SOD'] = subj_info.ix[:,'q1':'q15'].mean(axis=1)
    # Make names lowercase
    subj_info['Codename'] = subj_info['Codename'].str.lower()

    temp = []
    for i in range(0, len(full_setup['plume_angle_x'])):
        plume_angle_x = full_setup.plume_angle_x.iloc[i]
        plume_angle_z = full_setup.plume_angle_z.iloc[i]
        plume_rad = math.acos(plume_angle_x) 
    
        if (plume_angle_z<0):
            plume_rad = (2*np.pi)-plume_rad
        temp.append(plume_rad)
    full_setup['plume_rad'] = temp

    ## -- Opening a set of data from a single participant  --
    ## Use this code to also animate graphs
    # finals, results, results_trials, results_overall, userdata, folder, sniffpre_all, \
    #     sniffpost_all, sniff_df = single_user(full_setup, subj_info)

    ## -- Opening all files to concatenate new participants  --
    # concat_all: change between 'all' to redo all files, or only 'missing' to only cycle through new participants
    all_files, allresults_dict, missing_graphs, missing_files, alldata_dict, final_results, folder, \
    sniffpre_all, sniffpost_all, binned_sniff_dict, binned_sniff = concat_all(full_setup, subj_info, 'missing')


    # # # Save graphs of all data
    # just_graph(folder, all_files, full_setup, 1, 8)
    
    ## Run stats
    # sniff_stats(allresults_df, sniffpost_all, binned_sniff)
    
    ## Plot
    plots(final_results, sniffpost_all, binned_sniff)
    
    
    
    
    
    
    # sniffpost_all.columns=['frame', 'sniff_rate', 'x', 'z', 'distance', 'bin_dist', 'angle',
    #    'density_l', 'density_r', 'density_both', 'angle_acc', 'angle2source',
    #    'trial', 'codename', 'plume_found_frame', 'plume_found_percent',
    #    'condition', 'angle2source_deg', 'l-r', 'sound_change', 'angle_change',
    #    'dist_change', 'x_diff', 'z_diff']
    #
    # sniffpost_all.groupby('codename')['frame'].max()
        
        
        
        
        
        
        
        

    # # Sanity check to make sure the number of participants matches the allresults.csv
    # num_subj = 0
    # for i in list(all_files.values()):
    #     num_subj += len(i)
    #
    # num_subj2 = len(allresults_df['codename'].unique())
    # if num_subj2 == num_subj:
    #     print("All good, yo")
    # else:
    #     print("You done fucked up now, girl")
    #
    
    
    
    
    

    #
    #
    # # ----------------------------------------   !! --- Test Zone --- !!     -----------------------------------------
  
  