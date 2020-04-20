#!Les Warren @warre112
# April 18, 2020
# Lab 11, ABE 65100

#This script is desgined to use code from lab 10 and make presentation graphics for the two imput datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
   

    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    for i in range(0,len(DataDF)-1):
        if 0 > DataDF['Discharge'].iloc[i]:
            DataDF['Discharge'].iloc[i]=np.nan
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    DataDF = DataDF[startDate:endDate] #search for data
 
    MissingValues = DataDF["Discharge"].isna().sum() #missing values
    return( DataDF, MissingValues )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    colnames = ['site_no','Mean Flow','Coeff Var','Tqmean','R-B Index']
    monthdata= DataDF.resample('MS').mean() 

    MoDataDF = pd.DataFrame(0, index=monthdata.index, columns=colnames)
    
    MoDataDF['site_no']=DataDF.resample('MS')['site_no'].mean()
  
    MoDataDF['Mean Flow']=DataDF.resample('MS')['Discharge'].mean()
   
    MoDataDF['Coeff Var'] = (DataDF.resample('MS')['Discharge'].std()/ 
            DataDF.resample('MS')['Discharge'].mean())*100

    
    return ( MoDataDF )
def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
   
    colNames=['site_no','Mean Flow','Coeff Var','TQmean','R-B Index']
    MonthlyAverages=pd.DataFrame(0,index=range(1,13),columns=colNames)
    
    
    for n in range(0,12):
        MonthlyAverages.iloc[n,0]=MoDataDF['site_no'][::12].mean() #loop so that code is not needed for each variable
    
    
    index=[(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),(9,0),(10,1),(11,2)]
    
    for (n,m) in index:
        MonthlyAverages.iloc[n,1]=MoDataDF['Mean Flow'][m::12].mean() #mean every 12 months 
   
    
    return( MonthlyAverages )

def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    DataDF = pd.read_csv(fileName, header=0, delimiter=',', parse_dates=['Date'], comment='#', index_col=['Date'])
    
    return( DataDF )


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

##Code from Lab 10 to Read in Data and Dataframes
if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }

    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    DataDF = {} #blank dictionaries 
    MoDataDF = {}
    MonthlyAverages = {}
    MissingValues = {}
    
    for file in fileName.keys():
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
##Read in new files from Lab 10   
    AnnualMetrics=ReadMetrics('Annual_Metrics.csv') 
    MonthlyMetrics=ReadMetrics('Monthly_Metrics.csv')
    tippe=AnnualMetrics[AnnualMetrics['Station']=='Tippe'] #Sort by stattion name -> Tippecanoe rows only
    wildcat=AnnualMetrics[AnnualMetrics['Station']=='Wildcat']#Sort by stattion name -> Wildcat rows only


#Figures
#Daily Stream Flow Plot
    Wild5= DataDF['Wildcat']['2014-10-01' : '2019-09-30'] #Last 5 years
    Tippe5= DataDF['Tippe']['2014-10-01' : '2019-09-30'] #last 5 years 
    plt.figure(figsize=(16,10)) #custom size for better view
    plt.subplot(211)
    plt.plot(Tippe5['Discharge'], 'black',label = 'Tippecanoe')
    plt.ylabel('Discharge (cfs)')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(Wild5['Discharge'], 'blue',label = 'Wildcat')
    plt.xlabel('Date')
    plt.ylabel('Discharge (cfs)')
    plt.legend(loc='upper right') #adding legend location 
    plt.savefig('5 Year Daily Flow.png', dpi=96) # save the plot as PNG resolution of 96 dpi
    plt.close()

#Annual Coefficent  Plot
    fig = plt.figure(figsize=(16,10)) #custom figure size for better resolution
    plt.plot(tippe['Coeff Var'],'black',linestyle='None',marker='.',label='Tippecanoe')
    plt.plot(wildcat['Coeff Var'],'blue', linestyle='None',marker='*',label='Wildcat')
    plt.legend([riverName['Wildcat'],riverName['Tippe']], loc='best',edgecolor='k',fontsize=20)
    plt.xlabel("Year",fontsize=20)
    plt.ylabel("Coefficient of Variation",fontsize=20)
    plt.savefig("Annual Coefficient of Variation.png",dpi=96)# save the plot as PNG resolution of 96 dpi
    plt.close()
    
#Annual TQ Mean Plot
    fig = plt.figure(figsize=(16,10)) #custom figure size for better resolution
    plt.plot(tippe['TQmean'],'black',linestyle='None',marker='.',label='Tippecanoe')
    plt.plot(wildcat['TQmean'],'blue', linestyle='None',marker='*',label='Wildcat')
    plt.legend([riverName['Wildcat'],riverName['Tippe']], loc='best',edgecolor='k',fontsize=20)
    plt.xlabel("Year",fontsize=20)
    plt.ylabel("Tqmean",fontsize=20)
    plt.savefig("Annual TQ Mean.png",dpi=96)# save the plot as PNG with a resolution of 96 dpi
    plt.close()

# R-B Index Plot
    fig = plt.figure(figsize=(16,10)) #custom figure size for better resolution
    plt.plot(tippe['R-B Index'],'black',linestyle='None',marker='.',label='Tippecanoe')
    plt.plot(wildcat['R-B Index'],'blue', linestyle='None',marker='*',label='Wildcat')
    plt.legend([riverName['Wildcat'],riverName['Tippe']], loc='best',edgecolor='k',fontsize=20)
    plt.xlabel("Year",fontsize=20)
    plt.ylabel("R-B Index",fontsize=20)
    plt.savefig("Annual R-B Index.png",dpi=96)# save the plot as PNG with a resolution of 96 dpi
    plt.close()
    
#Average Annual Monthly Flow Plot
    fig = plt.figure(figsize=(16,10)) #custom figure size for better resolution
    plt.plot(MonthlyAverages['Tippe']['Mean Flow'],'black',linestyle='None',marker='.',label='Tippecanoe')
    plt.plot(MonthlyAverages['Wildcat']['Mean Flow'],'blue', linestyle='None',marker='*',label='Wildcat')
    plt.xticks(np.arange(1,13,1)) #custom x ticks to denote month of year
    plt.legend(loc='upper right',fontsize=20)
    plt.xlabel('Month of Year',fontsize=20)
    plt.ylabel('Discharge (cfs)',fontsize=20)
    plt.savefig('Average Monthly Flow.png')
    plt.close()

#Exceedence Probability Calculations    

    tippe2=tippe.drop(columns=['site_no', 'Mean Flow', 'Median', 'Coeff Var', 'Skew', 'TQmean', 'R-B Index', '7Q', '3xMedian'])
    tippe_flow=tippe2.sort_values('Peak Flow', ascending=False) #Sort values
    tippe_ranks1= stats.rankdata(tippe_flow['Peak Flow'], method='average') #Rank values
    tippe_ranks2=tippe_ranks1[::-1]
    tippe_ep=[(tippe_ranks2[i]/(len(tippe_flow)+1)) for i in range(len(tippe_flow))] #Excedence Calculation for Tippecanoe
    
    
    wildcat2=wildcat.drop(columns=['site_no', 'Mean Flow', 'Median', 'Coeff Var', 'Skew', 'TQmean', 'R-B Index', '7Q', '3xMedian'])
    wildcat_flow=wildcat2.sort_values('Peak Flow', ascending=False) #Sort Values
    wildcat_ranks1=stats.rankdata(wildcat_flow['Peak Flow'], method='average') #Rank Values
    wildcat_ranks2=wildcat_ranks1[::-1]
    wildcat_ep=[(wildcat_ranks2[i]/(len(wildcat_flow)+1)) for i in range(len(wildcat_flow))] #Excedence Calculation for Wildcat
    
# Excendence Probability Plot 
    fig = plt.figure(figsize=(16,10)) 
    plt.plot(tippe_ep, tippe_flow['Peak Flow'], label='Tippecanoe River', color='black')
    plt.plot(wildcat_ep, wildcat_flow['Peak Flow'], label='Wildcat River', color='blue')
    plt.xlabel("Exceedence Probability",fontsize=20)
    plt.ylabel("Peak Discharge (CFS)",fontsize=20)
    ax= plt.gca()
    ax.set_xlim(1,0) #reverse x axis 
    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig('Exceedence Probability.png', dpi=96) #Save plot as PNG with 96 dpi   
    plt.close()