#!Les Warren @warre112
# April 18, 2020
# Lab 11, ABE 65100

#This script is desgined to use code from lab 10 and make presentation graphics for the two imput datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    colnames = ['site_no','Mean Flow','Coeff Var','TQmean','R-B Index']
    monthdata= DataDF.resample('MS').mean() 

    MoDataDF = pd.DataFrame(0, index=monthdata.index, columns=colnames)
    
    MoDataDF['site_no']=DataDF.resample('MS')['site_no'].mean()
  
    MoDataDF['Mean Flow']=DataDF.resample('MS')['Discharge'].mean()
   

   

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

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
 