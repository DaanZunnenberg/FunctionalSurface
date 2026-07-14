/* ************************************************************************* */
/*                                                                           */
/*   1)  Output format (e.g., every 15 minutes)                              */
/*                                                                           */
/*        Obs  sym_root     date    itime_m      iprice      rtime_m         */
/*        1     DELL     20191120   09:30:00     55.820      09:30:00.492    */
/*        2     DELL     20191120   09:45:00     55.970      09:44:57.024    */
/*        3     DELL     20191120   10:00:00     55.900      09:59:55.101    */
/*        4     DELL     20191120   10:15:00     55.630      10:14:50.566    */
/*        5     DELL     20191120   10:30:00     55.730      10:29:55.183    */
/*        6     DELL     20191120   10:45:00     55.763      10:44:39.950    */
/*        7     DELL     20191120   11:00:00     55.720      10:59:56.296    */
/*        8     DELL     20191120   11:15:00     55.840      11:14:26.593    */
/*        9     DELL     20191120   11:30:00     55.920      11:29:59.995    */
/*       10     DELL     20191120   11:45:00     55.720      11:44:59.412    */
/*                                                                           */
/*  Please note Monthly TAQ truncates millisecond timestamp and floors it to */
/*  second timestamp. To be consistent with this, you may see millisecond    */
/*  trades at the second of HH:MM:SS:xxxx. For more information about this   */
/*  fact, please check FAQ #1/(2) in the WRDS millisecond IID manual.        */ 
/*                                                                           */
/*   2)  Definitions of several variables                                    */
/*                                                                           */
/*          sym_root -- stock symbol root                                    */
/*          sym_suffix -- stock symbol suffix                                */
/*          date   -- transaction date                                       */
/*          itime_m  -- interval time                                        */
/*          iprice -- interval price corresponding to interval time (itime)  */
/*          rtime_m  -- real time (Trading time)                             */
/*                                                                           */
/*   3) Logic: choose records close to the interval time (before it).        */
/*                                                                           */
/*          For example, for 9:45:00 (11/21/2019, DELL), you will have       */
/*          iprice=55.970,which is the transaction price occurred at 9:59.55,*/
/*          see above record 2.                                              */
/*                                                                           */
/*   4) Inputs -- see input area (users should modify this area)             */
/*                                                                           */
/*          a) %let taq_ds=taqms.ctm_20191120; * data set you are interested */
/*          b) %let start_time = '9:30:00't;  * starting time                */
/*          c) %let interval_seconds = 15*60; * interval is 15 minutes       */
/*                                                                           */
/*   5) Some related issues                                                  */
/*                                                                           */
/*          a) only choose 3 stocks: SPY,IBM and DELL                        */
/*          b) only one day. If you want multiple-day data, modify the       */
/*               program accordingly                                         */
/*          c) for less frequently traded stocks, when no trades within an   */
/*         interval, missing values will occur                           	 */
/*          d) no filter is used                                             */
/*                                                                           */
/*   6) If several records share the same time stamp, the program will pick  */
/*  up the first one.                                                    	 */
/*                                                                           */
/*   7) Step added by repeating the last available price if there are no     */
/*  trades during an interval. See  'do while(time_m >= itime_m)'  code.     */
/*                                                                           */
/* ************************************************************************* */
 

options nosource nodate nocenter nonumber ps=max ls=72;

/****** Input area (users should modify this area) **************************/
%let taqms_libpath = /wrds/nyse/sasdata/taqms/ct; * WRDS millisecond TAQ trade library;
libname taqms "&taqms_libpath.";
%let taq_ds=taqms.ctm_2019:;        * the taqms.ctm_2019: dataset contains all data from 2019
%let tickers = 'AAPL';              * comma-separated list of sym_root values to keep;
%let session_start_time = '9:30:00't;  * regular trading session start;
%let session_end_time = '16:30:00't;   * regular trading session end;
%let start_time_m = '9:30:00't;    * starting time_m;
%let interval_seconds =15*60;    * interval is 15*60 seconds (15 minutes);

/****** End of input area **********************/


/* Extract data for the configured tickers, we consider the time_m
  between the configured session start/end,  only retrieve SYM_ROOT SYM_SUFFIX DATE TIME_M and PRICE; */
data tempx;
     set &taq_ds(keep=sym_root sym_suffix date time_m price);
     where sym_root in (&tickers.) and sym_suffix=' '
     and time_m between &session_start_time. and &session_end_time.;
     by sym_root sym_suffix date time_m;
     retain itime_m rtime_m iprice; *Carry time and price values forward;
        format itime_m time12. rtime_m time12.9; * if you only need second timestamp, use 'time12.' instead.;
     if first.sym_root=1 or first.date=1 then do;
        */Initialize time_m and price when new symbol or date starts;
        rtime_m=time_m;
        iprice=price;
        itime_m= &start_time_m;
     end;
     if time_m >= itime_m then do; /*Interval reached;*/
           output; /*rtime_m and iprice hold the last observation values;*/
           itime_m = itime_m + &interval_seconds;
           do while(time_m >= itime_m); /*need to fill in all time_m intervals;*/
               output;
               itime_m = itime_m + &interval_seconds;
           end;
    end;
    rtime_m=time_m;
    iprice=price;
    keep sym_root sym_suffix date itime_m iprice rtime_m;
run;

 
Title "Final output -- &interval_seconds seconds";
proc print data=tempx (obs=400);
     var sym_root sym_suffix date itime_m iprice rtime_m;
run;

 
/* ********************************************************************************* */
/* *************  Material Copyright Wharton Research Data Services  *************** */
/* ****************************** All Rights Reserved ****************************** */
/* ********************************************************************************* */