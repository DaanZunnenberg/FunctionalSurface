/****** Input area (users should modify this area) **************************/
/* Enter your WRDS institution name and your WRDS username */
%let wrds_institution = vu;
%let wrds_username = %sysget(USER);   * defaults to the OS login of the account running SAS;
%let path=/home/&wrds_institution./&wrds_username./sasoutput;
%let interval_seconds = 1*60; /* 1 min */
%let start_time = '4:00:00.000000000't; /* starting time; */
%let tickers = 'SPY';
%let start_year = 2015;
%let end_year = 2023;

%let nbbo_libpath = /wrds/nyse/sasdata/taqms/nbbo;
%let cq_libpath = /wrds/nyse/sasdata/taqms/cq;
%let ct_libpath = /wrds/nyse/sasdata/taqms/ct;

/****** End of input area **********************/

libname project "&path.";
options errors=2;

libname nbbo "&nbbo_libpath.";
libname cq "&cq_libpath.";
libname ct "&ct_libpath.";


%MACRO LOOP();

%DO J=&start_year. %TO &end_year.;
	%DO M=1 %TO 12;
		%let month = %sysfunc(putn(&M,Z2.));

	    /* STEP 1: RETRIEVE DAILY TRADE AND QUOTE (DTAQ) FILES */

	    /* Retrieve NBBO data */
	    data DailyNBBO;

	        /* Enter NBBO file names in YYYYMMDD format for the dates you want */
	        /*set nbbo.nbbom_20150202;*/
			set nbbo.nbbom_&J.&month.:;
	        /*set nbbo.nbbom_&J.:;*/

			/* Enter company tickers you want */
	        where sym_root in (&tickers.) 

			/* This selects common stocks only */
	        and sym_suffix = '';
       
	        /* Alternatively, to select dual shares, preferred shares, etc., 
			   delete the two SAS code lines above, enable the two SAS code lines 
			   below, and replace sym_root with symbol throughout the code below.

			space=' ';symbol=catx(space,sym_root,sym_suffix);format symbol $17.;
			if symbol in ('AAPL','IBM','BRK A','BRK B') */

	        /* Quotes are retrieved prior to market open time to ensure that 
			   NBBO quotes are available for beginning of the day trades 
	        and (("9:00:00.000000000"t) <= time_m <= ("16:00:00.000000000"t));*/
	        format date date9.;
	        format time_m TIME20.9;
	    run;

	    /* Retrieve Quote data */
	    data DailyQuote;

	        /* Enter Quote file names in YYYYMMDD format for the same dates */
	        /*set cq.cqm_20150202;*/
			set cq.cqm_&J.&month.:;
	        /*set cq.cqm_&J.:;*/

			/* Enter the same company tickers as above */
	        where sym_root in (&tickers.) 

			/* This selects common stocks only */
	        and sym_suffix = '';

	        /* Alternatively, to select dual shares, preferred shares, etc., 
			   delete the two SAS code lines above, enable the two SAS code lines 
			   below, and replace sym_root with symbol throughout the code below.

			space=' ';symbol=catx(space,sym_root,sym_suffix);format symbol $17.;
			if symbol in ('AAPL','IBM','BRK A','BRK B') */

	        /* Quotes are retrieved prior to market open time to ensure that 
			   NBBO quotes are available for beginning of the day trades
	        and (("9:00:00.000000000"t) <= time_m <= ("16:00:00.000000000"t));*/
	        format date date9.;
	        format time_m TIME20.9;
	    run;


		/* STEP 2: CLEAN THE DTAQ NBBO FILE */ 

	    data NBBO2;
	        set DailyNBBO;

	        /* Quote Condition must be normal (i.e., A,B,H,O,R,W) */
	        if Qu_Cond not in ('A','B','H','O','R','W') then delete;

	    	/* If canceled then delete */
	        if Qu_Cancel='B' then delete;

	    	/* if both ask and bid are set to 0 or . then delete */
	        if Best_Ask le 0 and Best_Bid le 0 then delete;
	        if Best_Asksiz le 0 and Best_Bidsiz le 0 then delete;
	        if Best_Ask = . and Best_Bid = . then delete;
	        if Best_Asksiz = . and Best_Bidsiz = . then delete;

	    	/* Create spread and midpoint */
	        Spread=Best_Ask-Best_Bid;
	        Midpoint=(Best_Ask+Best_Bid)/2;

	    	/* If size/price = 0 or . then price/size is set to . */
	        if Best_Ask le 0 then do;
	            Best_Ask=.;
	            Best_Asksiz=.;
	        end;
	        if Best_Ask=. then Best_Asksiz=.;
	        if Best_Asksiz le 0 then do;
	            Best_Ask=.;
	            Best_Asksiz=.;
	        end;
	        if Best_Asksiz=. then Best_Ask=.;
	        if Best_Bid le 0 then do;
	            Best_Bid=.;
	            Best_Bidsiz=.;
	        end;
	        if Best_Bid=. then Best_Bidsiz=.;
	        if Best_Bidsiz le 0 then do;
	            Best_Bid=.;
	            Best_Bidsiz=.;
	        end;
	        if Best_Bidsiz=. then Best_Bid=.;

	    	/*	Bid/Ask size are in round lots, replace with new shares variable*/
	    	Best_BidSizeShares = Best_BidSiz * 100;
	    	Best_AskSizeShares = Best_AskSiz * 100;
	    run;

	    /* STEP 3: GET PREVIOUS MIDPOINT */

	    proc sort 
	        data=NBBO2 (drop = Best_BidSiz Best_AskSiz);
	        by sym_root date;
	    run; 

	    data NBBO2;
	        set NBBO2;
	        by sym_root date;
	        lmid=lag(Midpoint);
	        if first.sym_root or first.date then lmid=.;
	        lm25=lmid-2.5;
	        lp25=lmid+2.5;
	    run;

	    /* If the quoted spread is greater than $5.00 and the bid (ask) price is less
	       (greater) than the previous midpoint - $2.50 (previous midpoint + $2.50), 
	       then the bid (ask) is not considered. */

	    data NBBO2;
	        set NBBO2;
	        if Spread gt 5 and Best_Bid lt lm25 then do;
	            Best_Bid=.;
	            Best_BidSizeShares=.;
	        end;
	        if Spread gt 5 and Best_Ask gt lp25 then do;
	            Best_Ask=.;
	            Best_AskSizeShares=.;
	        end;
	    	keep date time_m sym_root Best_Bidex Best_Bid Best_BidSizeShares 
	             Best_Askex Best_Ask Best_AskSizeShares Qu_SeqNum;
	    run;

	    /* STEP 4: OUTPUT NEW NBBO RECORDS - IDENTIFY CHANGES IN NBBO RECORDS 
	       (CHANGES IN PRICE AND/OR DEPTH) */

	    data NBBO2;
	        set NBBO2;
	        if sym_root ne lag(sym_root) 
	           or date ne lag(date) 
	           or Best_Ask ne lag(Best_Ask) 
	           or Best_Bid ne lag(Best_Bid) 
	           or Best_AskSizeShares ne lag(Best_AskSizeShares) 
	           or Best_BidSizeShares ne lag(Best_BidSizeShares); 
	    run;

	    /* STEP 5: CLEAN DTAQ QUOTES DATA */

	    data quoteAB;
	        set DailyQuote;

	        /* Create spread and midpoint*/;
	        Spread=Ask-Bid;

	    	/* Delete if abnormal quote conditions */
	        if Qu_Cond not in ('A','B','H','O','R','W')then delete; 

	    	/* Delete if abnormal crossed markets */
	        if Bid>Ask then delete;

	    	/* Delete abnormal spreads*/
	        if Spread>5 then delete;

	    	/* Delete withdrawn Quotes. This is 
	    	   when an exchange temporarily has no quote, as indicated by quotes 
	    	   with price or depth fields containing values less than or equal to 0 
	    	   or equal to '.'. See discussion in Holden and Jacobsen (2014), 
	    	   page 11. */
	        if Ask le 0 or Ask =. then delete;
	        if Asksiz le 0 or Asksiz =. then delete;
	        if Bid le 0 or Bid =. then delete;
	        if Bidsiz le 0 or Bidsiz =. then delete;
	    	drop Sym_Suffix Bidex Askex Qu_Cancel RPI SSR LULD_BBO_CQS 
	             LULD_BBO_UTP FINRA_ADF_MPID SIP_Message_ID Spread NATL_BBO_LULD;
	    run;


	    /* STEP 7: THE NBBO FILE IS INCOMPLETE BY ITSELF (IF A SINGLE EXCHANGE 
	       HAS THE BEST BID AND OFFER, THE QUOTE IS INCLUDED IN THE QUOTES FILE, BUT 
	       NOT THE NBBO FILE). TO CREATE THE COMPLETE OFFICIAL NBBO, WE NEED TO 
	       MERGE WITH THE QUOTES FILE (SEE FOOTNOTE 6 AND 24 IN OUR PAPER) */

	    data quoteAB2 (rename=(Ask=Best_Ask Bid=Best_Bid));
	        set quoteAB;
	        where (Qu_Source = "C" and NatBBO_Ind='1') 
	           or (Qu_Source = "N" and NatBBO_Ind='4');
	        keep date time_m sym_root Qu_SeqNum Bid Best_BidSizeShares Ask 
	             Best_AskSizeShares;

	    	/*	Bid/Ask size are in round lots, replace with new shares variable
	    	and rename Best_BidSizeShares and Best_AskSizeShares*/
	    	Best_BidSizeShares = Bidsiz * 100;
	    	Best_AskSizeShares = Asksiz * 100;
	    run;

	    proc sort data=NBBO2;
	        by sym_root date Qu_SeqNum;
	    run;

	    proc sort data=quoteAB2;
	        by sym_root date Qu_SeqNum;
	    run;

	    data OfficialCompleteNBBO (drop=Best_Askex Best_Bidex);
	        set NBBO2 quoteAB2;
	        by sym_root date Qu_SeqNum;
	    run;

	    /* If the NBBO Contains two quotes in the exact same microseond, assume 
	       last quotes (based on sequence number) is active one */
	    proc sort data=OfficialCompleteNBBO;
	        by sym_root date time_m descending Qu_SeqNum;
	    run;

	    proc sort data=OfficialCompleteNBBO nodupkey;
	        by sym_root date time_m;
	    run;


	    data OfficialCompleteNBBO;
	        set OfficialCompleteNBBO;type='Q';
	        time_m=time_m+.000000001;
	    	drop Qu_SeqNum;
	    run;


	    /* STEP 13: DOWNLOAD THE FILES YOU WANT TO YOUR PROJECT FOLDER

	       By default, only the last three output files are downloaded to the 
	       "project" folder. You can download all nine of the files by 
	       removing the "comment outs" in the statements below */

	    /*data project.DailyNBBO;set DailyNBBO;run;
	    proc export data=DailyNBBO
		    outfile="&path./DailyNBBO.csv"
		    dbms=csv replace;
		    putnames=YES;
		run;*/

	    /*data project.DailyQuote;set DailyQuote;run;
	    proc export data=DailyQuote
		    outfile="&path./DailyQuote.csv"
		    dbms=csv replace;
		    putnames=YES;
		run;*/

	    /*data project.OfficialCompleteNBBO;set OfficialCompleteNBBO;run;*/
	    proc export data=OfficialCompleteNBBO
		    outfile="&path./OfficialCompleteNBBO.csv"
		    dbms=csv replace;
		    putnames=YES;
		run;

	    /* STEP: downscale to 1 min:
	        Screen data to find the trade before a set time interval */     

	     data xtemp;
	        set OfficialCompleteNBBO;
	        by sym_root date time_m;
	        format itime rtime TIME20.9;
	        if first.sym_root=1 or first.date=1 then do;
	         *Initialize time and price when new symbol or date starts;
	         rtime=time_m;
	         iprice=Best_Bid;
	         oprice=Best_Ask;
	         itime= &start_time;
	        end;
	        if time_m >= itime then do; *Interval reached;
	         output; *rtime and iprice hold the last observation values;
	         itime = itime + &interval_seconds;
	         do while(time_m >= itime); *need to fill in all time intervals;
	             output;
	             itime = itime + &interval_seconds;
	         end;
	        end;
	        rtime=time_m;
	        iprice=Best_Bid;
	        oprice=Best_Ask;
	        retain itime rtime iprice oprice; *Carry time and price values forward;
	        keep sym_root date itime rtime iprice oprice;
	    run;

	    proc append base=CompleteNBBO_1min data=xtemp force;
	    run;

	%END;
%END;
%MEND LOOP;

%LOOP();
run;


proc export data=CompleteNBBO_1min
    outfile="&path./CompleteNBBO_1min_daily.csv"
    dbms=csv replace;
    putnames=YES;
run;


