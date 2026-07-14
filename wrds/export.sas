/*        set the output_path macro variable to your local WRDS project folder                */
/*        this folder has string geometry '/home/university/name'                              */
/*        the queried data from data_fetcher.sas is stored using the alias tempx               */
/*        hence, to download the data, it is exported to &output_path./data_alias.csv          */

/****** Input area (users should modify this area) **************************/
%let output_path = /home/univ/user; * your local WRDS project folder;
%let output_file = data_alias;      * output CSV file name (without extension);

/****** End of input area **********************/

libname myHome "&output_path.";

proc export data=work.tempx
	outfile="&output_path./&output_file..csv"
  dbms=csv
  replace;
run;
