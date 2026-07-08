/*        set the libname myhome variable to the local folder                                  */
/*        this folder has string geometry '/home/university/name                               */
/*        the queried data is from data_fetcher.sas is stored using the alias tempx            */
/*        hence, to download the data, set the output file to /home/univ/user/data_alias.csv   */

libname myHome '/home/univ/user';

proc export data=work.tempx
	outfile='/home/univ/user/data_alias.csv'
  dbms=csv
  replace;
run;
