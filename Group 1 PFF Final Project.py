# -*- coding: utf-8 -*-
"""G1_Python For Finance.ipynb

@Prepared by Group 1:
1. Amin Ilyas
2. Hélène Do
3. Logba Elvire Ursule Vera Yapi
4. Nico Benedikt Horstmann
5. Pritam Ritu Raj

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy import optimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
##########Part A#####################################################
#Import files and basic transformation

fundquart = pd.read_excel('WRDS FundQuart_NorthAmerica_Q1-20_Q4-22_Compustat.xlsx', sheet_name= "FundQrt")
Covid_Data = pd.read_csv('IV1 Covid Data - Stringency Index Q1-20_Q4-22.csv')
Federal_fund = pd.read_csv('IV2 Macro Data - Federal Funds ER Q1-20_Q4-22.csv')
#Rename WRDS Quaterly columns name
fundquart.rename(columns={'atq':'Total Assets','ltq':'Total Liabilities','gvkey':'Global Company Key','datafqtr':'Fiscal Quarter by Year','fqtr':'Fiscal Quarter','datadate':'Data Date','fyearq':'Fiscal Year','indfmt':'Industry Format','consol':'Fi statement Consolidation code','popsrc':'Population Source','datafmt':'Data Format','tic':'Ticker','conm':'Company Name','curcdq':'ISO Currency Code','datacqtr':'Calendat Data Year & Quater','datafqty':'Fiscal Quarter by Year','cogsq':'COGS','niq':'Net income (loss)','req':'Retained Earnings','revtq':'Total Revenue','txtq':'Total Income Taxes','wcapq':'Working Capital','xintq':'Total Interested & Related Expense','xoprq':'Total Operating Expense','costat':'Active/Inactive Status marker','mkvaltq':'Total Market Value','busdesc':'Business Description','gsector':'GICS Sectors','loc':'ISO Country code'},inplace=True)
#Keep only USA
fundquart_USA = fundquart[fundquart["ISO Country code"] == "USA"]

#Filter by GICS Sectors : industrys (20) & Health Care (35)

#Select the industry
def MyCovidIndustryStats(Covid_Data,FEDFUNDS,fundquart_USA,GICS_Code):
    FundQ_industry = fundquart_USA.loc[fundquart_USA['GICS Sectors'] == GICS_Code]


    FundQ_industry_Q = FundQ_industry.groupby("Fiscal Quarter by Year").sum()

    #Drop 2019 Q3 and 2019Q4 industry
    FundQ_industry_Q.drop(FundQ_industry_Q.head(2).index,inplace=True)
    #Drop 2022 Q4 and 2023Q1 industry
    FundQ_industry_Q.drop(FundQ_industry_Q.tail(2).index,inplace=True)


    #Drop 2019 Q3 and 2019Q4 Healthcare
    #FundQ_healthcare_Q.drop(FundQ_healthcare_Q.head(2).index,inplace=True)
    #Drop 2022 Q4 and 2023Q1 Healthcare#
    #FundQ_healthcare_Q.drop(FundQ_healthcare_Q.tail(2).index,inplace=True)

    #FEDFUNDS basic transforrmation
    #Federal Funds transform
    Federal_fund.drop(Federal_fund.tail(1).index,inplace=True)


    ########################################################################

    ##############################Covid_Data basic transforrmation
    Covid_Data = Covid_Data[(Covid_Data["location"] == "United States")]

    Covid_Data= Covid_Data[["date",'new_cases', 'stringency_index']]

    Covid_Data["date"] = pd.to_datetime(Covid_Data["date"])

    Covid_Data['year'] = pd.DatetimeIndex(Covid_Data['date']).year
    Covid_Data['month'] = pd.DatetimeIndex(Covid_Data['date']).month

    Covid_cases = Covid_Data[["date","year", "month", "new_cases"]]
    Covid_stringency = Covid_Data[["date","year", "month", "stringency_index"]]

    Covid_cases= Covid_cases.groupby(["year", "month"]).sum()

    Covid_stringency= Covid_stringency.groupby(["year", "month"]).mean()

    Covid_cases.reset_index(inplace = True)
    Covid_stringency.reset_index(inplace = True)

    Covid_cases.drop(Covid_cases.tail(2).index,inplace=True)
    Covid_stringency.drop(Covid_stringency.tail(2).index,inplace=True)

    Covid_cases= Covid_cases.groupby(Covid_cases.index // 3).sum()
    Covid_stringency= Covid_stringency.groupby(Covid_stringency.index // 3).mean()

    #Reset the index to be able to use the Fiscal Quarter by Year column
    #FundQ_healthcare_Q.reset_index(inplace= True)
    FundQ_industry_Q.reset_index(inplace= True)

    #Add the Quarter columsn
    Covid_cases["Fiscal Quarter by Year"] = FundQ_industry_Q["Fiscal Quarter by Year"]
    Covid_stringency["Fiscal Quarter by Year"] = FundQ_industry_Q["Fiscal Quarter by Year"]
    Covid_Data = pd.merge(Covid_cases, Covid_stringency, on = "Fiscal Quarter by Year", how = "outer")
    Covid_Data = Covid_Data[["Fiscal Quarter by Year",'new_cases', 'stringency_index']]



    ########################################################################

    Federal_fund["Fiscal Quarter by Year"] = FundQ_industry_Q["Fiscal Quarter by Year"]

    Federal_fund["FEDFUNDS"] = pd.to_numeric(Federal_fund["FEDFUNDS"])

    Federal_fund["FEDFUNDS"] = Federal_fund["FEDFUNDS"]/100

    ##########################################################################################
    #New Version
    FundQ_sector = FundQ_industry.groupby(["cusip","Fiscal Quarter by Year"]).sum()

    FundQ_sector.reset_index(inplace = True)


    FundQ_sector = FundQ_sector.merge(Covid_Data, on="Fiscal Quarter by Year" , how='outer')
    FundQ_sector = FundQ_sector.merge(Federal_fund, on="Fiscal Quarter by Year" , how="outer")

    #maybe use inner merge also
    #####################################################################################



    #EBIT calculation
    FundQ_sector["EBIT"] = FundQ_sector["Total Revenue"] - FundQ_sector["COGS"] - FundQ_sector["Total Operating Expense"] 
    #Net Income + Income + Taxes = EBIT
    #Z-score
    FundQ_sector["Z_Score"] = 1.2*(FundQ_sector["Working Capital"]/FundQ_sector['Total Assets']) \
    +1.4*(FundQ_sector['Retained Earnings']/FundQ_sector['Total Assets'])\
    +3.3*(FundQ_sector['EBIT']/FundQ_sector['Total Assets']) \
    +0.6*(FundQ_sector['Total Market Value']/FundQ_sector['Total Liabilities'])\
    +1.0*(FundQ_sector['Total Revenue']/FundQ_sector['Total Assets'])

    #Drop missing Z-score values

    FundQ_sector = FundQ_sector[FundQ_sector["Z_Score"].notna()]
    #####################################################################
    ##MERGE!

    Total_df = FundQ_sector[["cusip", "Fiscal Quarter by Year", "new_cases", "stringency_index", "FEDFUNDS", "Total Market Value" , "Z_Score"]]

    Total_df.reset_index(inplace = True)

    Total_df = FundQ_sector[["cusip", "Fiscal Quarter by Year", "new_cases", "stringency_index", "FEDFUNDS", "Total Market Value" , "Z_Score"]]
    # take logarithm of the market capitalization
    # measure the firm size
    Total_df['size']=np.log(Total_df['Total Market Value'])

    Total_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN

    Total_df = Total_df.dropna()

    ############### Preliminary Visualization
    # number of sample firms by year
    obs_year=Total_df.groupby(['Fiscal Quarter by Year'])[['cusip']].count()
    obs_year.reset_index(inplace=True)
    obs_year.rename(columns={"cusip": "obs"}, inplace=True)

    #Histogram of Number of firms
    obs_year.plot(kind = "bar", x = "Fiscal Quarter by Year", y= "obs", ylabel = "Number of Sample firms", legend = None)
    plt.xticks(rotation = 360)

    #Pie Chart distribution
    #############################################################################

    # plot the distribution of Z-score
    # Z > 2.99 – "safe" zone
    # 1.81 <= Z <= 2.99 – "grey" zone
    # Z < 1.81 – "distress" zone

    # generate an indicator in the dataframe for this
    def finpos_classify(row):
        if row['Z_Score']<1.81:
            finpos='distress'
        elif row['Z_Score']<=2.99:
            finpos='grey'
        else:
            finpos='safe'
        return finpos

    Total_df['finpos']=Total_df.apply(finpos_classify,axis=1)

    # aggregate
    z_group_obs=Total_df.groupby(['finpos'])[['cusip']].count()
    z_group_obs.reset_index(inplace=True)
    z_group_obs.rename(columns={"cusip": "obs"}, inplace=True)

    # plot the distribution using a pie chart

    #colors
    colors = ['#ff9999','#66b3ff','#99ff99']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.pie(z_group_obs['obs'],colors=colors,labels=z_group_obs['finpos'],autopct='%1.1f%%')
    plt.show()

    #############################################################################################


    #Descriptive statistics
    print(Total_df[['new_cases','stringency_index','FEDFUNDS', "Z_Score"]].describe())

    #Correlation matrix
    Total_df[['new_cases','stringency_index','FEDFUNDS', "Z_Score"]].corr()

    ###########################################################################################

    # histogram of Z-score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(Total_df['Z_Score'],density=True,bins=50)
    ax.set_ylabel('Density')
    ax.set_xlabel("Zscore")
    plt.show()

    # histogram of new cases
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(Total_df['new_cases'],density=True,bins=30)
    ax.set_ylabel('Density')
    ax.set_xlabel("New Covid Cases")
    plt.show()

    # histogram of stringency_index
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(Total_df['stringency_index'],density=True,bins=30)
    ax.set_ylabel('Density')
    ax.set_xlabel("stringency_index")
    plt.show()

    # histogram of FEDFUNDS
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(Total_df['FEDFUNDS'],density=True,bins=30)
    ax.set_ylabel('Density')
    ax.set_xlabel("FEDFUNDS")
    plt.show()


    ################################################################
    #Log of all variables but the Z-score
    #Take the logarithm for the independent variables
    Total_df["logNC"] = np.log(1+Total_df["new_cases"])
    Total_df["logSI"] = np.log(1+Total_df["stringency_index"])
    Total_df["logFF"] = np.log(1+Total_df["FEDFUNDS"])

    ###################################################################

    # descriptive statistics of log independent variables
    print(Total_df[['logNC','logSI','logFF']].describe())

    ########################################################

    # trim the z-score at 2.5% and 97.5% percentile
    zscore_p2dot5=np.percentile(Total_df['Z_Score'],2.5)
    zscore_p97dot5=np.percentile(Total_df['Z_Score'],97.5)
    Total_df=Total_df[(Total_df['Z_Score']>=zscore_p2dot5) & (Total_df['Z_Score']<=zscore_p97dot5)]

    # histogram of trimmed Z-score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(Total_df['Z_Score'],density=True,bins=50)
    ax.set_ylabel('Density')
    ax.set_xlabel("Zscore")
    plt.show()



    # correlation analysis for the main variables
    # two variable pearson correlation

    corr, pval = stats.pearsonr(Total_df['logSI'],
                                Total_df['logFF'])
    print("The pearson correlation between the Stringency index and FEDFUNDS is " + str(round(corr,2)))
    #Descriptive statisticts for the winsorized Z-Score
    print(Total_df[["Z_Score"]].describe())
    #Pearson Correlation matrix
    print(Total_df[['logNC','logSI','logFF','Z_Score', "size"]].corr(method='pearson'))

    #Heatmap
    var_corrplot=sns.heatmap(Total_df[['logNC','logSI','logFF','Z_Score', "size"]].corr(), vmin=-0.2, vmax=1, cmap="PiYG")
    plt.show()
    fig=var_corrplot.get_figure()

    # two variable spearman correlation
    rho, pval = stats.spearmanr(Total_df['logSI'],
                                Total_df['logFF'])
    print("The spearman correlation between the Stringency index and FEDFUNDS is " + str(round(rho,2)))

    # spearman correlation matrix
    print(Total_df[['logNC','logSI','logFF','Z_Score', "size"]].corr(method='spearman'))

    ###################################################################################################
    ######################################Part B starts here###########################################
    ##################################################
    #                Normality Test
    ##################################################
    # Jarque-Bera test
    jbstat,p=stats.jarque_bera(Total_df['logNC'])
    print("")
    print("The Jarque-Bera test stat of New Cases is " + str(round(jbstat,2)))
    print("The p value of Jarque-Bera test is " + str(round(p,3)))

    #The p-value of Jarque–Bera test for new cases is <
    #0.01. Thus we can reject the null hypothesis of normal distribution
    #at 1% significance level. Thus, the new cases variable is not normally distributed

    jbstat,p=stats.jarque_bera(Total_df['logSI'])
    print("")
    print("The Jarque-Bera test stat of Stringency-index is " + str(round(jbstat,2)))
    print("The p value of Jarque-Bera test is " + str(round(p,3)))

    #The p-value of Jarque–Bera test for stringency_index is <
    #0.01. Thus we can reject the null hypothesis of normal distribution
    #at 1% significance level. Thus, the stringency variable is not normally distributed

    jbstat,p=stats.jarque_bera(Total_df['logFF'])
    print("")
    print("The Jarque-Bera test stat of FEDFUNDS is " + str(round(jbstat,2)))
    print("The p value of Jarque-Bera test is " + str(round(p,3)))

    #The p-value of Jarque–Bera test for FEDFUNDS is <
    #0.01. Thus we can reject the null hypothesis of normal distribution
    #at 1% significance level. Thus, the FEDFUNDS variable is not normally distributed

    jbstat,p=stats.jarque_bera(Total_df['Z_Score'])
    print("")
    print("The Jarque-Bera test stat of Z-score is " + str(round(jbstat,2)))
    print("The p value of Jarque-Bera test is " + str(round(p,3)))

    #The p-value of Jarque–Bera test for Z_Score is <
    #0.01. Thus we can reject the null hypothesis of normal distribution
    #at 1% significance level. Thus, the Z_Score variable is not normally distributed

    jbstat,p=stats.jarque_bera(Total_df['size'])
    print("")
    print("The Jarque-Bera test stat of size is " + str(round(jbstat,2)))
    print("The p value of Jarque-Bera test is " + str(round(p,3)))

    #The p-value of Jarque–Bera test for size is <
    #0.01. Thus we can reject the null hypothesis of normal distribution
    #at 1% significance level. Thus, the size variable is not normally distributed

    # Kolmogorov–Smirnov test
    # use a string as the second argument 
    # for the distribution to be tested against
    kstat,pval=stats.kstest(Total_df['logNC'],'norm')
    print("")
    print("The KS test stat of New Cases is " + str(round(kstat,2)))
    print("The p value of KS test is " + str(round(pval,3)))

    #The p-value of K-S test for new cases is < 0.01. Thus we can
    #reject the null hypothesis of normal distribution at 1% significance level.

    kstat,pval=stats.kstest(Total_df['logSI'],'norm')
    print("")
    print("The KS test stat of Stringency is " + str(round(kstat,2)))
    print("The p value of KS test is " + str(round(pval,3)))
    #The p-value of K-S test for Stringency is < 0.01. Thus we can
    #reject the null hypothesis of normal distribution at 1% significance level.

    kstat,pval=stats.kstest(Total_df['logFF'],'norm')
    print("")
    print("The KS test stat of FEDFUNDS is " + str(round(kstat,2)))
    print("The p value of KS test is " + str(round(pval,3)))
    #The p-value of K-S test for FEDFUNDS is < 0.01. Thus we can
    #reject the null hypothesis of normal distribution at 1% significance level.

    kstat,pval=stats.kstest(Total_df['Z_Score'],'norm')
    print("")
    print("The KS test stat of Z_Score is " + str(round(kstat,2)))
    print("The p value of KS test is " + str(round(pval,3)))
    #The p-value of K-S test for Z_Score is < 0.01. Thus we can
    #reject the null hypothesis of normal distribution at 1% significance level.


    kstat,pval=stats.kstest(Total_df['size'],'norm')
    print("")
    print("The KS test stat of size is " + str(round(kstat,2)))
    print("The p value of KS test is " + str(round(pval,3)))
    #The p-value of K-S test for size is < 0.01. Thus we can
    #reject the null hypothesis of normal distribution at 1% significance level.


    ##################################################
    #                     t-Test
    ##################################################
    # one-sample t-test
    tstat,pval=stats.ttest_1samp(Total_df['Z_Score'], popmean=1.81)
    print("")
    print("The t-test stat of Z_Score against 1.81 is " + str(round(tstat,2)))
    print("The p value of t-test is " + str(round(pval,3)))

    #The p-value of t-test for Z-score is > 0.01. Thus we can not reject the
    #null hypothesis at 1% significance level and conclude that the
    #sample average z-score is not significantly different from (higher than)
    #1.81.

    #Consistent with Pie chart (Most Firms are in the distress region)

    # two-sample t-test new cases
    tstat,pval=stats.ttest_ind(Total_df[Total_df['finpos']=="distress"]['logNC'], \
    Total_df[Total_df['finpos']!="distress"]['logNC'],equal_var=False)

    print("")
    print("The t-test stat of the diff between two groups is " + str(round(tstat,2)))
    print("The p value of t-test is " + str(round(pval,3)))

    #the p-value is the t-test is 0.0 (< 0.05), which means that the
    #difference in New cases is significantly different
    #between distressed and non-distressed firms.

    #Consistent with Pie chart

    # two-sample t-test stringency index
    tstat,pval=stats.ttest_ind(Total_df[Total_df['finpos']=="distress"]['logSI'], \
    Total_df[Total_df['finpos']!="distress"]['logSI'],equal_var=False)

    print("")
    print("The t-test stat of the diff between two groups is " + str(round(tstat,2)))
    print("The p value of t-test is " + str(round(pval,3)))

    #the p-value is the t-test is 0.0 (< 0.05), which means that the
    #difference in stringency_index is significantly different
    #between distressed and non-distressed firms.

    # two-sample t-test FEDFUNDS
    tstat,pval=stats.ttest_ind(Total_df[Total_df['finpos']=="distress"]['logFF'], \
    Total_df[Total_df['finpos']!="distress"]['logFF'],equal_var=False)

    print("")
    print("The t-test stat of the diff between two groups is " + str(round(tstat,2)))
    print("The p value of t-test is " + str(round(pval,3)))

    #the p-value is the t-test is 0.0 (< 0.05), which means that the
    #difference in FEDFUNDS is significantly different
    #between distressed and non-distressed firms.


    # two-sample t-test size
    tstat,pval=stats.ttest_ind(Total_df[Total_df['finpos']=="distress"]['size'], \
    Total_df[Total_df['finpos']!="distress"]['size'],equal_var=False)

    print("")
    print("The t-test stat of the diff between two groups is " + str(round(tstat,2)))
    print("The p value of t-test is " + str(round(pval,3)))

    #the p-value is the t-test is 0.0 (< 0.05), which means that the
    #difference in size is significantly different
    #between distressed and non-distressed firms.

    ##################################################
    #            Simple Linear Regression
    ##################################################
    # simple linear regression using statsmodels


    NC_zscore_reg = smf.ols(formula='Z_Score ~ 1 + logNC', data = Total_df).fit()
    print(NC_zscore_reg.summary())

    SI_zscore_reg = smf.ols(formula='Z_Score ~ 1 + logSI', data = Total_df).fit()
    print(SI_zscore_reg.summary())

    FF_zscore_reg = smf.ols(formula='Z_Score ~ 1 + logFF', data = Total_df).fit()
    print(FF_zscore_reg.summary())

    Size_zscore_reg = smf.ols(formula='Z_Score ~ 1 + size', data = Total_df).fit()
    print(Size_zscore_reg.summary())

    # visualizing simple linear regression
    # binscatter plot with fitted line



    #model = sm.OLS(Y, X, missing='drop')
    #model_result = model.fit()
    sm.graphics.plot_fit(NC_zscore_reg,1, vlines=False)
    sm.graphics.plot_fit(SI_zscore_reg,1, vlines=False)
    sm.graphics.plot_fit(FF_zscore_reg,1, vlines=False)

    ###############################################################################################
    #Here Part C begins
    ##################################################
    #            Multiple Linear Regression
    ##################################################


    Covid_zscore_reg = smf.ols(formula='Z_Score ~ 1 + logNC + logSI + logFF + size', data = Total_df).fit()
    print(Covid_zscore_reg.summary())

    sm.graphics.plot_fit(Covid_zscore_reg,1, vlines=False)

    ##################################################
    #               Quantile Regression
    ##################################################
    # the 60% unconditional quantiles of Z_score
    Z_Score_p60=Total_df['Z_Score'].quantile(0.6)
    print("The 60% (and below) percentile of Z_SCore is 1.61.")
     
    # quantile regression for 80% quantile
    e_qreg = smf.quantreg(formula='Z_Score ~ 1 + logNC + logSI + logFF + size', data = Total_df).fit(q=0.8)
    print(e_qreg.summary())
     
    # plot the quantile regression coeffcient and confidence intervals
    q_start=0.7
    q_end=0.98
    q_inc=0.02
    n=(q_end-q_start)/q_inc+1
    n=int(n)
    q_forplot=np.linspace(q_start,q_end,n)
    beta_forplot=np.zeros(n)
    se_forplot=np.zeros(n)
    for i in range(n):
        e_qreg = smf.quantreg(formula='Z_Score ~ 1 + logNC + logSI + logFF + size', data = Total_df).fit(q=q_forplot[i])
        beta=e_qreg.params
        beta_forplot[i]=beta[1]
        se=e_qreg.bse
        se_forplot[i]=se[1]
     
    e_reg_control = smf.ols(formula='Z_Score ~ 1 + logNC + logSI + logFF + size', data = Total_df).fit()
    beta=e_reg_control.params
    beta_ols=beta[1]
     
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(q_forplot,beta_forplot,label='Quantile Reg')
    ax.fill_between(q_forplot,beta_forplot-1.96*se_forplot,
                    beta_forplot+1.96*se_forplot,color='b',
                    alpha=0.1)
    ax.plot(q_forplot,beta_ols*np.ones(n),color='r',
            linestyle='dashed',linewidth=0.8,
            label='OLS Reg')
    ax.legend(loc='lower left')
    ax.set_xlabel('Quantile')
    ax.set_ylabel(r'$\beta_{logNC,logSI, logFF, size}$')
    plt.show()






   
#Filter by GICS Sectors : industrys (20) & Health Care (35)    
MyCovidIndustryStats(Covid_Data, Federal_fund, fundquart_USA, 20) #industrial


MyCovidIndustryStats(Covid_Data, Federal_fund, fundquart_USA, 35) #Healthcare
    
    

    
