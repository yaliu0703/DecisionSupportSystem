##################################################################################
#                                                                                #
# Main                                                                           #
# ::: Handles the navigation / routing and data loading / caching.               #
#                                                                                #
##################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import streamlit as st
import pickle
from io import BytesIO
import urllib


def main():
	'''Set main() function. Includes sidebar navigation and respective routing.'''

	st.sidebar.title("Explore")
	app_mode = st.sidebar.selectbox( "Choose an Action", [
		"About",
		"Start Evaluation",
		"Show Source Code"
	])

	

	# nav
	if   app_mode == "About":            show_about()
	elif app_mode == "Start Evaluation": explore_classified()
	elif app_mode == "Show Source Code": st.code(get_file_content_as_string())




@st.cache(show_spinner = False)
def read_text(fname):
	''' Display copy from a .txt file. '''
	with open(fname, 'r') as f:
		text = f.readlines()
	return text


def show_about():
	''' Home / About page '''
	st.title('About Home Equity Line of Credit (HELOC) applications decision support system')
	for line in read_text('about.txt'): #何总上传这部分的文本文件
		st.write(line)

@st.cache(show_spinner = False)
def get_file_content_as_string():
	''' Download a single file and make its content available as a string. '''
	url = 'https://raw.githubusercontent.com/zacheberhart/Learning-to-Feel/master/src/app.py'
	response = urllib.request.urlopen(url)
	return response.read().decode("utf-8")

##################################################################################
#                                                                                #
# Start Evaluation                                                               #
# ::: Allow the user to pick one or more labels to get a list of the top songs   #
# ::: classified with the respective label(s). Limit the list of songs returned  #
# ::: to 100, but allow the user to choose the quantity and the "Popularity", a  #
# ::: a metric provided by Spotify's API. Also allow the user to leave the app   #
# ::: and listen to the song on Spotify's Web App using the provided link.       #
#                                                                                #
##################################################################################

def explore_classified():

	
	# Text on Evaluation page 
	st.title('Home Equity Line of Credit (HELOC) applications decision support system')
	st.write('''
		1. User choose one row record from existing FICO credit report dataset
		2. User may customize the data if he wants to
		3. When user clicks the checkbox "Show evaluation result", the prediction result, explaination and graph will be shown
		#何总会负责这部分的文字润色
		
	''')

	# Step 1:User chooses one row record from existing FICO credit report dataset
	if st.checkbox('Check here to see existing credit report'):
		st.write(X_test)
	number = st.text_input('Choose a row in credit report for evaluation (0~2029):', 0)  # Input the index number
	
	# Step 2:Get user input
	newdata = get_input(int(number))
	
	# Step 3: when user checks, run model
	if st.checkbox('Show evaluation result'):
		run_model(newdata)
		
def run_model(newdata):
    BestBg = BaggingClassifier(base_estimator=svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', 
    degree=3, gamma='auto_deprecated', kernel='rbf', max_iter=-1, probability=False, random_state=1, shrinking=True, tol=0.001, verbose=False), 
    bootstrap=True, bootstrap_features=False, max_features=0.6, max_samples=0.3, n_estimators=50, n_jobs=None, oob_score=False, random_state=None, 
    verbose=0, warm_start=False)
    BestBg.fit(X_train_df, y_train_df)
    pred = BestBg.predict(newdata)
    if pred[0] == 1:
        st.text("Risk performance is Good")
    else:
        st.text("Risk performance is Bad")
    show_evaluation(pred,newdata)
        
def show_evaluation(pred,newdata):
	    	
	#find the mean and std
    trainstandard = {}

    for colname in list(X_train_df):
        num = []
        mean = np.mean(X_train_df[colname])
        std = np.std(list(X_train_df[colname]))
        num.append(mean)
        num.append(std)
        trainstandard[colname] = num

    #find the possibility of the variable value in standard distribution
    evaluation = {} 

    from scipy.stats import norm

    if pred[0] == 0:
        for colname in list(X_train_df):
            poss = norm.cdf(X_train_df[colname][0], loc=trainstandard[colname][0], scale = trainstandard[colname][1])
            if poss > 0.5:
                poss = 1 - poss
            evaluation[colname] = poss
        list1= sorted(evaluation.items(),key=lambda x:x[1])
        notgood = []
        for i in range(3):
            colname = list1[i][0]
            #significant = abs(float(newdata[colname][0]) - trainstandard[colname][0]) > 0.5*trainstandard[colname][1]
            notgood.append(colname)
            problem = ','.join(notgood)
        if len(notgood) > 0:        
            st.write('The customer is not so reliable because the person has bad',problem,'.')
        else:
            st.write('Some expert should look at the person data.')

        #how to use (-7,-8,-9)
    else:
        missingvalue = []
        for colname in list(X_train_df):
            arr = np.arange(len(newdata.index))
            judge = newdata[colname].isin([-7,-8,-9])
            people = list(arr[judge])
            if len(people) > 0:
                missingvalue.append(colname)
        if len(missingvalue) > 0:
            st.text('It is risky to allow the person to apply large number of loan.Because the person does not have enough historical data. So it is not sure about the risk.')
                


    
	
	
	
	
def get_input(index):
    
    values = X_test_df.iloc[index]  # Input the value from dataset

    # Create input variables for evaluation please use these variables for evaluation
    
    ExternalRiskEstimate = st.sidebar.slider('External Risk Estimate', 1.0, 100.0, float(values[0]))
    MSinceOldestTradeOpen = st.sidebar.text_input('Months Since Oldest Trade Open:', values[1])
    MSinceMostRecentTradeOpen = st.sidebar.text_input('Months Since Most Recent Trade Open:', values[2])
    AverageMInFile = st.sidebar.text_input('Average Months in File:', values[3])
    NumSatisfactoryTrades = st.sidebar.text_input('Number of Satisfactory Trades:', values[4])
    NumTrades60Ever2DerogPubRec = st.sidebar.text_input('the number of trade lines on a credit bureau report that record a payment received 60 days past its due date:', values[5])
    NumTrades90Ever2DerogPubRec = st.sidebar.text_input('the number of trade lines on a credit bureau report that record a payment received 90 days past its due date:', values[6])
    PercentTradesNeverDelq = st.sidebar.slider('Percent of Trades Never Delinquent', 0.0, 100.0, float(values[7]))
    MSinceMostRecentDelq = st.sidebar.text_input('Months Since Most Recent Delinquency', values[8])
    MaxDelq2PublicRecLast12M = st.sidebar.slider('Max Deliquncy on Public Records Last 12 Months:', 0.0, 9.0, float(values[9]))
    MaxDelqEver = st.sidebar.slider('Max Deliquncy Ever:', 1.0, 9.0, float(values[10]))
    NumTotalTrades = st.sidebar.text_input('Number of Total Trades', values[11],4)
    NumTradesOpeninLast12M = st.sidebar.text_input('Number of Trades Open in Last 12 Months', values[12],4)
    PercentInstallTrades = st.sidebar.slider('Percent of Installment Trades', 0.0, 100.0, float(values[13]))
    MSinceMostRecentInqexcl7days = st.sidebar.text_input('Months Since Most Recent Inq excl 7days:', values[14])
    NumInqLast6M = st.sidebar.text_input('Number of Inq Last 6 Months:', values[15])
    NumInqLast6Mexcl7days = st.sidebar.text_input('Number of Inq Last 6 Months excl 7days. Excluding the last 7 days removes inquiries that are likely due to price comparision shopping.:', values[16])
    NetFractionRevolvingBurden = st.sidebar.text_input('Net Fraction Revolving Burden:', values[17])
    NetFractionInstallBurden = st.sidebar.text_input('Net Fraction Installment Burden:', values[18])
    NumRevolvingTradesWBalance = st.sidebar.text_input('Number Revolving Trades with Balance:', values[19])
    NumInstallTradesWBalance = st.sidebar.text_input('Number Installment Trades with Balance:', values[20])
    NumBank2NatlTradesWHighUtilization = st.sidebar.text_input('Number Bank/Natl Trades w high utilization ratio:', values[21])
    PercentTradesWBalance = st.sidebar.slider('Percent Trades with Balance:', 0.0, 100.0, float(values[22]))
    
    newdata = pd.DataFrame()
    newdata = newdata.append({'ExternalRiskEstimate':ExternalRiskEstimate,
            'MSinceOldestTradeOpen':MSinceOldestTradeOpen,
            'MSinceMostRecentTradeOpen':MSinceMostRecentTradeOpen,
            'AverageMInFile':AverageMInFile, 
            'NumSatisfactoryTrades':NumSatisfactoryTrades,
            'NumTrades60Ever2DerogPubRec':NumTrades60Ever2DerogPubRec,
            'NumTrades90Ever2DerogPubRec':NumTrades90Ever2DerogPubRec,
            'PercentTradesNeverDelq':PercentTradesNeverDelq,
            'MSinceMostRecentDelq':MSinceMostRecentDelq,
            'MaxDelq2PublicRecLast12M':MaxDelq2PublicRecLast12M,
            'MaxDelqEver':MaxDelqEver,
            'NumTotalTrades':NumTotalTrades,
            "NumTradesOpeninLast12M": NumTradesOpeninLast12M,
            'MaxDelq2PublicRecLast12M':MaxDelq2PublicRecLast12M,
            'PercentInstallTrades':PercentInstallTrades,
            'MSinceMostRecentInqexcl7days':MSinceMostRecentInqexcl7days, 
            'NumInqLast6M':NumInqLast6M,
            'NumInqLast6Mexcl7days':NumInqLast6Mexcl7days,
            'NetFractionRevolvingBurden':NetFractionRevolvingBurden, 
            'NetFractionInstallBurden':NetFractionInstallBurden,
            'NumRevolvingTradesWBalance':NumRevolvingTradesWBalance,
            'NumInstallTradesWBalance':NumInstallTradesWBalance,
            'NumBank2NatlTradesWHighUtilization':NumBank2NatlTradesWHighUtilization,
            'PercentTradesWBalance':PercentTradesWBalance},ignore_index=True)
    
    
    return newdata
        
   
 

    

       
##################################################################################
#                                                                                #
# Execute                                                                        #
#                                                                                #
##################################################################################


if __name__ == "__main__":
	
    X_test = pickle.load(open('data/X_test.sav', 'rb'))
    y_test = pickle.load(open('data/y_test.sav', 'rb'))
    X_train = pickle.load(open('data/X_train.sav', 'rb'))
    y_train = pickle.load(open('data/y_train.sav', 'rb'))
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

	# execute
    main()

