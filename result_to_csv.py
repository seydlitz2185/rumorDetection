# %%
import os
import regex as re
import pandas as pd

# %%
reports = [file   for file in os.listdir('report') if re.match(r'.*\.txt',file)]

report_text = []
matricx_text = []

cloumns = ['model','report','matrix',
           'precision_0','recall_0','f1_0','support_0',
           'precision_1','recall_1','f1_1','support_1',
           'accuracy','macro avg_precision','macro avg_recall','macro avg_f1','weighted avg_precision','weighted avg_recall','weighted avg_f1',]

df = pd.DataFrame(columns=cloumns)
reports  = sorted(reports)
for i in range(len(reports)):
    with open('report_matrix/'+reports[i].replace('.txt','_matrix.txt'),'r') as f:
        matrix = f.read()
        nums = re.findall(r'[0-9]+',matrix)
        nums = [[nums[0],nums[1]], [nums[2],nums[3]]]
    with open('report/'+reports[i],'r') as f:
        report = reports[i].replace('.txt','')
        text = f.read()
        scores = re.findall(r'[0-9]+.[0-9]+',text)
        df.loc[i,'model'] = report.split('_')[0]
        df.loc[i,'report'] = text
        df.loc[i,'matrix'] = nums
        df.loc[i,'precision_0'] = scores[0]
        df.loc[i,'recall_0'] = scores[1]
        df.loc[i,'f1_0'] = scores[2]
        df.loc[i,'support_0'] = scores[3]
        df.loc[i,'precision_1'] = scores[4]
        df.loc[i,'recall_1'] = scores[5]
        df.loc[i,'f1_1'] = scores[6]
        df.loc[i,'support_1'] = scores[7]
        df.loc[i,'accuracy'] = scores[8]
        df.loc[i,'macro avg_precision'] = scores[10]
        df.loc[i,'macro avg_recall'] = scores[11]
        df.loc[i,'macro avg_f1'] = scores[12]
        df.loc[i,'weighted avg_precision'] = scores[14]
        df.loc[i,'weighted avg_recall'] = scores[15]
        df.loc[i,'weighted avg_f1'] = scores[16]



# %%
if os.path.exists('report_csv') == False:
    os.mkdir('report_csv')
    df.to_csv('report_csv/report_1.csv',index=False)
else:
    report_csv = [file   for file in os.listdir('report_csv') if re.match(r'.*\.csv',file)]
    df.to_csv('report_csv/report_'+str(len(report_csv)+1)+'.csv',index=False)


