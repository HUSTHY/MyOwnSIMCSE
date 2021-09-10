import pandas as pd
import numpy as np


def get_acc_sbert(df,threshold):
    correct_df = df[ ((df['label']==1) & (df['Sbert_pre_similarity']>= threshold))|((df['label']==0) & (df['Sbert_pre_similarity']< threshold)) ]
    acc = len(correct_df)/len(df)
    return acc


# def get_acc_regr(df,threshold):
#     correct_df = df[((df['label'] == 1) & (df['pre_regres_similarity'] >= threshold)) | ((df['label'] == 0) & (df['pre_regres_similarity'] < threshold))]
#     acc = len(correct_df)/len(df)
#     return acc

def get_acc_simcseunsup(df,threshold):
    correct_df = df[ ((df['label']==1) & (df['simcseunsup_similarity']>= threshold))|((df['label']==0) & (df['simcseunsup_similarity']< threshold)) ]
    acc = len(correct_df)/len(df)
    return acc

def get_acc_simcsesup(df,threshold):
    correct_df = df[ ((df['label']==1) & (df['simcsesup_similarity']>= threshold))|((df['label']==0) & (df['simcsesup_similarity']< threshold)) ]
    acc = len(correct_df)/len(df)
    return acc


if __name__ == '__main__':
    threholds = np.arange(0,1,0.01).tolist()
    df = pd.read_excel('./output/classification_val_dataset_2W_0831_similarity_simcse_result.xlsx')


    best_sbert_acc = 0
    best_sbert_threshold = 0

    best_simcsesup_acc = 0
    best_simcsesup_threshold = 0

    best_simcseunsup_acc = 0
    best_simcseunsup_threshold = 0

    for threhold in threholds:
        sbert_acc = get_acc_sbert(df,threhold)
        simcsesup_acc = get_acc_simcsesup(df, threhold)
        simcseunsup_acc = get_acc_simcseunsup(df, threhold)


        if best_sbert_acc < sbert_acc:
            best_sbert_acc = sbert_acc
            best_sbert_threshold = threhold

        if best_simcsesup_acc < simcsesup_acc:
            best_simcsesup_acc = simcsesup_acc
            best_simcsesup_threshold = threhold

        if best_simcseunsup_acc < simcseunsup_acc:
            best_simcseunsup_acc = simcseunsup_acc
            best_simcseunsup_threshold = threhold


    print('best_sbert_threshold: %.4f -------best_sbert_acc:%.4f'%(best_sbert_threshold,best_sbert_acc))
    print('best_simcsesup_threshold: %.4f -------best_simcsesup_acc:%.4f' % (best_simcsesup_threshold, best_simcsesup_acc))
    print('best_simcseunsup_threshold: %.4f -------best_simcseunsup_acc:%.4f' % (best_simcseunsup_threshold, best_simcseunsup_acc))

