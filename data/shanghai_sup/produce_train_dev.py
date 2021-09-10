import pandas as pd
import random
from tqdm import tqdm

def produce_dataset(path,save_path):
    df = pd.read_excel(path)
    if 'main_question' in df.columns.tolist():
        df = df[['main_question','simi_question']]
        df['question'] = df['main_question'].str.cat(df['simi_question'], sep='||')
    else:
        df = df[['simi_question']]
        df['question'] = df['simi_question']
    df.dropna(inplace=True)
    print(len(df))
    texts_a = []
    texts_b = []
    questions = df['question'].values.tolist()

    lens = []
    max_len = 0
    for question in questions:
        temp = question.strip('||').split('||')
        temp = list(set(temp))
        temp.sort()
        lens.append(len(temp))
        if len(temp) > max_len:
            max_len = len(temp)
    lens.sort()
    for index in tqdm(range(max_len)):
        for question in questions:
            temp = question.strip('||').split('||')
            temp = list(set(temp))
            while '' in temp:
                temp.remove('')
            temp.sort()
            if index + 1 < len(temp):
                texts_a.append(temp[index])
                texts_b.append(temp[index + 1])
            if index + 1 >= len(temp):
                r = random.randint(0,len(temp) - 1)
                texts_a.append(temp[r])
                r = random.randint(0,len(temp) - 1)
                texts_b.append(temp[r])

    new_df = pd.DataFrame({'text_a':texts_a,'text_b':texts_b})
    writer = pd.ExcelWriter(save_path)
    new_df.to_excel(writer,index=False)
    writer.save()




if __name__ == '__main__':
    file_path = 'shanghai_template_inside_2021-0907.xlsx'
    save_path = 'train_2021-0907.xlsx'
    produce_dataset(file_path,save_path)

    file_path = 'shanghai_template_outside_2021-0907.xlsx'
    save_path = 'dev_2021-0907.xlsx'
    produce_dataset(file_path, save_path)
