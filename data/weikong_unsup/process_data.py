import pandas as pd
import random
if __name__ == '__main__':

    df = pd.read_excel('shanghai_template_train_2021-0820_new_new.xlsx')
    df.fillna('',inplace=True)
    main_quesitons = df['main_question'].values.tolist()
    simi_questions = df['simi_question'].values.tolist()

    all_questions = []
    for main_quesiton,simi_question in zip(main_quesitons,simi_questions):
        main_quesiton = main_quesiton.strip('\n')
        if len(main_quesiton) > 0:
            all_questions.append(main_quesiton)
        temps = simi_question.replace('\n','').split('||')
        for temp in temps:
            if len(temp) > 0:
                all_questions.append(temp)

    print(len(all_questions))
    all_questions = list(set(all_questions))
    print(all_questions[0:20])
    print(len(all_questions))

    random.shuffle(all_questions)
    print(all_questions[0:20])

    new_df = pd.DataFrame()
    new_df['question'] = all_questions
    writer = pd.ExcelWriter('train_weikong_dataset.xlsx')
    new_df.to_excel(writer,index=False)
    writer.save()