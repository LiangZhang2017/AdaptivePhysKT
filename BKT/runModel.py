
import math
from BKT.bkt import ClassicBKT
import numpy as np
from sklearn.utils import shuffle
import csv

def crossValidate(model,exper_data,n_folds):
    print('crossValidate')
    # shuffle the dataframes by student_id
    exper_data=shuffle(exper_data,random_state=42)
    print(exper_data)
    interaction=exper_data.to_numpy()
    #print(interaction)

    print("the length of exper_data is {}".format(len(exper_data)))

    #make the folds
    bin_size=float(len(interaction))/n_folds
    bins=[]

    for i in range(n_folds):
        start=int(math.floor(i*bin_size))
        end=int(math.floor((i+1)*bin_size))
        bin=[]

        for index_bin in range(start,end):
            # print("the elements is:")
            # print(interaction[index_bin])
            bin=bin+([interaction[index_bin].tolist()])

        bins.append(bin)
        print("Submissions in bin" + str(i) +": "+ str(len(bin)))

    # do crossvalidation
    print(len(bins))

    mae_list = []
    rmse_list = []
    auc_list = []

    for i in range(n_folds):
        traning_set=[]
        test_set=[]

        for j in range(n_folds):
            if j!=i:
                for bin in bins[j]:
                    traning_set.append(bin)
            else:
                test_set=bins[j]

        print("Crossvalidation is "+str(i))
        print("length of Training set is: "+str(len(traning_set)))

        model.fit(traning_set,i)
        mae,rmse=model.writePrediction(test_set,i)

        mae_list.append(mae)
        rmse_list.append(rmse)
        # auc_list.append(auc)

    return mae_list,rmse_list

def main(m,n_folds,exper_data,output_path):
    print("run Model main")
    print("exper_data is : ", exper_data)

    if m=="bkt":
        bkt=ClassicBKT()
        bkt.generateParams(output_path)
        bkt.generateSkillMap(exper_data)
        mae_list, rmse_list=crossValidate(bkt,exper_data,int(n_folds))

        writer = csv.writer(open(output_path + '/' + 'all_Results'+ '.csv', 'w'))
        writer.writerow(['Metric','Value'])
        # line1=['AUC',sum(auc_list)/n_folds]
        line2 =['rmse', sum(rmse_list)/n_folds]
        line3=['mae', sum(mae_list)/n_folds]
        # writer.writerow(line1)
        writer.writerow(line2)
        writer.writerow(line3)