import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import BorderlineSMOTE
selected_features = [' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Mean',' Fwd Packet Length Std',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean','Fwd IAT Total' ,'Bwd IAT Total','Fwd PSH Flags','Fwd Packets/s',' Bwd Packets/s',' Packet Length Mean',' SYN Flag Count',' Average Packet Size','Active Mean','Idle Mean','label_encoded']


#读取全部数据
def merge_data():
    df1 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df2 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    df3 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Friday-WorkingHours-Morning.pcap_ISCX.csv")
    df4 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv")
    df5 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    df6 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    df7 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Tuesday-WorkingHours.pcap_ISCX.csv")
    df8 = pd.read_csv("D:\IDSWORK\MachineLearningCVE\Wednesday-workingHours.pcap_ISCX.csv")

    frame=[df1,df2,df3,df4,df5,df6,df7,df8]
    col_list=df1.columns
    for df in frame:
        df[' Label'] = df[' Label'].replace(['Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection'], 'Web_Attack')
        #
        df[' Label'] = df[' Label'].replace(['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DDoS','DoS GoldenEye', 'Heartbleed'], 'DoS/DDos')
        #
        df[' Label'] = df[' Label'].replace(['FTP-Patator', 'SSH-Patator'], 'Brute_Force')
        #
        df[' Label'] = df[' Label'].replace(['Bot'], 'Botnet')
        df=df.drop([0])

    result=pd.concat(frame)
    result.columns=col_list
#替换空值为0，方便处理
    result.replace([-np.inf,np.inf],np.nan,inplace=True)
    result.fillna(0,inplace=True)
    result=data_clear(result)
    result=label_encode(result)
    result.to_csv("IDS2017-3.csv",index=False)
def data_clear(df):
    zero_col=df.columns[(df==0).all(axis=0)]
    if not zero_col.empty:
        for col in zero_col:
            df=df.drop(columns=col)
        print(zero_col)
    else:
        print("No free col")
    return df
def label_encode(df):
    # 替换label列中的值
    # 初始化LabelEncoder对象
    le = LabelEncoder()
    # 对'label'列进行标签编码
    df['label_encoded'] = le.fit_transform(df[' Label'])

    # 打印编码后的DataFrame
    #print(df)
    # Packet Attack Distribution
    print('查看数据集标签特征')
    print(df[' Label'].value_counts())
    # 删除'label'列
    df = df.drop(columns=[' Label'])
    print(df['label_encoded'].value_counts())
    return df
def split_data(df):
    Data = df.iloc[1:, :-1]  # 所有行，除了最后一列的数据
    Label = df.iloc[1:, -1]
    x_train, x_test, y_train, y_test = train_test_split(Data, Label, test_size=0.95, random_state=0)
    x_train = pd.concat([x_train,y_train],axis=1)
    x_test = pd.concat([x_test,y_test],axis=1)
    #x_train.to_csv('train.csv',index=False)
    #x_test.to_csv('test.csv',index=False)
    x_train.to_csv('IDS_test_real.csv', index=False)
    print(y_train.value_counts())
def normalize_data(df,df2):
    x_train = df.iloc[1:, :-1]  # 所有行，除了最后一列的数据
    y_train = df.iloc[1:, -1]

    x_test = df2.iloc[1:, :-1]  # 所有行，除了最后一列的数据
    y_test = df2.iloc[1:, -1]

    scaler = MinMaxScaler()
    #
    # 进行归一化
    cols = x_train.columns
    sc_train = scaler.fit_transform(x_train)
    sc_test = scaler.transform(x_test)

    #保存
    sc_traindf = pd.DataFrame(sc_train, columns=cols)
    sc_testdf = pd.DataFrame(sc_test, columns=cols)
    #y_train = pd.Series(y_train, name="label_encoded")
    #y_test = pd.Series(y_test, name="label_encoded")
    sc_traindf = sc_traindf.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    sc_testdf =sc_testdf.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    sc_traindf=pd.concat([sc_traindf,y_train],axis=1)
    sc_testdf = pd.concat([sc_testdf, y_test], axis=1)
    #sc_traindf = sc_traindf.drop(0)
    # for label in cols:
    #     if label not in selected_features:
    #         sc_traindf.drop(columns=label,inplace=True)
    sc_traindf.to_csv('Raw_train.csv',index=False)
    sc_testdf.to_csv('IDS_test.csv',index=False)
def random_under_sample(df):
    Data = df.iloc[1:, :-1]  # 所有行，除了最后一列的数据
    Label = df.iloc[1:, -1]
    sampling_strategy={0:9999,1:10000,2:0,3:10000,4:270,5:10000,6:10000}
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy,random_state=1)
    x_resample,y_resample=rus.fit_resample(Data,Label)
    x_resample=pd.DataFrame(x_resample,columns=Data.columns)
    y_resample_series=pd.Series(y_resample,name="label_encoded")
    #x_resample, y_resample_series = shuffle(x_resample, y_resample_series, random_state=42)
    # shuffled_indices = np.random.permutation(len(x_resample))
    # X_res_shuffled = x_resample.iloc[shuffled_indices].reset_index(drop=True)
    # y_res_shuffled = y_resample_series.iloc[shuffled_indices].reset_index(drop=True)
    x_resample=pd.concat([x_resample,y_resample_series],axis=1)
    #x_resample=x_resample.sample(frac=1, replace=False,random_state=1,ignore_index=True)
    x_resample.to_csv('IDS_dtrain.csv',index=False)
    y=x_resample.iloc[1:,-1]
    print(y.value_counts())
def over_sample(df):
    Data = df.iloc[1:, :-1]  # 所有行，除了最后一列的数据
    Label = df.iloc[1:, -1]
    sampling_strategy = { 1: 13290, 2: 97190,  4: 270, 6: 15240}
    bsmote=BorderlineSMOTE(sampling_strategy=sampling_strategy,random_state=10,k_neighbors=5)
    x_resample, y_resample = bsmote.fit_resample(Data, Label)
    x_resample = pd.DataFrame(x_resample, columns=Data.columns)
    y_resample_series = pd.Series(y_resample, name="label_encoded")

    # x_resample, y_resample_series = shuffle(x_resample, y_resample_series, random_state=42)
    # shuffled_indices = np.random.permutation(len(x_resample))
    # X_res_shuffled = x_resample.iloc[shuffled_indices].reset_index(drop=True)
    # y_res_shuffled = y_resample_series.iloc[shuffled_indices].reset_index(drop=True)
    x_resample = pd.concat([x_resample, y_resample_series], axis=1)
    # x_resample=x_resample.sample(frac=1, replace=False,random_state=1,ignore_index=True)
    x_resample.to_csv('IDS_over_resample.csv', index=False)
    y = x_resample.iloc[1:, -1]
    print(y.value_counts())
if __name__ == '__main__':
    #merge_data()
    #raw_data = merge_data()
    file = 'IDS_oversample_train.csv'
    raw_data=pd.read_csv(file)
    # file2 = 'test.csv'
    # test_data=pd.read_csv(file2)
    #raw_data.to_csv(file,index=False)
    #split_data(raw_data)
    #normalize_data(raw_data,test_data)
    #print(raw_data["label_encoded"].value_counts())
    random_under_sample(raw_data)
    #over_sample(raw_data)

