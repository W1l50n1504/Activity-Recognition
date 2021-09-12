import pandas as pd

sensor1 = "C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/rawdata/sensor_fOQPV_RXTT2kPgMrV25K-j.txt"
sensor2 = "C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/rawdata/sensor_frKGD5xeRe67UghTP3z5A3.txt"
sensor3 = "C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/rawdata/sensor_fUOAIa1pTpm1AGDfORuywD.txt"

dataPath = 'C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/data/0.csv'
accel = 'C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/data/accel.csv'
gyro = 'C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/data/gyro.csv'
accelMod = 'C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/data/accel_modificato.csv'
gyroMod = 'C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/data/gyro_modificato.csv'
# x,y,z,canc,data,ora
cols2 = ['x', 'y', 'z', 'canc', 'data', 'ora']


def clear(sensor):
    cols = ['sensor', 'data']

    s = pd.read_csv(sensor, sep='"', header=None)
    s = s.drop(s.columns[0], axis=1)
    rslt_df1 = pd.DataFrame(columns=cols, dtype='float32')
    rslt_df2 = pd.DataFrame(columns=cols, dtype='float32')

    rslt_df1 = s[s[1] == 'ACCELEROMETER']
    rslt_df2 = s[s[1] == 'GYROSCOPE']

    rslt_df1 = rslt_df1.drop(rslt_df1.columns[1], axis=1)
    rslt_df1 = rslt_df1.drop(rslt_df1.columns[2], axis=1)

    rslt_df2 = rslt_df2.drop(rslt_df2.columns[1], axis=1)
    rslt_df2 = rslt_df2.drop(rslt_df2.columns[2], axis=1)

    rslt_df1 = rslt_df1.drop(rslt_df1.columns[0], axis=1)
    rslt_df2 = rslt_df2.drop(rslt_df2.columns[0], axis=1)

    print(rslt_df1)
    print(rslt_df2)

    rslt_df1.to_csv(accel, index=False, index_label=False)
    rslt_df2.to_csv(gyro, index=False, index_label=False)


def puliziaSensore(sensore, save):
    columns1 = ['x', 'y', 'z', 'ora']
    df = pd.read_csv(sensore, sep=' |,', engine='python')  # , dtype='float64')

    print(df)
    ded = pd.DataFrame(columns=columns1)

    ded['x'] = df['x'].replace({'"': ''}, regex=True)
    ded['y'] = df['y']
    ded['z'] = df['z']
    ded['ora'] = df['ora'].replace({'"': ''}, regex=True)

    print(ded)
    ded.to_csv(save, index=False)


def join():
    columns = ['xacc', 'yacc', 'zacc', 'xgyro', 'ygyro', 'zgyro']
    #fixa walk rifai
    accdata = "C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/activity/laying/sub1_acc.csv"
    gyrodata = "C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/activity/laying/sub1_gyro.csv"
    final = "C:/Users/david/PycharmProjects/Activity-Recognition/com/dataset/ShieldApp/activity/laying/sub3.csv"

    acc = pd.read_csv(accdata)
    gyro = pd.read_csv(gyrodata)

    print(acc)
    print(gyro)

    finaldf = pd.concat([acc, gyro], axis=1).reindex(acc.index)
    finaldf = finaldf.dropna()

    finaldf = finaldf.drop(finaldf['ora'], axis=1)
    print(finaldf)

    finaldf.to_csv(final, index=None)


if __name__ == '__main__':
    #clear(sensor3)

    #puliziaSensore(accel, accelMod)
    #puliziaSensore(gyro, gyroMod)
    #accele = pd.read_csv(accelMod)
    #gyros = pd.read_csv(gyroMod)

    # ordinare i dati per l'orario di cattura
    #accele = accele.sort_values(by='ora')
    #accele.to_csv(accelMod, index=False)
    #gyros = gyros.sort_values(by='ora')
    #gyros.to_csv(gyroMod, index=False)

    # print(accele)
    # print(gyros)
    join()

