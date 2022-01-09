import keras.models
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


def load_dataset(path):
    return pd.read_csv(path).sort_values('ts')


def create_sequence(df, lookback, features):
    x, y = [], []
    temp_df = df[features]
    scaled_df = scale_values(temp_df.to_numpy())
    for i in range(df.shape[0]-lookback):
        x.append(scaled_df[i:(i+lookback)])
        y.append(df.iloc[i+lookback]['label'])

    return np.array(x), np.array(y)


def create_model(input_shape, num_labels):
    gru = Sequential()
    gru.add(GRU(128, return_sequences=True, input_shape=input_shape))
    gru.add(GRU(128, return_sequences=True))
    gru.add(GRU(128))
    gru.add(Dense(num_labels, activation='sigmoid'))
    return gru


def scale_values(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)


def save_model_and_loss(model, history):
    curr_date = datetime.now()
    dir_name = curr_date.year.__str__() \
               + "-" \
               + curr_date.month.__str__().zfill(2) \
               + "-" \
               + curr_date.day.__str__().zfill(2) \
               + "at" \
               + curr_date.time().hour.__str__().zfill(2) \
               + "-" \
               + curr_date.time().minute.__str__().zfill(2)

    save_path = os.path.join(os.path.abspath(os.getcwd()), dir_name)
    os.mkdir(save_path)
    model.save_weights(save_path + "/model.h5")

    plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path + "/accuracy.png")
    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path + "/loss.png")


if __name__ == '__main__':
    url = sys.argv[1]
    mode = sys.argv[2]

    if url is None or mode is None:
        print("Specify url or mode of the dataset")
    else:
        if mode == "train":
            df = load_dataset(url)
            features = ['id.orig_p', 'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts',
                        'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'proto_icmp',
                        'proto_tcp', 'proto_udp', 'conn_state_OTH', 'conn_state_REJ',
                        'conn_state_RSTO', 'conn_state_RSTOS0', 'conn_state_RSTR',
                        'conn_state_RSTRH', 'conn_state_S0', 'conn_state_S1', 'conn_state_S2',
                        'conn_state_S3', 'conn_state_SF', 'conn_state_SH', 'conn_state_SHR',
                        'service_-', 'service_dhcp', 'service_dns', 'service_http',
                        'service_irc', 'service_ssh', 'service_ssl']

            df_node = df[df['id.orig_h'] == '192.168.1.198']

            x, y = create_sequence(df_node, 25, features)

            y_encoded = pd.get_dummies(pd.Series(y)).to_numpy()

            x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=.2, random_state=42)

            gru = create_model((x_train.shape[1], x_train.shape[2]), y_train.shape[1])
            gru.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

            trained_model = gru.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=1, shuffle=False)

            loss = trained_model.history['loss']
            val_loss = trained_model.history['val_loss']

            save_model_and_loss(gru, trained_model)

        elif mode == "evaluate":
            print("Start")

            df = load_dataset(url)

            features = ['id.orig_p', 'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts',
                        'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'proto_icmp',
                        'proto_tcp', 'proto_udp', 'conn_state_OTH', 'conn_state_REJ',
                        'conn_state_RSTO', 'conn_state_RSTOS0', 'conn_state_RSTR',
                        'conn_state_RSTRH', 'conn_state_S0', 'conn_state_S1', 'conn_state_S2',
                        'conn_state_S3', 'conn_state_SF', 'conn_state_SH', 'conn_state_SHR',
                        'service_-', 'service_dhcp', 'service_dns', 'service_http',
                        'service_irc', 'service_ssh', 'service_ssl']

            df_node = df[df['id.orig_h'] == '192.168.1.198']

            x, y = create_sequence(df_node, 25, features)

            y_encoded = pd.get_dummies(pd.Series(y)).to_numpy()

            _, x_test, _, y_test = train_test_split(x, y_encoded, test_size=.2, random_state=42)

            model = create_model((x_test.shape[1], x_test.shape[2]), y_test.shape[1])
            model.load_weights(r"C:\dev\anomaly_detection\2022-01-09at18-54\model.h5")

            a = np.argmax(y_test, axis=1)
            b = np.argmax(model.predict(x_test), axis=1)
            print(classification_report(a, b))

        else:
            print("Unknown mode")


        print("Finished..")

