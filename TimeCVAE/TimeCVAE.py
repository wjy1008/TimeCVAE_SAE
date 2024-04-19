import os, warnings, sys
from re import T

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore')
import torch
from SAE import SAEModel1
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Conv1DTranspose, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from vae_base import BaseVariationalAutoencoder, Sampling
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score


def dtw_distance(series1, series2):
    """计算两个时间序列之间的DTW距离"""
    distance, path = fastdtw(series1, series2, dist=euclidean)
    return distance
def calculate_dtw_distances(original_data, virtual_data):
    """计算虚拟样本与原始样本之间的平均DTW距离"""
    num_virtual_samples = len(virtual_data)
    dtw_distances = np.zeros(num_virtual_samples)

    for i in range(num_virtual_samples):
        distances = [dtw_distance(virtual_data[i], original_data[j]) for j in range(len(original_data))]
        dtw_distances[i] = np.mean(distances)

    return dtw_distances
class TimecVariationalAutoencoderConv(BaseVariationalAutoencoder):
    def __init__(self,
                 hidden_layer_sizes,
                 seq_len1,
                 feat_dim1,
                 conditional,
                 mean_y=0,
                 trend_poly=0,
                 num_gen_seas=0,
                 custom_seas=None,
                 use_scaler=False,
                 use_residual_conn=True,
                 seed=123,
                 **kwargs
                 ):
        tf.random.set_seed(seed)
        super(TimecVariationalAutoencoderConv, self).__init__(**kwargs)
        self.seq_len1 = seq_len1
        self.feat_dim1 = feat_dim1
        self.conditional=conditional
        self.mean_y = mean_y
        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.num_gen_seas = num_gen_seas
        self.custom_seas = custom_seas
        self.use_scaler = use_scaler
        self.use_residual_conn = use_residual_conn
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.seed=seed

    def _get_encoder(self):
        if self.conditional:
            encoder_inputs1 = Input(shape=(self.seq_len, self.feat_dim), name='encoder_input1')
            encoder_inputs = Input(shape=(self.seq_len1, self.feat_dim1), name='encoder_input')
            x = tf.concat((encoder_inputs1, encoder_inputs), axis=2)
            for i, num_filters in enumerate(self.hidden_layer_sizes):
                x = Conv1D(
                    filters=num_filters,
                    kernel_size=3,
                    strides=2,
                    activation='relu',
                    padding='same',
                    name=f'enc_conv_{i}')(x)

            x = Flatten(name='enc_flatten')(x)

            # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
            self.encoder_last_dense_dim = x.get_shape()[-1]

            z_mean = Dense(self.latent_dim, name="z_mean")(x)
            z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

            encoder_output = Sampling()([z_mean, z_log_var])
            self.encoder_output = encoder_output

            encoder = Model([encoder_inputs1,encoder_inputs], [z_mean, z_log_var, encoder_output], name="encoder")

            # encoder.summary()
        else:
            encoder_inputs = Input(shape=(self.seq_len1, self.feat_dim1), name='encoder_input')
            x=encoder_inputs
            for i, num_filters in enumerate(self.hidden_layer_sizes):
                x = Conv1D(
                    filters=num_filters,
                    kernel_size=3,
                    strides=2,
                    activation='relu',
                    padding='same',
                    name=f'enc_conv_{i}')(x)

            x = Flatten(name='enc_flatten')(x)

            # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
            self.encoder_last_dense_dim = x.get_shape()[-1]

            z_mean = Dense(self.latent_dim, name="z_mean")(x)
            z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

            encoder_output = Sampling()([z_mean, z_log_var])
            self.encoder_output = encoder_output

            encoder = Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        return encoder

    def _get_decoder(self):
        if self.conditional:
            decoder_inputs = Input(shape=(self.latent_dim), name='decoder_input')
            decoder_inputs1 = Input(shape=(self.seq_len1, self.feat_dim1), name='decoder_input1')
            decoder_inputs12 = Flatten(name='enc_flatten')(decoder_inputs1)
            x = tf.concat((decoder_inputs, decoder_inputs12), axis=1)
            outputs = self.level_model(x)

            # trend polynomials
            if self.trend_poly is not None and self.trend_poly > 0:
                trend_vals = self.trend_model(x)
                if outputs is None:
                    outputs = trend_vals,
                else:
                    outputs = outputs + trend_vals
            # custom seasons
            if self.custom_seas is not None and len(self.custom_seas) > 0:
                cust_seas_vals = self.custom_seasonal_model(x)
                if outputs is None:
                    outputs = cust_seas_vals,
                else:
                    outputs = outputs + cust_seas_vals

            if self.use_residual_conn:
                residuals = self._get_decoder_residual(x)
                outputs = residuals if outputs is None else outputs + residuals

            if self.use_scaler and outputs is not None:
                scale = self.scale_model(x)
                outputs *= scale

            # outputs = Activation(activation='sigmoid')(outputs)

            if outputs is None:
                raise Exception('''Error: No decoder model to use. 
                        You must use one or more of:
                        trend, generic seasonality(ies), custom seasonality(ies), and/or residual connection. ''')

            # decoder = Model(decoder_inputs, [outputs, freq, phase, amplitude], name="decoder")
            decoder = Model([decoder_inputs,decoder_inputs1], outputs, name="decoder")
        else:
            decoder_inputs = Input(shape=(self.latent_dim), name='decoder_input')
            x =decoder_inputs
            x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation='relu')(x)
            x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)

            for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
                x = Conv1DTranspose(
                    filters=num_filters,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation='relu',
                    name=f'dec_deconv_{i}')(x)

            # last de-convolution
            x = Conv1DTranspose(
                filters=self.feat_dim,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu',
                name=f'dec_deconv__{i + 1}')(x)

            x = Flatten(name='dec_flatten')(x)
            x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
            self.decoder_outputs = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
            decoder = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        return decoder

    def level_model(self, z):
        level_params = Dense(self.feat_dim, name="level_params", activation='relu')(z)
        level_params = Dense(self.feat_dim, name="level_params2")(level_params)
        level_params = Reshape(target_shape=(1, self.feat_dim))(level_params)  # shape: (N, 1, D)

        ones_tensor = tf.ones(shape=[1, self.seq_len, 1], dtype=tf.float32)  # shape: (1, T, D)

        level_vals = level_params * ones_tensor
        # print('level_vals', tf.shape(level_vals))
        return level_vals

    def scale_model(self, z):
        scale_params = Dense(self.feat_dim, name="scale_params", activation='relu')(z)
        scale_params = Dense(self.feat_dim, name="scale_params2")(scale_params)
        scale_params = Reshape(target_shape=(1, self.feat_dim))(scale_params)  # shape: (N, 1, D)

        scale_vals = tf.repeat(scale_params, repeats=self.seq_len, axis=1)  # shape: (N, T, D)
        # print('scale_vals', tf.shape(scale_vals))
        return scale_vals

    def trend_model(self, z):
        trend_params = Dense(self.feat_dim * self.trend_poly, name="trend_params", activation='relu')(z)
        trend_params = Dense(self.feat_dim * self.trend_poly, name="trend_params2")(trend_params)
        trend_params = Reshape(target_shape=(self.feat_dim, self.trend_poly))(trend_params)  # shape: N x D x P
        # print("trend params shape", trend_params.shape)
        # shape of trend_params: (N, D, P)  P = num_poly

        lin_space = K.arange(0, float(self.seq_len), 1) / self.seq_len  # shape of lin_space : 1d tensor of length T
        poly_space = K.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0)  # shape: P x T
        # print('poly_space', poly_space.shape, poly_space[0])

        trend_vals = K.dot(trend_params, poly_space)  # shape (N, D, T)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])  # shape: (N, T, D)
        trend_vals = K.cast(trend_vals, tf.float32)
        # print('trend_vals shape', tf.shape(trend_vals))
        return trend_vals

    def custom_seasonal_model(self, z):

        N = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[N, self.feat_dim, self.seq_len], dtype=tf.int32)

        all_seas_vals = []
        for i, season_tup in enumerate(self.custom_seas):
            num_seasons, len_per_season = i, season_tup

            season_params = Dense(self.feat_dim * num_seasons)(z)  # shape: (N, D * S)
            season_params = Reshape(target_shape=(self.feat_dim, num_seasons))(season_params)  # shape: (N, D, S)
            # print('\nshape of season_params', tf.shape(season_params))

            season_indexes_over_time = self._get_season_indexes_over_seq(num_seasons, len_per_season)  # shape: (T, )
            # print("season_indexes_over_time shape: ", tf.shape(season_indexes_over_time))

            dim2_idxes = ones_tensor * tf.reshape(season_indexes_over_time, shape=(1, 1, -1))  # shape: (1, 1, T)
            # print("dim2_idxes shape: ", tf.shape(dim2_idxes))

            season_vals = tf.gather(season_params, dim2_idxes, batch_dims=-1)  # shape (N, D, T)
            # print("season_vals shape: ", tf.shape(season_vals))

            all_seas_vals.append(season_vals)

        all_seas_vals = K.stack(all_seas_vals, axis=-1)  # shape: (N, D, T, S)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)  # shape (N, D, T)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0, 2, 1])  # shape (N, T, D)
        # print('final shape:', tf.shape(all_seas_vals))
        return all_seas_vals

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        curr_len = 0
        season_idx = []
        curr_idx = 0
        while curr_len < self.seq_len:
            reps = len_per_season if curr_len + len_per_season <= self.seq_len else self.seq_len - curr_len
            season_idx.extend([curr_idx] * reps)
            curr_idx += 1
            if curr_idx == num_seasons: curr_idx = 0
            curr_len += reps
        return season_idx

    def _get_decoder_residual(self, x):

        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation='relu')(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu',
                name=f'dec_deconv_{i}')(x)

        # last de-convolution
        x = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name=f'dec_deconv__{i + 1}')(x)

        x = Flatten(name='dec_flatten')(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        residuals = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        return residuals

    def train_step(self, data):
        if self.conditional:
            with tf.GradientTape() as tape:
                X, y_Z = data
                y=y_Z[0]
                mean_y=y_Z[1]
                log_var_y = y_Z[2]
                z_mean, z_log_var, z = self.encoder([X, y])

                reconstruction = self.decoder([z, y])

                reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

                kl_loss = -0.5 * (1 - (log_var_y - z_log_var) - (
                            tf.square(z_mean - mean_y) + tf.exp(z_log_var))/ tf.exp(log_var_y))
                kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
                # kl_loss = kl_loss / self.latent_dim

                total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)

            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
        else:

            with tf.GradientTape() as tape:
                X, y = data
                z_mean, z_log_var, z = self.encoder(y)
                reconstruction = self.decoder(z)

                reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
                # kl_loss = kl_loss / self.latent_dim

                total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)

            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def call(self,data,num_samples=2):
        if self.conditional:
            x, y = data
            z_mean, z_log_var, encoder_output = self.encoder([x, y])

            generated_samples = []
            for _ in range(num_samples):
                x_decoded = self.decoder([encoder_output, y])
                if len(x_decoded.shape) == 1:
                    x_decoded = x_decoded.reshape((1, -1))
                generated_samples.append(x_decoded)
            return generated_samples
        else:
            x, y = data
            z_mean, z_log_var, encoder_output = self.encoder(y)
            x_decoded = self.decoder(encoder_output)
            if len(x_decoded.shape) == 1: x_decoded = x_decoded.reshape((1, -1))
            return x_decoded, z_mean,z_log_var


#####################################################################################################
#####################################################################################################


if __name__ == '__main__':
    TRAIN_SIZE = 500
    # 数据读取
    data = pd.read_csv(r"C:\Users\wjy\Desktop\daima\shujuji\SRU_data.txt", sep='\s+')
    data = np.array(data)
    x = data[:, 0:len(data[0]) - 2]
    y = data[:, len(data[0]) - 1]

    untrainInputdata = np.zeros(shape=[10071, 20], dtype=float)
    targetOutputdata = np.zeros(shape=[10071, 1])

    for i in range(9, 10080):
        untrainInputdata[i - 9, :] = [x[i, 0], x[i - 5, 0], x[i - 7, 0], x[i - 9, 0],
                                      x[i, 1], x[i - 5, 1], x[i - 7, 1], x[i - 9, 1],
                                      x[i, 2], x[i - 5, 2], x[i - 7, 2], x[i - 9, 2],
                                      x[i, 3], x[i - 5, 3], x[i - 7, 3], x[i - 9, 3],
                                      x[i, 4], x[i - 5, 4], x[i - 7, 4], x[i - 9, 4]]
        targetOutputdata[i - 9] = y[i]

    # X_train, X_test, y_train, y_testTrue

    train_x = untrainInputdata[0:500, :]
    train_y = targetOutputdata[0:500]
    test_X = untrainInputdata[500:700, :]
    test_y = targetOutputdata[500:700]
    train_X = train_x.reshape(-1,5, 20)
    train_Y = train_y.reshape(-1, 5, 1)
    train_X = tf.convert_to_tensor(train_X)
    train_Y = tf.convert_to_tensor(train_Y)
    x_train = tf.cast(train_X, dtype=tf.float32)
    y_train = tf.cast(train_Y, dtype=tf.float32)
    # random_indices = np.random.choice(500, size=250, replace=False)
    # random_train_x = train_x[random_indices]
    # random_train_y = train_y[random_indices]
    # random_train_X = random_train_x.reshape(-1, 5, 20)
    # random_train_Y = random_train_y.reshape(-1, 5, 1)
    # random_train_X = tf.convert_to_tensor(random_train_X)
    # random_train_Y = tf.convert_to_tensor(random_train_Y)
    # random_x_train = tf.cast(random_train_X, dtype=tf.float32)
    # random_y_train = tf.cast(random_train_Y, dtype=tf.float32)
    N1, T1, D1 = y_train.shape
    N, T, D = x_train.shape

    vae1 =TimecVariationalAutoencoderConv(
        seq_len=T,
        feat_dim=D,
        seq_len1=T1,
        feat_dim1=D1,
        latent_dim=2,
        hidden_layer_sizes=[100, 200],
        seed=1024,
        conditional=False
    )
    vae1.compile(optimizer=Adam())
    # vae.summary() ; sys.exit()

    r1 = vae1.fit(x_train, y_train, epochs=8, batch_size=64, shuffle=True)

    x_decoded1,z_mean,z_log_var = vae1.predict((x_train, y_train))
    print(z_mean)
    vae =TimecVariationalAutoencoderConv(
        seq_len=T,
        feat_dim=D,
        seq_len1=T1,
        feat_dim1=D1,
        latent_dim=2,
        hidden_layer_sizes=[100, 200],
        trend_poly=2,
        custom_seas=[3,2,2,2,2,2],
        seed=1024,
        conditional=True
    )
    vae.compile(optimizer=Adam())
    # vae.summary() ; sys.exit()
    r = vae.fit(x_train, (y_train,z_mean,z_log_var),  epochs=50, batch_size=64, shuffle=True)
    x_decoded = vae.predict((x_train,y_train))
    sample_array = np.array(x_decoded)
    x_test = np.array(x_decoded).reshape(-1, 20)
    print(x_test)
    dtw_distances = calculate_dtw_distances(x_train, x_test)
    # 确定筛选阈值
    threshold = np.percentile(dtw_distances,75)
    # 筛选出相似度最低（DTW距离最小）的虚拟样本
    selected_virtual_data = x_test[dtw_distances >= threshold]

    x_test = np.array(selected_virtual_data)
    x_test = np.array(x_test)
    x_test = x_test.reshape(-1, 20)
    print(x_test.shape)

    mdl = SAEModel1(AEList=[20, 16, 12, 6], sup_epoch=100, unsp_epoch=250, unsp_batch_size=50, sp_batch_size=20,
                    sp_lr=0.02, unsp_lr=0.01, device=torch.device('cuda:0'), seed=1024, ss_lambda=1.6).fit(train_x,
                                                                                                           train_y)
    output_train = mdl.predict(train_x)
    output_test = mdl.predict(test_X)
    output_tes1 = mdl.predict(x_test)
    x_test1 = np.vstack((train_x, x_test))
    y_test1 = np.concatenate((train_y, output_tes1))
    mdl1 = SAEModel1(AEList=[20, 16, 12, 6], sup_epoch=100, unsp_epoch=250, unsp_batch_size=50, sp_batch_size=20,
                     sp_lr=0.01, unsp_lr=0.01, device=torch.device('cuda:0'), seed=1024, ss_lambda=1.6).fit(x_test1,
                                                                                                            y_test1)
    output_test11 = mdl1.predict(test_X)
    test_rmse = np.sqrt(mean_squared_error(output_test, test_y))
    test_rmse1 = np.sqrt(mean_squared_error(output_test11, test_y))
    test_r2 = r2_score(output_test, test_y)
    r2 = r2_score(output_test11, test_y)

    print('test_rmse = ' + str(round(test_rmse, 5)))
    print("rmse:", test_rmse1)
    print("test_r2:", test_r2)
    print("r2", r2)
