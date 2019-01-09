import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

scale_model = MinMaxScaler()
nmf_model = NMF(n_components=500, init='random', random_state=0)

from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.optimizers import Adam

from F1Metric import F1Metric, F1MetricTensorBoard

from scipy import linalg
from numpy import dot

def load_data_kfold(k, category, zero_class):
    ground_truth = pd.read_csv("./datasets/combined-ground-truth.csv")

    topics_features = pd.read_csv("./datasets/combined-instagram-twitter-text-content-topics.csv")
    liwc_features = pd.read_csv("./datasets/combined-instagram-twitter-text-content-liwc.csv")
    text_features = pd.read_csv('./datasets/combined-instagram-twitter-text-content-twitter.csv')

    features = topics_features.merge(liwc_features, on='row ID')
    features = features.merge(text_features, on='row ID')
    features = features.merge(ground_truth[['row ID', category]], on='row ID')
    features.rename(index=str, columns={category: 'class'}, inplace=True)

    features.drop(['row ID'], axis=1, inplace=True)

    X = features.drop('class', axis=1)
    y = np.array((features['class'] == zero_class).astype('int'))

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=.1)

    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train, y_train))

    return folds, X_train, y_train, X_test, y_test


def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    # X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            print('Iteration {}:'.format(i)),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print('fit residual', np.round(fit_residual, 4)),
            print('total residual', np.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y

def load_nmf_features(k, category, zero_class):

    ground_truth = pd.read_csv("./datasets/combined-ground-truth.csv")

    topics_features = pd.read_csv("./datasets/combined-instagram-twitter-text-content-topics.csv")
    liwc_features = pd.read_csv("./datasets/combined-instagram-twitter-text-content-liwc.csv")
    text_features = pd.read_csv('./datasets/combined-instagram-twitter-text-content-twitter.csv')
    image_features = pd.read_csv("./datasets/image-concepts.csv")

    features = topics_features.merge(liwc_features, on='row ID')
    features = features.merge(text_features, on='row ID')
    features = features.merge(image_features, on='row ID', how='left')
    features = features.fillna(0)

    features_scaled = scale_model.fit_transform(features.drop('row ID', axis=1))
    # features_scaled = nmf_model.fit_transform(features_scaled)
    A, H = nmf(features_scaled, 1000)

    features_scaled = pd.DataFrame(dot(A, H))
    features_scaled['row ID'] = features['row ID']

    features_scaled = features_scaled.merge(ground_truth[['row ID', category]], on='row ID')
    features_scaled.rename(index=str, columns={category: 'class'}, inplace=True)

    X = features_scaled.drop(['row ID', 'class'], axis=1)
    y = np.array((features_scaled['class'] == zero_class).astype('int'))

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=.1)

    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train, y_train))

    return folds, X_train, y_train, X_test, y_test


def create_simple_nn_mode(input_size, category, fold, lr = 0.0001):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_size,)))
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(128))
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])


    return model, model.get_weights()


model = None
initial_weights = None

def train_model(epochs=10, batch_size=128, log_dir='./logs', category = 'tf', zero_class = 'Thinking'):
    global model, initial_weights
    print('TRAINING CATEGORY ' + category)

    # folds, X_train, y_train, X_test, y_test = load_data_kfold(10, category, zero_class)

    folds, X_train, y_train, X_test, y_test = load_nmf_features(10, category, zero_class)

    # checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')


    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold ', j + 1)

        if model is None:
            model, initial_weights = create_simple_nn_mode(X_train.shape[1], category=category, fold=j)
        else:
            model.set_weights(initial_weights)

        tboard = F1MetricTensorBoard(log_dir=log_dir + '_fold=' + str(j) , histogram_freq=1, write_graph=True, write_images=False)
        f1_metric = F1Metric()

        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_valid_cv = X_train[val_idx]
        y_valid_cv = y_train[val_idx]

        model.fit(X_train_cv, y_train_cv, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_valid_cv, y_valid_cv),
              callbacks=[f1_metric, tboard])


categories = [('tf','Thinking'), ('ei','Extraversion') ,('si','Sensing') ,('jp','Judging')]

for category, zero_class in categories:
    train_model(epochs=30, log_dir='./logs_nmf/' + category, category = category, zero_class = zero_class)

# train_model(epochs=30)