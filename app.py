import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

st.title("APLIKASI BREAST CANCER WISCONSIN DIAGNOSTIC")
st.write("##### Nama  : Yanuar Pratama Dicha Putra")
st.write("##### Nim   : 210411100190")
st.write("##### Kelas : Penambangan Data B ")
st.write("##### E-mail : yanuarprtamadp@gmail.com")
st.write("##### Github : https://github.com/yanuarprtama")
st.write("##### Link Colab : https://colab.research.google.com/drive/1f9ZeJ5kNZ_adGFuzI5wYgPKlsy_xJyow?usp=sharing")

description, preprocessing, pca_menu, modeling, implementation = st.tabs(
    ["Description", "Preprocessing", "PCA", "Modeling", "Implementation"])


with description:
    st.write("""# Deskripsi Aplikasi""")
    st.write(" Aplikasi ini digunakan untuk memprediksi kanker payudara ganas atau jinak dengan memasukkan hasil pemeriksaan maka akan keluar dengan hasil akurasi dari 92 sampai 95 dengan menggunakan metode Naive Bayes, KNN, Decision Tree, ANNBP ")
    st.write(" Aplikasi ini dibuat sebagai syarat lulusnya matakuliah Penambangan Data kelas B ")
    st.write("###### Dataset yang digunakan Adalah : ")
    st.write("###### Breast Cancer Wisconsin Diagnostic Dataset (Kumpulan Data Diagnostik Kanker Payudara Wisconsin) ")
    st.write("###### Sumber Dataset : https://www.kaggle.com/datasets/utkarshx27/breast-cancer-wisconsin-diagnostic-dataset")
    st.write("""# Deskripsi Data""")
    st.write(" Data ini berbentuk file csv dengan jumlah kolom yaitu 32 kolom dan jumlah data dari dataset sebanyak 569 Data")
    st.write(" y. The outcomes. A factor with two levels denoting whether a mass is malignant ('M') or benign ('B'). ")
    st.write(" x. The predictors. A matrix with the mean, standard error and worst value of each of 10 nuclear measurements on the slide, for 30 total features per biopsy: ")
    st.write(" radius. Nucleus radius (mean of distances from center to points on perimeter). ")
    st.write(" texture. Nucleus texture (standard deviation of grayscale values). ")
    st.write(" perimeter. Nucleus perimeter. ")
    st.write(" area. Nucleus area. ")
    st.write(" smoothness. Nucleus smoothness (local variation in radius lengths). ")
    st.write(" compactness. Nucleus compactness (perimeter^2/area - 1). ")
    st.write(" concavity, Nucleus concavity (severity of concave portions of the contour). ")
    st.write(" concave_pts. Number of concave portions of the nucleus contour. ")
    st.write(" symmetry. Nucleus symmetry. ")
    st.write(" fractal_dim. Nucleus fractal dimension ('coastline approximation' -1). ")
    st.write("Informasi Kolom ")
    st.write("1) x.radius_mean : Mean radius of the tumor cells ")
    st.write("2) x.texture_mean : Mean texture of the tumor cells ")
    st.write("3) x.perimeter_mean : Mean perimeter of the tumor cells ")
    st.write("4) x.area_mean : Mean area of the tumor cells ")
    st.write("5) x.smoothness_mean : Mean smoothness of the tumor cells ")
    st.write("6) x.compactness_mean : Mean compactness of the tumor cells ")
    st.write("7) x.concavity_mean : Mean concavity of the tumor cells ")
    st.write("8) x.concave_points_mean : Mean number of concave portions of the contour of the tumor cells ")
    st.write("9) x.symmetry_mean : Mean symmetry of the tumor cells")
    st.write("10) x.fractal_dimension_mean : Mean 'coastline approximation' of the tumor cells ")
    st.write("11) x.radius_se : Standard error of the radius of the tumor cells ")
    st.write("12) x.texture_se : Standard error of the texture of the tumor cells ")
    st.write("13) x.perimeter_se : Standard error of the perimeter of the tumor cells ")
    st.write("14) x.area_se : Standard error of the area of the tumor cells ")
    st.write("15) x.smoothness_se : Standard error of the smoothness of the tumor cells ")
    st.write("16) x.compactness_se : Standard error of the compactness of the tumor cells ")
    st.write("17) x.concavity_se : Standard error of the concavity of the tumor cells ")
    st.write("18) x.concave_points_se : Standard error of the number of concave portions of the contour of the tumor cells ")
    st.write("19) x.symmetry_se	: Standard error of the symmetry of the tumor cells ")
    st.write("20) x.fractal_dimension_se : Standard error of the 'coastline approximation' of the tumor cells ")
    st.write("21) x.radius_worst : Worst (largest) radius of the tumor cells ")
    st.write("22) x.texture_worst : Worst (most severe) texture of the tumor cells ")
    st.write("23) x.perimeter_worst : Worst (largest) perimeter of the tumor cells ")
    st.write("24) x.area_worst : Worst (largest) area of the tumor cells ")
    st.write("25) x.smoothness_worst : Worst (most severe) smoothness of the tumor cells ")
    st.write("26) x.compactness_worst : Worst (most severe) compactness of the tumor cells ")
    st.write("27) x.concavity_worst : Worst (most severe) concavity of the tumor cells ")
    st.write("28) x.concave_points_worst : Worst (most severe) number of concave portions of the contour of the tumor cells ")
    st.write("29) x.symmetry_worst : Worst (most severe) symmetry of the tumor cells ")
    st.write("30) x.fractal_dimension_worst : Worst (most severe) 'coastline approximation' of the tumor cells ")
    st.write("31) y : Target ")
    st.write("""B = Mass is benign/kanker jinak""")
    st.write("""M = Mass is malignant/kanker ganas""")

    st.write("""# Dataset """)
    df = pd.read_csv('https://raw.githubusercontent.com/yanuarprtama/Pendata-UAS/main/brca.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
   
    # Mendefinisikan Varible X dan Y
    X = df.drop(columns=['y','no'])
    y = df['y'].values
    df
    X
    df_min = X.min()
    df_max = X.max()

    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('y Label')
    dumies = pd.get_dummies(df.y).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1': [dumies[0]],
        '2': [dumies[1]]
    })

    st.write(labels)

with pca_menu:
    st.subheader("PCA")
    pca_components = st.selectbox(
        "Number of PCA components", [2, 3, 4, 5])

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)

    pca = PCA(n_components=pca_components)
    pca_features = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(pca_features, columns=[
                          f"PC{i+1}" for i in range(pca_components)])
    pca_df["y"] = df["y"]

    st.write(pca_df.head())

    X_pca = pca_df.drop("y", axis=1)
    y_pca = pca_df["y"]

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, y_pca, test_size=0.2, random_state=42)

    # Sklearn PCA Naive Bayes
    st.subheader("Sklearn PCA Naive Bayes")
    nb_pca = GaussianNB()
    nb_pca.fit(X_train_pca, y_train_pca)
    y_pred_nb_pca = nb_pca.predict(X_test_pca)
    accuracy_nb_pca = accuracy_score(y_test_pca, y_pred_nb_pca)
    st.write("Accuracy:",100 * accuracy_nb_pca)

    # Sklearn PCA KNN
    st.subheader("Sklearn PCA KNN")
    knn_pca = KNeighborsClassifier()
    knn_pca.fit(X_train_pca, y_train_pca)
    y_pred_knn_pca = knn_pca.predict(X_test_pca)
    accuracy_knn_pca = accuracy_score(y_test_pca, y_pred_knn_pca)
    st.write("Accuracy:",100 * accuracy_knn_pca)

    # Sklearn PCA Decision Tree
    st.subheader("Sklearn PCA Decision Tree")
    dt_pca = DecisionTreeClassifier()
    dt_pca.fit(X_train_pca, y_train_pca)
    y_pred_dt_pca = dt_pca.predict(X_test_pca)
    accuracy_dt_pca = accuracy_score(y_test_pca, y_pred_dt_pca)
    st.write("Accuracy:",100 * accuracy_dt_pca)

    # Sklearn PCA ANNBP
    st.subheader("Sklearn PCA ANNBP")
    annbp_pca = MLPClassifier()
    annbp_pca.fit(X_train_pca, y_train_pca)
    y_pred_annbp_pca = annbp_pca.predict(X_test_pca)
    accuracy_annbp_pca = accuracy_score(y_test_pca, y_pred_annbp_pca)
    st.write("Accuracy:",100 * accuracy_annbp_pca)

    # Save PCA models to pickle files
    pickle.dump(nb_pca, open("nb_pca_model.pkl", "wb"))
    pickle.dump(knn_pca, open("knn_pca_model.pkl", "wb"))
    pickle.dump(dt_pca, open("dt_pca_model.pkl", "wb"))
    pickle.dump(annbp_pca, open("annbp_pca_model.pkl", "wb"))


with modeling:
    # Nilai X training dan Nilai X testing
    training, test = train_test_split(
        scaled_features, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(
        y, test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        mlp_model = st.checkbox('ANNBackpropagation')

        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Naive Bayes Classification 
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)

        y_compare = np.vstack((test_label, y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        

        # KNN
        K = 10
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        # Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))

        # ANNBP
        # Menggunakan 2 layer tersembunyi dengan 100 neuron masing-masing
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(training, training_label)
        mlp_predict = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_predict))

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
                pickle.dump(gaussian, open("naive_bayes_model.pkl", "wb"))
            if k_nn:
                st.write(
                    "Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
                pickle.dump(knn, open("knn_model.pkl", "wb"))
            if destree:
                st.write(
                    "Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
                pickle.dump(dt, open("decision_tree_model.pkl", "wb"))
            if mlp_model:
                st.write(
                    'Model ANN (MLP) accuracy score: {0:0.2f}'.format(mlp_accuracy))
                pickle.dump(mlp, open("ann_model.pkl", "wb"))

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi': [gaussian_akurasi, knn_akurasi, dt_akurasi, mlp_accuracy],
                'Model': ['Naive Bayes', 'K-NN', 'Decission Tree', 'ANNBP'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)


with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        x_radius_mean = st.number_input('Mean radius of the tumor cells : ')
        x_texture_mean = st.number_input('Mean texture of the tumor cells : ')
        x_perimeter_mean = st.number_input('Mean perimeter of the tumor cells : ')
        x_area_mean = st.number_input('Mean area of the tumor cells : ')
        x_smoothness_mean = st.number_input('Mean smoothness of the tumor cells : ')
        x_compactness_mean = st.number_input('Mean compactness of the tumor cells : ')
        x_concavity_mean = st.number_input('Mean concavity of the tumor cells : ')
        x_concave_pts_mean = st.number_input('Mean number of concave portions of the contour of the tumor cells : ')
        x_symmetry_mean = st.number_input('Mean symmetry of the tumor cells : ')
        x_fractal_dim_mean = st.number_input('Mean coastline approximation of the tumor cells : ')
        x_radius_se = st.number_input('Standard error of the radius of the tumor cells : ')
        x_texture_se = st.number_input('Standard error of the texture of the tumor cells : ')
        x_perimeter_se = st.number_input('Standard error of the perimeter of the tumor cells : ')
        x_area_se = st.number_input('Standard error of the area of the tumor cells : ')
        x_smoothness_se = st.number_input('Standard error of the smoothness of the tumor cells : ')
        x_compactness_se = st.number_input('Standard error of the compactness of the tumor cells : ')
        x_concavity_se = st.number_input('Standard error of the concavity of the tumor cells : ')
        x_concave_pts_se = st.number_input('Standard error of the number of concave portions of the contour of the tumor cells : ')
        x_symmetry_se = st.number_input('Standard error of the symmetry of the tumor cells : ')
        x_fractal_dim_se = st.number_input('Standard error of the coastline approximation of the tumor cells : ')
        x_radius_worst = st.number_input(' Worst (largest) radius of the tumor cells : ')
        x_texture_worst = st.number_input('Worst (most severe) texture of the tumor cells : ')
        x_perimeter_worst = st.number_input('Worst (largest) perimeter of the tumor cells : ')
        x_area_worst = st.number_input('Worst (largest) area of the tumor cells : ')
        x_smoothness_worst = st.number_input('Worst (most severe) smoothness of the tumor cells : ')
        x_compactness_worst = st.number_input('Worst (most severe) compactness of the tumor cells : ')
        x_concavity_worst = st.number_input('Worst (most severe) concavity of the tumor cells : ')
        x_concave_pts_worst = st.number_input('Worst (most severe) number of concave portions of the contour of the tumor cells : ')
        x_symmetry_worst = st.number_input('Worst (most severe) symmetry of the tumor cells : ')
        x_fractal_dim_worst = st.number_input('Worst (most severe) coastline approximation of the tumor cells : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi dibawah ini:',
                             ('Naive Bayes', 'K-NN', 'Decision Tree', 'ANNBackpropaganation'))

        apply_pca = st.checkbox("Include PCA")

        prediksi = st.form_submit_button("Submit")

        if prediksi:
            inputs = np.array([
                x_radius_mean,
                x_texture_mean,
                x_perimeter_mean,
                x_area_mean,
                x_smoothness_mean,
                x_compactness_mean,
                x_concavity_mean,
                x_concave_pts_mean,
                x_symmetry_mean,
                x_fractal_dim_mean,
                x_radius_se,
                x_texture_se,
                x_perimeter_se,
                x_area_se,
                x_smoothness_se,
                x_compactness_se,
                x_concavity_se,
                x_concave_pts_se,
                x_symmetry_se,
                x_fractal_dim_se,
                x_radius_worst,
                x_texture_worst,
                x_perimeter_worst,
                x_area_worst,
                x_smoothness_worst,
                x_compactness_worst,
                x_concavity_worst,
                x_concave_pts_worst,
                x_symmetry_worst,
                x_fractal_dim_worst
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if apply_pca and X.shape[1] > 1 and X.shape[0] > 1:
                pca = PCA(n_components=min(X.shape[1], X.shape[0]))
                X_pca = pca.fit_transform(X)
                input_norm = pca.transform(input_norm)

            if model == 'Naive Bayes':
                mod = pickle.load(open("naive_bayes_model.pkl", "rb"))
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'K-NN':
                mod = pickle.load(open("knn_model.pkl", "rb"))
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'Decision Tree':
                mod = pickle.load(open("decision_tree_model.pkl", "rb"))
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'ANNBackpropaganation':
                mod = pickle.load(open("ann_model.pkl", "rb"))
                if apply_pca:
                    input_norm = pca.transform(input_norm)

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
            ada = 1
            tidak_ada = 0
            if input_pred == ada:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'ditemukan bahwa Breast Cancer Ganas')
            else:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'ditemukan bahwa Breast Cancer Jinak')
                
            # Save input_pred to pickle file
            pickle.dump(input_pred, open("input_pred.pkl", "wb"))