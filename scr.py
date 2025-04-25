import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Пути к данным
styles_file = "styles.csv"
links_file = "images.csv"      # CSV с колонками filename, link
imgs_features_file = "imgs_features.pkl"
files_idx_file = "files.pkl"

# Загрузка признаков и списка файлов
with open(imgs_features_file, "rb") as f:
    imgs_features = pickle.load(f)
with open(files_idx_file, "rb") as f:
    files_all = pickle.load(f)
files = files_all[: imgs_features.shape[0]]

# Загрузка таблиц
styles = pd.read_csv(styles_file, on_bad_lines='skip')
links_df = pd.read_csv(links_file)
link_map = dict(zip(links_df["filename"], links_df["link"]))

# Сопоставление id ↔ индекс
file_ids = [int(os.path.basename(p).replace(".jpg","")) for p in files]
id_to_index = {pid: idx for idx, pid in enumerate(file_ids)}

# Кластеризация
features_norm = imgs_features / np.linalg.norm(imgs_features, axis=1, keepdims=True)
kmeans = KMeans(n_clusters=20, random_state=42)
kmeans_labels = kmeans.fit_predict(features_norm)

# Функция отображения одного товара по feature-index
def show_item(idx, title=None):
    pid = file_ids[idx]
    fname = f"{pid}.jpg"
    link = link_map.get(fname)
    if title:
        st.subheader(title)
    if link:
        st.image(link, width=200)
    else:
        st.warning(f"Ссылка для {fname} не найдена")
    info = styles[styles["id"] == pid].iloc[0]
    st.markdown(f"**ID:** {pid}")
    st.markdown(f"**Название:** {info.productDisplayName}")
    st.markdown(f"**Пол:** {info.gender}")
    st.markdown(f"**Категория:** {info.masterCategory} → {info.subCategory} → {info.articleType}")
    st.markdown(f"**Цвет:** {info.baseColour}")
    st.markdown(f"**Сезон:** {info.season} {int(info.year) if not pd.isna(info.year) else ''}")
    st.markdown(f"**Назначение:** {info.usage}")

# UI
st.title("👗 Рекомендательная система одежды")

tab1, tab2 = st.tabs(["🔍 По ID", "🎛 Фильтры"])

# Вкладка рекомендаций по ID
with tab1:
    user_id = st.number_input("Введите ID товара:", min_value=min(file_ids), max_value=max(file_ids), step=1)
    if st.button("Показать рекомендации"):
        if user_id not in id_to_index:
            st.error("ID не найден среди доступных.")
        else:
            idx = id_to_index[user_id]
            show_item(idx, title="Вы выбрали")
            feat = imgs_features[idx].reshape(1, -1)
            members = np.where(kmeans_labels == kmeans_labels[idx])[0]
            members = members[members != idx]
            sims = cosine_similarity(feat, imgs_features[members])[0]
            top3 = members[np.argsort(sims)[-3:]]
            st.subheader("Похожие товары:")
            cols = st.columns(3)
            for i,m in enumerate(top3):
                with cols[i]:
                    show_item(m)

# Вкладка фильтров с пагинацией
with tab2:
    st.header("🎛 Фильтры по атрибутам")

    # подготовка опций (как было)
    for col in ["gender","baseColour","season","usage"]:
        styles[col] = styles[col].astype(str)
    gender = st.selectbox("Пол", ["Все"] + sorted(styles["gender"].unique()))
    colour = st.selectbox("Цвет", ["Все"] + sorted(styles["baseColour"].unique()))
    season = st.selectbox("Сезон", ["Все"] + sorted(styles["season"].unique()))
    usage = st.selectbox("Назначение", ["Все"] + sorted(styles["usage"].unique()))

    # фильтрация
    filtered = styles.copy()
    if gender != "Все":   filtered = filtered[filtered["gender"] == gender]
    if colour != "Все":   filtered = filtered[filtered["baseColour"] == colour]
    if season != "Все":   filtered = filtered[filtered["season"] == season]
    if usage != "Все":    filtered = filtered[filtered["usage"] == usage]

    st.write(f"Найдено товаров: {len(filtered)}")

    # инициализируем счетчик видимых
    if "num_to_show" not in st.session_state:
        st.session_state.num_to_show = 10

    # показываем первые num_to_show
    for _, row in filtered.head(st.session_state.num_to_show).iterrows():
        pid = row["id"]
        if pid in id_to_index:
            show_item(id_to_index[pid])
            st.markdown("---")

    # кнопка «Показать ещё»
    if st.session_state.num_to_show < len(filtered):
        if st.button("Показать ещё"):
            st.session_state.num_to_show += 10

