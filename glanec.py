from io import BytesIO
import streamlit as st
import os
import base64
import pandas as pd
from PIL import Image, ImageOps
from processor import process_faces_fast, generate_embeddings, run_clustering, sync_external_faces, find_duplicates, align_leaders

def get_b64(path):
    try:
        img = Image.open(path)
        img = ImageOps.fit(img, (300, 300))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return ""

st.set_page_config(page_title="Face Manager Pro", layout="wide")

st.markdown("""
<style>
    .img-container { width: 100%; aspect-ratio: 1/1; border-radius: 15px; overflow: hidden; border: 4px solid transparent; }
    .img-container img { width: 100%; height: 100%; object-fit: cover; }
    .selected-active { border-color: #28a745 !important; }

    /* ограничение для фото в диалогвом окне */
    [data-testid="stDialog"] img {
        max-height: 80vh !important;
        width: auto !important;
        margin: auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Пути
RAW_PATH = r"C:\Users\flumi\Downloads\ML\ProjectAM\folders\photos"
FACES_PATH = r"C:\Users\flumi\Downloads\ML\ProjectAM\folders\faces"

# БДшки
if 'photos_db' not in st.session_state:
    if os.path.exists("photos_metadata.csv"):
        st.session_state.photos_db = pd.read_csv("photos_metadata.csv")
    else:
        st.session_state.photos_db = pd.DataFrame(columns=['filepath', 'face_count'])

if 'faces_db' not in st.session_state:
    if os.path.exists("faces_metadata.csv"):
        df = pd.read_csv("faces_metadata.csv")
        df['embedding'] = df['embedding'].fillna("").astype(str)
        if 'is_duplicate_of' not in df.columns:
            df['is_duplicate_of'] = None
        st.session_state.faces_db = df
    else:
        st.session_state.faces_db = pd.DataFrame(columns=['face_path', 'parent_photo', 'bbox', 'tag'])

if 'selected' not in st.session_state:
    st.session_state.selected = set()

@st.dialog("Просмотр объекта", width="large")
def open_card(path, is_face=False):
    norm_path = os.path.normpath(path)
    col_img, col_info = st.columns([2, 1])

    with col_img:
        # слева оригинал
        st.image(Image.open(path), use_container_width=True)
        st.caption(f"Оригинал: {os.path.basename(path)}")

    with col_info:
        if is_face:
            st.write("### 👤 Информация о лице")
            # получение данных о лице
            row = st.session_state.faces_db[st.session_state.faces_db['face_path'].apply(os.path.normpath) == norm_path]

            if not row.empty:
                face_data = row.iloc[0]
                c_id = int(face_data['cluster']) if 'cluster' in face_data else -1

                st.markdown(f"**Группа (Cluster):** `{c_id}`")

                # выравненная версия сбоку
                if 'aligned_path' in face_data and pd.notna(face_data['aligned_path']):
                    aligned_p = os.path.normpath(str(face_data['aligned_path']))
                    if os.path.exists(aligned_p):
                        st.write("---")
                        st.write("✨ **Выровненный эталон (Aligned)**")
                        # Показываем выровненное лицо отдельно
                        st.image(Image.open(aligned_p), width=150)
                        # -------------------------------------------

                # колво дубликатов
                if 'is_duplicate_of' in st.session_state.faces_db.columns:
                    duplicates = st.session_state.faces_db[
                        st.session_state.faces_db['is_duplicate_of'].apply(
                            lambda x: os.path.normpath(str(x)) if pd.notna(x) else ""
                        ) == norm_path
                        ]
                    num_dupes = len(duplicates)
                else:
                    duplicates = pd.DataFrame()
                    num_dupes = 0

                st.markdown(f"**Количество дубликатов:** `{num_dupes}`")

                if not duplicates.empty:
                    st.write(f"### 👥 Дубликаты ({num_dupes})")
                    d_cols = st.columns(3)
                    for d_idx, (_, d_row) in enumerate(duplicates.iterrows()):
                        with d_cols[d_idx % 3]:
                            d_path = d_row['face_path']
                            if os.path.exists(d_path):
                                st.image(ImageOps.fit(Image.open(d_path), (150, 150)), use_container_width=True)
        else:
            st.write("###Лица на фото")

            found_faces = st.session_state.faces_db[
                st.session_state.faces_db['parent_photo'].apply(os.path.normpath) == norm_path
                ]

            if not found_faces.empty:
                face_cols = st.columns(2)
                for f_idx, (f_row_idx, f_row) in enumerate(found_faces.iterrows()):
                    with face_cols[f_idx % 2]:
                        face_img_path = f_row['face_path']
                        if os.path.exists(face_img_path):
                            f_img = Image.open(face_img_path)
                            f_thumb = ImageOps.fit(f_img, (200, 200))
                            st.image(f_thumb, use_container_width=True)

                            # подпись кластера
                            c_id = f_row['cluster']
                            tag_val = f_row['tag']

                            display_text = f"Группа: {c_id}"
                            if tag_val and str(tag_val) != 'nan' and str(tag_val).strip() != "":
                                display_text = f"👤 {tag_val} (Гр. {c_id})"

                            # Выделяем цветом: шум (-1) серым, остальные — зеленым
                            if c_id == -1:
                                st.caption(f" {display_text} (Шум)")
                            else:
                                st.caption(display_text)
            else:
                st.warning("Лица еще не выделены.")

        st.write("---")
        st.caption(os.path.basename(path))


# сайдбар
st.sidebar.title("Face Manager")

# 1. Основные инструменты
if st.sidebar.button("Найти лица", use_container_width=True, type="primary"):
    prog = st.sidebar.progress(0); log = st.sidebar.empty()
    st.session_state.photos_db, st.session_state.faces_db, stats = process_faces_fast(
        RAW_PATH, FACES_PATH, st.session_state.photos_db, st.session_state.faces_db, log, prog
    )
    st.session_state.photos_db.to_csv("photos_metadata.csv", index=False)
    st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
    st.rerun()

if st.sidebar.button("Создать эмбеддинги", use_container_width=True):
    prog = st.sidebar.progress(0); log = st.sidebar.empty()
    st.session_state.faces_db, stats = generate_embeddings(st.session_state.faces_db, log, prog)
    st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
    st.rerun()

if st.sidebar.button("Найти дубликаты", use_container_width=True):
    log = st.sidebar.empty()
    st.session_state.faces_db = find_duplicates(st.session_state.faces_db, log)
    st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
    st.rerun()

if st.sidebar.button("Выровнять лица", use_container_width=True):
    prog = st.sidebar.progress(0)
    log = st.sidebar.empty()
    st.session_state.faces_db = align_leaders(st.session_state.faces_db, FACES_PATH, log, prog)
    st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
    st.rerun()

if st.sidebar.button("Группировать", use_container_width=True):
    log = st.sidebar.empty()
    st.session_state.faces_db = run_clustering(st.session_state.faces_db, log)
    st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
    st.rerun()

st.sidebar.markdown("---")
# 2. Синхронизация
if st.sidebar.button("🔄 Обновить список лиц", use_container_width=True):
    st.session_state.faces_db, added = sync_external_faces(FACES_PATH, st.session_state.faces_db)
    if added > 0:
        st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
        st.sidebar.success(f"Добавлено извне: {added}")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Фильтрация")

# 1. получение списка кластеров
if not st.session_state.faces_db.empty:
    all_clusters = sorted(st.session_state.faces_db['cluster'].unique().tolist())
    cluster_options = ["Все"] + [int(c) for c in all_clusters]
    selected_cluster = st.sidebar.selectbox("Показать группу (кластер):", cluster_options)
else:
    selected_cluster = "Все"

st.sidebar.markdown("---")
# 3. Очистка
if st.sidebar.button("Очистить группы", use_container_width=True):
    st.session_state.faces_db['cluster'] = -1
    st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
    st.rerun()

if st.sidebar.button("Сбросить все базы", use_container_width=True):
    import shutil
    if os.path.exists(FACES_PATH):
        shutil.rmtree(FACES_PATH); os.makedirs(FACES_PATH)
    st.session_state.photos_db = pd.DataFrame(columns=['filepath', 'face_count'])
    st.session_state.faces_db = pd.DataFrame(columns=['face_path', 'parent_photo', 'bbox', 'tag', 'embedding', 'cluster'])
    st.session_state.photos_db.to_csv("photos_metadata.csv", index=False)
    st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
    st.rerun()

if st.sidebar.button("Очистить эмбеддинги", use_container_width=True):
    if not st.session_state.faces_db.empty:
        st.session_state.faces_db['embedding'] = ""
        st.session_state.faces_db['cluster'] = -1

        st.session_state.faces_db.to_csv("faces_metadata.csv", index=False)
        st.sidebar.success("Эмбеддинги и кластеры очищены")
        st.rerun()



app_mode = st.sidebar.radio("Папка:", ["Исходные фото", "Лица"])
num_cols = st.sidebar.slider("Сетка", 2, 8, 5)
select_mode = st.sidebar.toggle("🔘 Режим выделения")

# галерея
active_folder = RAW_PATH if app_mode == "Исходные фото" else FACES_PATH
st.title(f"📂 {app_mode}")

files_to_show = []

if not os.path.exists(active_folder):
    st.error(f"Путь не найден: {active_folder}")
else:
    if app_mode == "Лица":
        df = st.session_state.faces_db
        # показывает только не дублиикаты
        if 'is_duplicate_of' in df.columns:
            df = df[df['is_duplicate_of'].isna()]

        if selected_cluster != "Все":
            df = df[df['cluster'] == int(selected_cluster)]

        files_to_show = [os.path.normpath(p) for p in df['face_path'].values if os.path.exists(p)]

    else:
        all_raw_files = [
            os.path.normpath(os.path.join(active_folder, f))
            for f in os.listdir(active_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if selected_cluster != "Все":
            # нахождение путей родительских фото
            valid_parents = st.session_state.faces_db[
                st.session_state.faces_db['cluster'] == int(selected_cluster)
                ]['parent_photo'].apply(os.path.normpath).unique()

            files_to_show = [f for f in all_raw_files if f in valid_parents]
        else:
            files_to_show = all_raw_files

# отрисовка галереи
if files_to_show:
    cols = st.columns(num_cols)
    for idx, full_path in enumerate(files_to_show):
        is_sel = full_path in st.session_state.selected
        filename = os.path.basename(full_path)

        with cols[idx % num_cols]:
            # 1. Фото
            b64 = get_b64(full_path)
            sel_class = "selected-active" if is_sel else ""
            st.markdown(f'''
                <div class="img-card {sel_class}">
                    <img src="data:image/jpeg;base64,{b64}">
                </div>
            ''', unsafe_allow_html=True)

            # 2. Кнопка
            btn_label = "Выбрано" if is_sel else ("Выбрать" if select_mode else "Открыть")
            if st.button(btn_label, key=f"btn_{idx}_{filename}", use_container_width=True):
                if select_mode:
                    if is_sel:
                        st.session_state.selected.remove(full_path)
                    else:
                        st.session_state.selected.add(full_path)
                    st.rerun()
                else:
                    open_card(full_path, is_face=(app_mode == "Лица"))

            # 3. Доп. инфо под кнопкой
            if app_mode == "Лица":
                face_data = st.session_state.faces_db[st.session_state.faces_db['face_path'] == full_path]

                if not face_data.empty:
                    c_id = int(face_data['cluster'].iloc[0])

                    if 'is_duplicate_of' in st.session_state.faces_db.columns:
                        # подсчёт совпадения путей
                        num_dupes = len(st.session_state.faces_db[
                                            st.session_state.faces_db['is_duplicate_of'].apply(
                                                lambda x: os.path.normpath(str(x)) if pd.notna(x) else ""
                                            ) == os.path.normpath(full_path)
                                            ])
                    else:
                        num_dupes = 0

                    st.caption(f"Группа: {c_id} | Дубликатов: {num_dupes}")
else:
    st.info("Нет объектов, соответствующих выбранному фильтру.")
