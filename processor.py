import os
import cv2
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
import ast
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from insightface.utils import face_align


def align_leaders(faces_db, faces_path, status_container, progress_bar):
    db = faces_db.copy()
    if 'aligned_path' not in db.columns:
        db['aligned_path'] = None

    # кого надо обработать нахождение
    mask_to_process = (db['is_duplicate_of'].isna() | (db['is_duplicate_of'] == "")) & \
                      (db['aligned_path'].isna() | (db['aligned_path'] == ""))

    to_process = db[mask_to_process]
    total = len(to_process)

    if total == 0:
        status_container.info("Все лидеры уже выровнены.")
        return db

    app = None

    current_step = 0
    for idx, row in to_process.iterrows():
        current_step += 1
        progress_bar.progress(current_step / total)

        face_p = os.path.normpath(str(row['face_path']))
        parent_p = os.path.normpath(str(row['parent_photo']))
        is_external = "External" in parent_p or not os.path.exists(parent_p)
        source_path = face_p if is_external else parent_p

        img_cv = cv2.imread(source_path)
        if img_cv is None: continue

        best_face_kps = None

        # проверка точек
        if 'kps' in row and pd.notna(row['kps']) and str(row['kps']).strip() not in ["", "[]"]:
            try:
                best_face_kps = np.array(ast.literal_eval(str(row['kps'])))
                status_container.write(f"⚡ [База] Выравнивание: `{os.path.basename(face_p)}`")
            except:
                best_face_kps = None

        if best_face_kps is None:
            if app is None:
                status_container.write("🧠 Загрузка ИИ для внешних лиц...")
                app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                app.prepare(ctx_id=0, det_size=(320, 320))

            status_container.write(f"🔍 [ИИ] Поиск точек: `{os.path.basename(face_p)}`")
            if is_external:
                img_cv = cv2.copyMakeBorder(img_cv, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            faces = app.get(img_cv)
            if faces:
                best_face_kps = max(faces, key=lambda x: x.det_score).kps
            else:
                continue

        # выравнивание
        try:
            aimg = face_align.norm_crop(img_cv, landmark=best_face_kps, image_size=112)
            face_name = os.path.basename(face_p)
            aligned_p = os.path.normpath(os.path.join(faces_path, f"aligned_{face_name}"))

            if cv2.imwrite(aligned_p, aimg):
                db.at[idx, 'aligned_path'] = aligned_p
        except Exception as e:
            status_container.write(f"❌ Ошибка: {e}")

    return db

def find_duplicates(faces_db, status_container, threshold=0.7):
    db = faces_db.copy()

    # Сбор векторов
    valid_embeddings = []
    valid_indices = []
    for idx, row in db.iterrows():
        emb = row.get('embedding', '')
        if pd.isna(emb) or str(emb) in ["", "[]"]: continue
        try:
            vector = ast.literal_eval(emb) if isinstance(emb, str) else emb
            valid_embeddings.append(np.array(vector))
            valid_indices.append(idx)
        except:
            continue

    if len(valid_embeddings) < 2:
        status_container.warning("Мало данных для поиска дублей.")
        return db

    X = np.array(valid_embeddings).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)

    status_container.write("🔍 Сравнение лиц...")
    sim_matrix = cosine_similarity(X)

    # столбцы
    db['is_duplicate_of'] = None

    visited = set()
    duplicates_count = 0

    for i in range(len(X)):
        if i in visited: continue

        leader_idx = valid_indices[i]
        visited.add(i)

        for j in range(i + 1, len(X)):
            if j not in visited and sim_matrix[i, j] >= threshold:
                dupe_idx = valid_indices[j]
                # путь одного из дубликатов
                db.at[dupe_idx, 'is_duplicate_of'] = db.at[leader_idx, 'face_path']
                visited.add(j)
                duplicates_count += 1

    status_container.success(f"Найдено и скрыто дубликатов: {duplicates_count}")
    return db

def process_faces_fast(raw_path, faces_path, photos_db, faces_db, status_container, progress_bar):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # проверка устройства
    active_providers = app.models['detection'].session.get_providers()

    if 'CUDAExecutionProvider' in active_providers:
        device_info = "**Работает на NVIDIA GPU (CUDA)**"
    else:
        device_info = "**Работает на CPU**"

    status_container.info(device_info)

    raw_files = [f for f in os.listdir(raw_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(raw_files)

    processed_paths = set(os.path.normpath(p) for p in photos_db['filepath'].values) if not photos_db.empty else set()

    # списки для накопления данных
    new_faces_list = []
    new_photos_list = []
    stats = {"processed": 0, "skipped": 0, "faces_found": 0}

    for i, filename in enumerate(raw_files):
        file_path = os.path.normpath(os.path.join(raw_path, filename))
        progress_bar.progress((i + 1) / total_files)

        if file_path in processed_paths:
            stats["skipped"] += 1
            continue

        img_cv = cv2.imread(file_path)
        if img_cv is None: continue

        # детекция
        faces = app.get(img_cv)

        for j, face in enumerate(faces):
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = np.clip(box, 0, [img_cv.shape[1], img_cv.shape[0], img_cv.shape[1], img_cv.shape[0]])

            face_img = img_cv[y1:y2, x1:x2]
            if face_img.size == 0: continue

            face_filename = f"face_{i}_{j}.jpg"
            face_p = os.path.join(faces_path, face_filename)
            cv2.imwrite(face_p, face_img)

            new_faces_list.append({
                'face_path': face_p,
                'parent_photo': file_path,
                'bbox': box.tolist(),
                'kps': face.kps.tolist(),
                'tag': '',
                'embedding': face.normed_embedding.tolist() if face.embedding is not None else '',
                'cluster': -1
            })
            stats["faces_found"] += 1

        new_photos_list.append({'filepath': file_path, 'face_count': len(faces)})
        stats["processed"] += 1

    # сборка датафрейма
    if new_faces_list:
        updated_faces = pd.concat([faces_db, pd.DataFrame(new_faces_list)], ignore_index=True)
    else:
        updated_faces = faces_db

    if new_photos_list:
        updated_photos = pd.concat([photos_db, pd.DataFrame(new_photos_list)], ignore_index=True)
    else:
        updated_photos = photos_db

    return updated_photos, updated_faces, stats


def generate_embeddings(faces_db, status_container, progress_bar):
    # инициализация модели
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    model = app.models.get('recognition')

    updated_faces = faces_db.copy()
    total = len(updated_faces)
    processed, skipped = 0, 0

    for idx, row in updated_faces.iterrows():
        progress_bar.progress((idx + 1) / total)

        # пропускаем уже готовые
        emb = row.get('embedding', '')
        if pd.notna(emb) and str(emb) != "" and str(emb) != "[]":
            skipped += 1
            continue

        # Aligned или обычный кроп
        source_path = row.get('aligned_path')
        if pd.isna(source_path) or str(source_path).strip() == "" or not os.path.exists(str(source_path)):
            source_path = row['face_path']
        else:
            source_path = str(source_path)

        img = cv2.imread(source_path)
        if img is None: continue

        status_container.write(
            f"Вектор ({'Aligned' if 'aligned' in source_path else 'Crop'}): `{os.path.basename(source_path)}`")

        # ArcFace требует 112x112
        face_img = cv2.resize(img, (112, 112))
        feat = model.get_feat(face_img).flatten()

        # L2
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm

        # сохраняем как строку для CSV
        updated_faces.at[idx, 'embedding'] = str(feat.tolist())
        processed += 1

    return updated_faces, {"processed": processed, "skipped": skipped}


def run_clustering(faces_db, status_container):
    db = faces_db.copy()
    valid_embeddings = []
    valid_indices = []

    # 1. сбор векторов для лидеров
    for idx, row in db.iterrows():
        emb = row.get('embedding', '')

        # пропуск если дубликат
        if pd.notna(row.get('is_duplicate_of')) and str(row.get('is_duplicate_of')).strip() != "":
            continue

        if pd.isna(emb) or str(emb) in ["", "[]"]: continue
        try:
            vector = ast.literal_eval(emb) if isinstance(emb, str) else emb
            valid_embeddings.append(np.array(vector))
            valid_indices.append(idx)
        except:
            continue

    if len(valid_embeddings) < 3:
        status_container.error("Нужно больше уникальных лиц (лидеров) для анализа.")
        return db

    X = np.array(valid_embeddings).astype(np.float32)
    X_normed = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    n_samples = len(X)

    status_container.write(f"Уменьшение размерности (UMAP)... {n_samples}")

    # reducer = UMAP(
    #     n_neighbors=10,
    #     n_components=5,
    #     metric='cosine',
    #     random_state=42
    # )
    # X_reduced = reducer.fit_transform(X_normed)
    #
    # status_container.write("Поиск кластеров (HDBSCAN)...")
    #
    # clusterer = HDBSCAN(
    #     min_cluster_size=10,
    #     cluster_selection_epsilon=0.1,
    #     metric='euclidean',
    # )



    # адаптивный UMAP
    # колво соседей должно расти вместе с датасетом, но не превышать разумные пределы
    if n_samples < 50:
        n_neigh = min(10, n_samples - 1)
    elif n_samples < 500:
        n_neigh = 20
    else:
        n_neigh = min(30, n_samples // 20) # Растет пропорционально объему

    reducer = UMAP(
        n_neighbors=25,
        n_components=5,
        metric='cosine',
        random_state=42
    )
    X_reduced = reducer.fit_transform(X_normed)

    status_container.write(f"Поиск кластеров (HDBSCAN)...{n_samples}")

    # адаптивный HDBSCAN
    if n_samples < 50:
        min_size = 3
        min_samp = 1
        eps_threshold = 0.1
    elif n_samples < 300:
        min_size = 8
        min_samp = 2
        eps_threshold = 0.2
    else:
        # Для больших архивов: кластером считаем группы от 3% до 5% от общего объема
        min_size = max(15, int(n_samples * 0.03))
        min_samp = max(5, int(min_size // 3))
        eps_threshold = 0.4

    clusterer = HDBSCAN(
        min_cluster_size=10,
        cluster_selection_epsilon=0.15,
        metric='euclidean',
    )

    labels = clusterer.fit_predict(X_reduced)
    db['cluster'] = -1
    # проставляем кластеры лидерам
    db.loc[valid_indices, 'cluster'] = labels.astype(int)

    # проходим по всем дубликатам и даем им кластер их лидера
    for idx, row in db.iterrows():
        leader_path = row.get('is_duplicate_of')
        if pd.notna(leader_path) and str(leader_path).strip() != "":
            leader_row = db[db['face_path'] == leader_path]
            if not leader_row.empty:
                db.at[idx, 'cluster'] = leader_row['cluster'].iloc[0]

    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    status_container.success(f"Найдено групп: {n_found}")

    return db

def sync_external_faces(faces_path, faces_db):
    valid_ext = ('.png', '.jpg', '.jpeg')
    files_on_disk = [os.path.normpath(os.path.join(faces_path, f)) for f in os.listdir(faces_path) if
                     f.lower().endswith(valid_ext)]
    files_in_db = set(faces_db['face_path'].apply(os.path.normpath).values) if not faces_db.empty else set()

    new_files = [f for f in files_on_disk if f not in files_in_db]
    if not new_files: return faces_db, 0

    new_rows = []
    for f in new_files:
        new_rows.append(
            {'face_path': f, 'parent_photo': 'External', 'bbox': 'None', 'tag': '', 'embedding': '', 'cluster': -1})

    return pd.concat([faces_db, pd.DataFrame(new_rows)], ignore_index=True), len(new_files)
