from __future__ import annotations

import glob
import io
import os
import random
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

try:
    import aisuite as ai
except ImportError:
    ai = None


st.set_page_config(page_title="IVE PK + Lucky Vicky Hub", layout="wide")


# ========= 模型載入 & 影像工具 =========
@st.cache_resource(show_spinner=False)
def load_face_model() -> Optional[FaceAnalysis]:
    """Load InsightFace FaceAnalysis model once per session."""
    if FaceAnalysis is None:
        return None

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def ensure_folder_structure(members: List[str], base_dirs: List[str]) -> List[str]:
    created = []
    for base in base_dirs:
        for member in members:
            path = os.path.join(base, member)
            os.makedirs(path, exist_ok=True)
            created.append(path)
    return created


def iter_images(folder: str) -> List[str]:
    return [
        path
        for path in glob.glob(os.path.join(folder, "*"))
        if path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]


def summarize_folder(folder: str, members: List[str], member_dict: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for member in members:
        path = os.path.join(folder, member)
        count = len(iter_images(path)) if os.path.isdir(path) else 0
        rows.append(
            {
                "英文名稱": member,
                "顯示名稱": member_dict.get(member, ""),
                "照片數": count,
                "路徑": os.path.abspath(path),
            }
        )
    return pd.DataFrame(rows)


def sample_member_images(
    base_dir: str,
    members: List[str],
    member_dict: Dict[str, str],
    per_member: int = 1,
    limit: int = 12,
) -> List[Dict[str, str]]:
    """Pick a few images per member so the UI can show quick previews."""
    samples: List[Dict[str, str]] = []
    for member in members:
        folder = os.path.join(base_dir, member)
        candidates = iter_images(folder)
        if not candidates:
            continue
        picks = random.sample(candidates, k=min(per_member, len(candidates)))
        for path in picks:
            samples.append(
                {
                    "path": path,
                    "member_en": member,
                    "member_zh": member_dict.get(member, member),
                }
            )
    random.shuffle(samples)
    return samples[:limit]


def extract_face_embedding(app: FaceAnalysis, image_path: str) -> Optional[np.ndarray]:
    """原始版本（保留），實際使用改成快取版 get_face_embedding。"""
    if app is None or cv2 is None:
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    faces = app.get(img)
    if not faces:
        return None

    return faces[0].embedding


@st.cache_data(show_spinner=False)
def get_face_embedding(image_path: str) -> Optional[np.ndarray]:
    """快取版的人臉特徵抽取，避免同一張圖一再計算。"""
    app = load_face_model()
    if app is None or cv2 is None:
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    faces = app.get(img)
    if not faces:
        return None

    return faces[0].embedding


# ========= 特徵資料庫 & 評估 =========
def build_face_database(
    app: FaceAnalysis,  # 參數保留但實際不用（用快取）
    photos_dir: str,
    members: List[str],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    stats = []
    database: Dict[str, np.ndarray] = {}

    for member in members:
        folder = os.path.join(photos_dir, member)
        image_files = iter_images(folder)
        embeddings = []

        for path in image_files:
            vec = get_face_embedding(path)
            if vec is not None:
                embeddings.append(vec)

        if embeddings:
            database[member] = np.mean(embeddings, axis=0)

        stats.append(
            {
                "英文名稱": member,
                "訓練照片": len(image_files),
                "成功提取": len(embeddings),
            }
        )

    return database, pd.DataFrame(stats)


def recognize_member(
    face_db: Dict[str, np.ndarray],
    features: Optional[np.ndarray],
) -> Tuple[Optional[str], float]:
    if not face_db or features is None:
        return None, 0.0

    best_match = None
    best_score = float("inf")

    for member, db_vec in face_db.items():
        denom = float(np.linalg.norm(features) * np.linalg.norm(db_vec))
        if denom == 0:
            continue
        similarity = float(np.dot(features, db_vec) / denom)
        distance = 1 - similarity
        if distance < best_score:
            best_score = distance
            best_match = member

    confidence = 1 - best_score
    return best_match, confidence


def rank_members(
    face_db: Dict[str, np.ndarray],
    features: Optional[np.ndarray],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    if not face_db or features is None:
        return []

    scores: List[Tuple[str, float]] = []
    for member, db_vec in face_db.items():
        denom = float(np.linalg.norm(features) * np.linalg.norm(db_vec))
        if denom == 0:
            continue
        similarity = float(np.dot(features, db_vec) / denom)
        scores.append((member, similarity))

    # Sort by confidence descending
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def evaluate_testset(
    app: FaceAnalysis,  # 參數保留但實際不用
    face_db: Dict[str, np.ndarray],
    test_dir: str,
    members: List[str],
    member_dict: Dict[str, str],
    threshold: float,
) -> pd.DataFrame:
    rows = []

    for member in members:
        folder = os.path.join(test_dir, member)
        for path in iter_images(folder):
            features = get_face_embedding(path)
            prediction, confidence = recognize_member(face_db, features)
            predicted_zh = member_dict.get(prediction or "", "無法辨識")
            actual_zh = member_dict.get(member, member)
            is_correct = bool(prediction == member and confidence >= threshold)

            rows.append(
                {
                    "圖片": os.path.basename(path),
                    "實際成員_en": member,
                    "實際成員": actual_zh,
                    "預測成員_en": prediction or "",
                    "AI 預測": predicted_zh if prediction and confidence >= threshold else "信心不足",
                    "信心度": round(confidence, 3),
                    "結果": "✅ 正確" if is_correct else "❌ 錯誤",
                    "來源": os.path.abspath(path),
                }
            )

    return pd.DataFrame(rows)


def summarize_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    temp = results.copy()
    temp["正確"] = temp["結果"] == "✅ 正確"
    summary = temp.groupby("實際成員").agg(
        總張數=("圖片", "count"),
        答對張數=("正確", "sum"),
        平均信心=("信心度", "mean"),
    )
    summary["準確率(%)"] = (summary["答對張數"] / summary["總張數"] * 100).round(1)
    summary["平均信心"] = summary["平均信心"].round(3)
    return summary.reset_index()


def threshold_sweep(
    results: pd.DataFrame,
    start: float,
    end: float,
    step: float,
) -> pd.DataFrame:
    """不同門檻對準確率與 coverage 的影響。"""
    if results.empty:
        return pd.DataFrame()

    rows = []
    total = len(results)
    t = start
    while t <= end + 1e-9:
        correct_mask = (
            (results["預測成員_en"] == results["實際成員_en"])
            & (results["信心度"] >= t)
        )
        conf_mask = results["信心度"] >= t

        correct = int(correct_mask.sum())
        covered = int(conf_mask.sum())
        accuracy = correct / total * 100 if total else 0.0
        coverage = covered / total * 100 if total else 0.0

        rows.append(
            {
                "門檻": round(t, 3),
                "準確率(%)": round(accuracy, 1),
                "啟用樣本比例(%)": round(coverage, 1),
            }
        )
        t += step

    return pd.DataFrame(rows)


def confusion_matrix_df(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return pd.crosstab(
        results["實際成員"],
        results["AI 預測"],
        rownames=["實際"],
        colnames=["AI 預測"],
        dropna=False,
    )


def pca_face_db(face_db: Dict[str, np.ndarray], member_dict: Dict[str, str]) -> pd.DataFrame:
    """用 PCA 把每位成員的平均特徵壓成 2D，方便視覺化。"""
    if not face_db:
        return pd.DataFrame()

    names_en = list(face_db.keys())
    X = np.stack([face_db[name] for name in names_en], axis=0)
    X = X - X.mean(axis=0, keepdims=True)  # 中心化

    # SVD 做 PCA
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    components = X @ vt[:2].T  # (n, 2)

    rows = []
    for i, name_en in enumerate(names_en):
        rows.append(
            {
                "成員英文": name_en,
                "成員": member_dict.get(name_en, name_en),
                "x": float(components[i, 0]),
                "y": float(components[i, 1]),
            }
        )
    return pd.DataFrame(rows)


def unzip_dataset(upload, target_dir: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(upload.read())
        tmp_path = tmp.name

    with zipfile.ZipFile(tmp_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(tmp_path)


# ========= PK 狀態 =========
@dataclass
class QuizState:
    pool: List[Dict[str, str]]
    current: Optional[Dict[str, str]]
    user_score: int = 0
    ai_score: int = 0
    rounds: int = 0
    last_result: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)
    wrong_pool: List[Dict[str, str]] = field(default_factory=list)
    mode: str = "all"  # all / wrong_only


def prepare_quiz_pool(base_dir: str, members: List[str], member_dict: Dict[str, str]) -> List[Dict[str, str]]:
    pool = []
    for member in members:
        folder = os.path.join(base_dir, member)
        for path in iter_images(folder):
            pool.append(
                {
                    "path": path,
                    "member_en": member,
                    "member_zh": member_dict.get(member, member),
                }
            )
    random.shuffle(pool)
    return pool


def pick_next_question(
    pool: List[Dict[str, str]],
    face_db: Dict[str, np.ndarray],
    difficulty: str,
) -> Optional[Dict[str, str]]:
    """依難度挑題：
    - Normal：完全隨機
    - Easy：盡量挑 AI 信心高的
    - Hard：盡量挑 AI 信心偏低的
    """
    if not pool:
        return None
    if not face_db or difficulty == "Normal":
        return random.choice(pool)

    easy_thr = 0.8
    hard_thr = 0.6
    max_tries = min(30, len(pool) * 2)

    chosen = None
    for _ in range(max_tries):
        cand = random.choice(pool)
        features = get_face_embedding(cand["path"])
        _, conf = recognize_member(face_db, features)

        if difficulty == "Easy" and conf >= easy_thr:
            chosen = cand
            break
        if difficulty == "Hard" and 0.0 < conf <= hard_thr:
            chosen = cand
            break

    if chosen is None:
        chosen = random.choice(pool)
    return chosen


# ========= LLM 呼叫 =========
def lucky_vicky_post(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str,
) -> str:
    if ai is None:
        raise RuntimeError("尚未安裝 aisuite，請先執行 `pip install aisuite[all]`。")

    client = ai.Client()
    response = client.chat.completions.create(
        model=f"{provider}:{model}",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def require_packages() -> bool:
    missing = []
    if FaceAnalysis is None:
        missing.append("insightface")
    if cv2 is None:
        missing.append("opencv-python")

    if missing:
        st.warning(
            f"缺少套件：{', '.join(missing)}。請先執行 "
            "`pip install insightface onnxruntime opencv-python` 後再啟動 Streamlit。"
        )
        return False
    return True


# ========= 主程式 =========
def main() -> None:
    st.title("IVE 成員 PK + Lucky Vicky Hub")
    st.caption("結合 InsightFace PK 與 AISuite Lucky Vicky 生成器的一站式 Streamlit 介面。")

    default_en = "yujin,wonyoung,gaeul,rei,liz,leeseo"
    default_zh = "兪真유진,員瑛원영,秋天가을,Rei레이,Liz리즈,李瑞이서"

    with st.sidebar:
        st.header("基本設定")
        members_en_str = st.text_input("成員英文名稱 (逗號隔開)", default_en)
        members_zh_str = st.text_input("成員顯示名稱 (逗號隔開)", default_zh)
        photos_dir = st.text_input("訓練照片資料夾", "photos")
        test_dir = st.text_input("測試照片資料夾", "test_photos")
        player_name = st.text_input("玩家名稱", "Player1")
        difficulty = st.selectbox("PK 難度", ["Easy", "Normal", "Hard"], index=1)
        threshold = st.slider("AI 信心門檻", min_value=0.3, max_value=0.95, value=0.6, step=0.05)
        max_rounds = st.number_input("PK 回合數", min_value=3, max_value=30, value=10, step=1)

        st.divider()
        st.markdown("#### Lucky Vicky API 設定")
        provider_options = {
            "mistral": {"env": "MISTRAL_API_KEY", "models": ["ministral-8b-latest", "mistral-large-latest"]},
            "groq": {
                "env": "GROQ_API_KEY",
                "models": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "gemma2-9b-it"],
            },
            "openai": {"env": "OPENAI_API_KEY", "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]},
        }
        provider = st.selectbox("模型供應商", list(provider_options.keys()))
        model = st.selectbox("模型", provider_options[provider]["models"])
        HARDCODED_KEYS = {
            # "openai": "你的 OPENAI KEY",
            # "groq": "你的 GROQ KEY",
            "mistral": "h97MT06BhbTrKUqDzEq4zjVenhpc32Bm",
        }

        api_key = HARDCODED_KEYS.get(provider, "")
        if api_key:
            os.environ[provider_options[provider]["env"]] = api_key.strip()
            st.caption(f"已載入預設金鑰：{provider} -> {api_key}")
        else:
            st.warning("目前這個供應商沒有設定金鑰，請在程式中填入 HARDCODED_KEYS 或自行設環境變數。")
        
        st.markdown("---")
        st.caption("作者：陳宏昱 7113040017 ")

    members_en = [m.strip() for m in members_en_str.split(",") if m.strip()]
    members_zh = [m.strip() for m in members_zh_str.split(",") if m.strip()]

    if len(members_en) != len(members_zh):
        st.error("英文名稱與顯示名稱數量不一致，請重新確認設定。")
        return

    member_dict = dict(zip(members_en, members_zh))

    if "face_db" not in st.session_state:
        st.session_state["face_db"] = {}
    if "db_stats" not in st.session_state:
        st.session_state["db_stats"] = pd.DataFrame()
    if "test_results" not in st.session_state:
        st.session_state["test_results"] = pd.DataFrame()
    if "quiz_state" not in st.session_state:
        st.session_state["quiz_state"] = QuizState(pool=[], current=None)
    if "lucky_history" not in st.session_state:
        st.session_state["lucky_history"] = []

    require_packages()
    model_app = load_face_model()

    tabs = st.tabs(
        [
            "1️⃣ 資料夾管理",
            "2️⃣ 人臉特徵 & 測試",
            "3️⃣ IVE PK 小遊戲",
            "4️⃣ Lucky Vicky 生成器",
        ]
    )

    # ----- Tab 1: dataset helper -----
    with tabs[0]:
        st.subheader("資料夾管理與資料準備")
        st.markdown(
            """
            1. 建立 `photos` 與 `test_photos` 的成員子資料夾。  
            2. 可直接上傳 zip (例如 Demo03v 的 photos.zip) 並解壓縮到指定資料夾。  
            3. 下方會顯示當前資料夾的照片數量統計。
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("建立資料夾結構"):
                ensure_folder_structure(members_en, [photos_dir, test_dir])
                st.success("資料夾建立完成，請開始放入照片。")

        with col2:
            upload = st.file_uploader("上傳 photos/test_photos zip", type="zip")
            if upload is not None:
                unzip_dataset(upload, ".")
                st.success("解壓縮完成，請到下方查看統計。")

        st.markdown("##### 訓練照片概況")
        train_df = summarize_folder(photos_dir, members_en, member_dict)
        st.dataframe(train_df, use_container_width=True)
        lack_train = train_df[train_df["照片數"] == 0]["顯示名稱"].tolist()
        if lack_train:
            st.warning(f"訓練資料夾中以下成員目前沒有照片：{', '.join(lack_train)}")

        st.markdown("##### 測試照片概況")
        test_df = summarize_folder(test_dir, members_en, member_dict)
        st.dataframe(test_df, use_container_width=True)
        lack_test = test_df[test_df["照片數"] == 0]["顯示名稱"].tolist()
        if lack_test:
            st.info(f"測試資料夾中以下成員目前沒有照片：{', '.join(lack_test)}")

        st.markdown("##### 照片快速檢視")
        with st.expander("隨機抽樣，檢查資料品質"):
            sample_from = st.radio("抽樣來源", ["訓練 photos", "測試 test_photos"], horizontal=True)
            per_member = st.slider("每位成員抽幾張？", 0, 3, 1, step=1)
            total_limit = st.slider("最多顯示張數", 3, 18, 9, step=3)
            if st.button("抽樣預覽", type="primary"):
                base_dir = photos_dir if sample_from == "訓練 photos" else test_dir
                samples = sample_member_images(
                    base_dir,
                    members_en,
                    member_dict,
                    per_member=per_member,
                    limit=total_limit,
                )
                if not samples:
                    st.info("目前沒有可抽樣的照片，請先放入影像。")
                else:
                    for i in range(0, len(samples), 3):
                        cols = st.columns(3)
                        for col, sample in zip(cols, samples[i : i + 3]):
                            caption = f"{sample['member_zh']} | {os.path.basename(sample['path'])}"
                            col.image(sample["path"], caption=caption, use_container_width=True)

    # ----- Tab 2: feature building + evaluation -----
    with tabs[1]:
        st.subheader("InsightFace 特徵建立與測試")
        if model_app is None:
            st.warning("尚未載入 InsightFace 模型，請確認必要套件已安裝。")
        else:
            if st.button("重新計算人臉特徵資料庫"):
                with st.spinner("建立資料庫中..."):
                    face_db, stats = build_face_database(model_app, photos_dir, members_en)
                    st.session_state["face_db"] = face_db
                    st.session_state["db_stats"] = stats
                    st.success(f"完成！成功建立 {len(face_db)} 位成員的特徵。")

            if not st.session_state["face_db"]:
                st.info("請先建立人臉特徵資料庫。")
            else:
                st.markdown("##### 特徵建立結果")
                st.dataframe(st.session_state["db_stats"], use_container_width=True)

                if st.button("測試 test_photos 準確率"):
                    with st.spinner("評估中..."):
                        results = evaluate_testset(
                            model_app,
                            st.session_state["face_db"],
                            test_dir,
                            members_en,
                            member_dict,
                            threshold,
                        )
                        st.session_state["test_results"] = results

                if not st.session_state["test_results"].empty:
                    results = st.session_state["test_results"]
                    st.markdown(f"##### 測試結果，共 {len(results)} 張")
                    correct = (results["結果"] == "✅ 正確").sum()
                    accuracy = correct / len(results) * 100 if len(results) else 0
                    st.metric("準確率", f"{accuracy:.1f}%")
                    st.dataframe(results, use_container_width=True)
                    st.download_button(
                        "下載逐張結果 (CSV)",
                        results.to_csv(index=False).encode("utf-8"),
                        file_name="test_results.csv",
                        mime="text/csv",
                    )

                    with st.expander("不同門檻的準確率分析"):
                        start_thr = st.number_input("起始門檻", 0.0, 1.0, 0.3, 0.05)
                        end_thr = st.number_input("結束門檻", 0.0, 1.0, 0.95, 0.05)
                        step_thr = st.number_input("步進", 0.01, 0.5, 0.05, 0.01)
                        if st.button("計算最佳門檻"):
                            sweep_df = threshold_sweep(results, start_thr, end_thr, step_thr)
                            if sweep_df.empty:
                                st.info("沒有資料可以分析。")
                            else:
                                st.dataframe(sweep_df, use_container_width=True)
                                st.line_chart(
                                    sweep_df.set_index("門檻")[["準確率(%)", "啟用樣本比例(%)"]]
                                )
                                best_row = sweep_df.loc[sweep_df["準確率(%)"].idxmax()]
                                st.success(
                                    f"建議門檻約為 {best_row['門檻']:.2f}，"
                                    f"準確率 {best_row['準確率(%)']:.1f}% ，"
                                    f"啟用樣本比例 {best_row['啟用樣本比例(%)']:.1f}%"
                                )

                    with st.expander("Confusion Matrix (混淆矩陣)"):
                        cm = confusion_matrix_df(results)
                        if cm.empty:
                            st.info("目前沒有可用的結果。")
                        else:
                            st.dataframe(cm, use_container_width=True)

                    summary = summarize_accuracy(results)
                    if not summary.empty:
                        st.markdown("###### 各成員表現")
                        summary = summary.sort_values(by="準確率(%)", ascending=True)
                        st.dataframe(summary, use_container_width=True)
                        st.download_button(
                            "下載成員摘要 (CSV)",
                            summary.to_csv(index=False).encode("utf-8"),
                            file_name="member_summary.csv",
                            mime="text/csv",
                        )

                with st.expander("人臉特徵 PCA 2D 可視化"):
                    pca_df = pca_face_db(st.session_state["face_db"], member_dict)
                    if pca_df.empty:
                        st.caption("目前還沒有特徵資料。")
                    else:
                        st.scatter_chart(pca_df, x="x", y="y", color="成員")
                        st.dataframe(pca_df[["成員", "x", "y"]], use_container_width=True)

                st.divider()
                st.markdown("##### 即時單張測試")
                test_image = st.file_uploader("上傳單張照片進行辨識", type=["jpg", "jpeg", "png"], key="single_test")
                if test_image is not None:
                    image_bytes = test_image.read()
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption="上傳圖片", width=300)

                    temp_path = os.path.join(tempfile.gettempdir(), f"streamlit_test_{test_image.name}")
                    with open(temp_path, "wb") as f:
                        f.write(image_bytes)

                    features = get_face_embedding(temp_path)
                    prediction, conf = recognize_member(st.session_state["face_db"], features)
                    predicted_name = member_dict.get(prediction or "", "無法辨識")

                    if prediction and conf >= threshold:
                        st.success(f"AI 預測：{predicted_name} (信心度 {conf:.2f})")
                    else:
                        st.error("無法辨識，請嘗試清晰的單人照片。")

    # ----- Tab 3: PK mini-game -----
    with tabs[2]:
        st.subheader("和 AI PK 誰比較會認 IVE 成員")
        quiz_state: QuizState = st.session_state["quiz_state"]

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("重新載入測試題庫"):
                quiz_state.pool = prepare_quiz_pool(photos_dir, members_en, member_dict)
                quiz_state.current = None
                quiz_state.user_score = 0
                quiz_state.ai_score = 0
                quiz_state.rounds = 0
                quiz_state.last_result = ""
                quiz_state.history.clear()
                quiz_state.wrong_pool.clear()
                quiz_state.mode = "all"
                st.toast(f"載入 {len(quiz_state.pool)} 題完成！（來源：{photos_dir}）")

        with col2:
            if st.button("下一題", disabled=not quiz_state.pool):
                quiz_state.current = pick_next_question(
                    quiz_state.pool,
                    st.session_state["face_db"],
                    difficulty,
                )
                st.session_state["user_choice"] = ""

        with col3:
            if st.button("重置分數"):
                quiz_state.user_score = 0
                quiz_state.ai_score = 0
                quiz_state.rounds = 0
                quiz_state.last_result = ""

        col4, col5 = st.columns(2)
        with col4:
            if st.button("只出我答錯的題目", disabled=not quiz_state.wrong_pool):
                quiz_state.pool = list(quiz_state.wrong_pool)
                quiz_state.mode = "wrong_only"
                quiz_state.current = None
                st.toast(f"已切換到錯題練習模式，共 {len(quiz_state.pool)} 題。")
        with col5:
            if st.button("改回全部題庫", disabled=False):
                quiz_state.pool = prepare_quiz_pool(photos_dir, members_en, member_dict)
                quiz_state.mode = "all"
                quiz_state.current = None
                st.toast(f"已改回全部題庫，總題數 {len(quiz_state.pool)}。")

        if not st.session_state["face_db"]:
            st.info("請先在『人臉特徵 & 測試』分頁建立人臉特徵資料庫，AI 才能參與 PK。")
        elif not quiz_state.pool:
            st.info("請先按「重新載入測試題庫」。")
        else:
            with st.expander("先複習一下各成員長相"):
                study_samples = sample_member_images(photos_dir, members_en, member_dict, per_member=1, limit=6)
                if not study_samples:
                    st.caption("資料夾裡還沒有照片可供複習。")
                else:
                    for i in range(0, len(study_samples), 3):
                        cols = st.columns(3)
                        for col, sample in zip(cols, study_samples[i : i + 3]):
                            col.image(sample["path"], caption=sample["member_zh"], use_container_width=True)

            all_names = list(member_dict.values())
            if not all_names:
                st.warning("沒有可選的成員名稱，請先在側欄設定成員名單。")
            elif quiz_state.current:
                st.image(quiz_state.current["path"], caption="請猜猜這是哪位成員？", width=320)
                with st.expander("需要提示嗎？"):
                    if st.button("AI 給我兩個可能的人選"):
                        features = get_face_embedding(quiz_state.current["path"])
                        top_candidates = rank_members(st.session_state["face_db"], features, top_k=2)
                        if not top_candidates:
                            st.info("目前無法給提示，可能影像品質不佳。")
                        else:
                            for name, conf in top_candidates:
                                st.write(f"{member_dict.get(name, name)} (信心 {conf:.2f})")

                correct_answer = quiz_state.current["member_zh"]
                if difficulty == "Easy":
                    others = [name for name in all_names if name != correct_answer]
                    random.shuffle(others)
                    options = [correct_answer] + others[:2]
                    random.shuffle(options)
                else:
                    options = all_names

                # 清理舊的 radio 選項
                if (
                    "user_choice" in st.session_state
                    and st.session_state["user_choice"] not in options
                ):
                    del st.session_state["user_choice"]

                with st.form("pk_form"):
                    user_choice = st.radio(
                        f"你的答案（難度：{difficulty}）",
                        options=options,
                        key="user_choice",
                        horizontal=True,
                    )
                    submitted = st.form_submit_button("提交答案")

                if submitted and user_choice:
                    features = get_face_embedding(quiz_state.current["path"])
                    prediction, conf = recognize_member(st.session_state["face_db"], features)
                    ai_choice = member_dict.get(prediction or "", "無法辨識")
                    correct_answer = quiz_state.current["member_zh"]

                    user_correct = user_choice == correct_answer
                    ai_correct = prediction == quiz_state.current["member_en"] and conf >= threshold

                    quiz_state.rounds += 1
                    quiz_state.user_score += int(user_correct)
                    quiz_state.ai_score += int(ai_correct)

                    quiz_state.last_result = (
                        f"第 {quiz_state.rounds} 回合：正解 {correct_answer}\n"
                        f"你回答 {user_choice} {'✅' if user_correct else '❌'}；"
                        f"AI 回答 {ai_choice} (信心 {conf:.2f}) {'✅' if ai_correct else '❌'}"
                    )

                    # 紀錄歷史
                    quiz_state.history.append(
                        {
                            "時間": datetime.now().strftime("%H:%M:%S"),
                            "玩家": player_name,
                            "圖片": os.path.basename(quiz_state.current["path"]),
                            "正解": correct_answer,
                            "玩家答案": user_choice,
                            "AI 答案": ai_choice,
                            "AI 信心": round(conf, 3),
                            "玩家是否答對": user_correct,
                            "AI 是否答對": ai_correct,
                        }
                    )

                    # 收集錯題
                    if not user_correct:
                        if all(item["path"] != quiz_state.current["path"] for item in quiz_state.wrong_pool):
                            quiz_state.wrong_pool.append(quiz_state.current)

                    if quiz_state.rounds >= max_rounds:
                        if quiz_state.user_score > quiz_state.ai_score:
                            quiz_state.last_result += "\n🎉 你戰勝 AI！"
                        elif quiz_state.user_score < quiz_state.ai_score:
                            quiz_state.last_result += "\n🤖 AI 更勝一籌，下次加油！"
                        else:
                            quiz_state.last_result += "\n🤝 平手，勢均力敵！"

            # 分數 & 進度
            progress = quiz_state.rounds / max_rounds if max_rounds else 0.0
            st.progress(min(progress, 1.0))

            st.markdown(
                f"""
                **目前比分（玩家：{player_name}）**  
                👤 玩家：{quiz_state.user_score} 分  
                🤖 AI：{quiz_state.ai_score} 分  
                已玩：{quiz_state.rounds}/{max_rounds} 回合  
                題庫模式：{"錯題練習" if quiz_state.mode == "wrong_only" else "全部題目"}
                """
            )
            if quiz_state.last_result:
                st.info(quiz_state.last_result)

            if quiz_state.history:
                with st.expander("歷史紀錄 & 玩家表現"):
                    hist_df = pd.DataFrame(quiz_state.history)
                    st.dataframe(hist_df, use_container_width=True)
                    summary = (
                        hist_df.groupby("玩家")
                        .agg(
                            作答題數=("圖片", "count"),
                            玩家答對=("玩家是否答對", "sum"),
                            AI答對=("AI 是否答對", "sum"),
                        )
                        .assign(
                            玩家正確率=lambda d: (d["玩家答對"] / d["作答題數"] * 100).round(1),
                            AI正確率=lambda d: (d["AI答對"] / d["作答題數"] * 100).round(1),
                        )
                    )
                    st.markdown("###### 玩家排行榜（依玩家正確率排序）")
                    st.dataframe(
                        summary.sort_values("玩家正確率", ascending=False),
                        use_container_width=True,
                    )

    # ----- Tab 4: Lucky Vicky generator -----
    with tabs[3]:
        st.subheader("Lucky Vicky 員瑛式思考生成器 ")

        style_templates = {
            "IG 貼文": (
                "請用台灣習慣的中文來寫這段 IG 貼文：\n"
                "請用員瑛式思考，什麼都正向思維任何使用者的事情，"
                "用第一人稱社群媒體口吻說一次，說明為什麼這是一件超幸運的事，"
                "語氣可以可愛、有點碎念，但要溫暖，"
                "最後以「完全是 Lucky Vicky 呀!」結尾，可以加入 emoji。"
            ),
            "限時動態": (
                "請用很口語、短句的方式，像 IG 限時動態那樣，"
                "把使用者的事情當成一個超好笑又超幸運的小插曲，"
                "句子可以分行，適合貼在限時動態文字，"
                "最後加上一句「完全是 Lucky Vicky 呀!」與幾個適合的 emoji。"
            ),
            "Line 給朋友": (
                "請用像在 Line 上跟好朋友聊天的口吻，"
                "一邊吐槽、一邊安慰、一邊把事情轉成好運，"
                "適合直接複製貼給朋友看，"
                "最後一句是「完全是 Lucky Vicky 呀!」並加上一兩個 emoji。"
            ),
        }

        style_choice = st.selectbox("風格範本", list(style_templates.keys()))
        system_prompt = st.text_area(
            "System Prompt",
            value=style_templates[style_choice],
            height=180,
        )
        user_prompt = st.text_area("請輸入想轉念的事件", height=120, placeholder="例如：今天咖啡灑到電腦上了...")

        if st.button("Lucky Vicky 魔法 ✨"):
            if not api_key:
                st.error("請先在左側設定 API 金鑰。")
            elif not user_prompt.strip():
                st.error("請輸入事件描述。")
            else:
                try:
                    with st.spinner("生成中..."):
                        result = lucky_vicky_post(system_prompt, user_prompt, provider, model)
                    st.success("生成完成！")
                    st.write(result)

                    st.session_state["lucky_history"].append(
                        {
                            "時間": datetime.now().strftime("%H:%M:%S"),
                            "provider": provider,
                            "model": model,
                            "事件": user_prompt.strip(),
                            "回覆": result,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"呼叫 LLM 時發生錯誤：{exc}")

        if st.session_state["lucky_history"]:
            with st.expander("查看歷史 Lucky Vicky 語錄"):
                hist_df = pd.DataFrame(st.session_state["lucky_history"])
                st.dataframe(hist_df[["時間", "provider", "model", "事件"]], use_container_width=True)
                st.download_button(
                    "下載完整語錄 (CSV)",
                    hist_df.to_csv(index=False).encode("utf-8"),
                    file_name="lucky_vicky_history.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
