"""
app.py
โปรแกรมจำแนกโรค COVID จากภาพ X-ray ด้วย Streamlit

หมายเหตุสำคัญ (อ่านก่อนใช้งาน):
โมเดล .pkcls ที่ให้มา (w10_model.pkcls, w10_SVM_model.pkcls, w10_tree_model_.pkcls)
ถูกฝึกด้วยโปรแกรม Orange Data Mining โดยใช้ "Image Embedding" (SqueezeNet)
แปลงภาพ X-ray แต่ละภาพให้เป็นเวกเตอร์ตัวเลข 1000 มิติ (คอลัมน์ n0-n999) ก่อนนำไปฝึกโมเดล
ดังนั้นแอปนี้จึงไม่ได้ให้ผู้ใช้กรอกค่าตัวแปรเอง แต่ให้ "อัปโหลดภาพ X-ray"
แล้วแอปจะแปลงภาพเป็น embedding แบบเดียวกับตอนฝึกโมเดลให้อัตโนมัติ ก่อนส่งเข้าโมเดลเพื่อทำนาย
"""

import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import Orange  # ต้องมี Orange3 เพื่อให้ joblib.load ตีความ (unpickle) โครงสร้างโมเดล Orange ได้
from ndf.example_models import squeezenet  # โมเดล SqueezeNet แบบออฟไลน์ (ไม่ต้องต่อเน็ต) สำหรับสกัด embedding จากภาพ

# ---------------------------------------------------------------------------
# 1) ค่ากำหนด (CONFIG)
# ---------------------------------------------------------------------------
MODEL_DIR = "models"  # << โฟลเดอร์ที่เก็บไฟล์โมเดล .pkcls (แก้ path ให้ตรงกับที่เก็บจริงบนเซิร์ฟเวอร์)

# ขนาดภาพและค่าที่ใช้ปรับสี ต้อง "เหมือนตอนฝึกโมเดลทุกประการ" (มาจาก Orange Image Embedding: SqueezeNet)
TARGET_IMAGE_SIZE = (227, 227)
MEAN_PIXEL = [104.006, 116.669, 122.679]  # ค่าเฉลี่ยสีแบบ ImageNet (BGR)

# ---------------------------------------------------------------------------
# 2) ฟังก์ชันช่วยเหลือ
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล SqueezeNet สำหรับสกัด embedding...")
def load_embedder():
    """โหลดโมเดล SqueezeNet (ทำงานแบบออฟไลน์ ไม่ต้องต่อเน็ต) ไว้ใช้ครั้งเดียวตอนเริ่มแอป"""
    return squeezenet(include_softmax=False)


@st.cache_resource(show_spinner="กำลังโหลดโมเดลจำแนกโรค...")
def load_model(model_path: str):
    """โหลดโมเดลจำแนกโรค (.pkcls) จาก Orange ด้วย joblib"""
    return joblib.load(model_path)


def image_to_embedding(image: Image.Image, embedder) -> np.ndarray:
    """
    แปลงภาพ X-ray ให้เป็นเวกเตอร์ 1000 มิติ (เหมือนขั้นตอน Image Embedding ใน Orange)
    ขั้นตอน:
      1. แปลงเป็น RGB และย่อ/ขยายภาพเป็น 227x227 พิกเซล
      2. สลับช่องสี RGB -> BGR (สไตล์ Caffe) แล้วลบค่าเฉลี่ยสี (ImageNet mean)
      3. ส่งเข้าโมเดล SqueezeNet เพื่อดึง embedding ออกมา 1000 ค่า
    """
    img = image.convert("RGB").resize(TARGET_IMAGE_SIZE, Image.LANCZOS)

    arr = np.array(img, dtype=float)[None, ...]  # shape (1, 227, 227, 3)
    bgr = arr.copy()
    bgr[:, :, :, 0] = arr[:, :, :, 2]  # R <-> B
    bgr[:, :, :, 2] = arr[:, :, :, 0]
    bgr = bgr - MEAN_PIXEL

    embedding = embedder.predict([bgr])[0][0]  # shape (1000,)
    return embedding.reshape(1, -1)  # shape (1, 1000) พร้อมส่งเข้าโมเดลจำแนกโรค


def predict_disease(model, embedding: np.ndarray):
    """ส่ง embedding เข้าโมเดล Orange เพื่อทำนายผล พร้อมความน่าจะเป็นของแต่ละคลาส"""
    pred_idx, probs = model(embedding, ret=Orange.classification.Model.ValueProbs)
    class_names = model.domain.class_var.values  # เช่น ('covid', 'normal', 'pneumonia')
    predicted_label = class_names[int(pred_idx[0])]
    prob_dict = {cls: float(p) for cls, p in zip(class_names, probs[0])}
    return predicted_label, prob_dict


# ป้ายกำกับภาษาไทยสำหรับแต่ละคลาส (แก้ไขได้ตามต้องการ)
LABEL_MAP_TH = {
    "covid": "พบลักษณะเข้าข่ายโรค COVID-19",
    "normal": "ปกติ ไม่พบความผิดปกติ",
    "pneumonia": "พบลักษณะเข้าข่ายปอดอักเสบ (Pneumonia)",
}


# ---------------------------------------------------------------------------
# 3) ส่วนหลักของแอป (Main UI)
# ---------------------------------------------------------------------------
st.title("โปรแกรมจำแนกโรค covid จากภาพ x-ray")

st.write(
    "อัปโหลดภาพเอกซเรย์ทรวงอก (Chest X-ray) แล้วกดปุ่ม **ทำนายผล** "
    "ระบบจะจำแนกว่าภาพเข้าข่าย **COVID-19**, **ปกติ (Normal)** หรือ **ปอดอักเสบ (Pneumonia)**"
)

# --- 3.1 UI ให้ผู้ใช้เลือกโมเดลเอง ---
st.sidebar.header("เลือกโมเดล")

model_path = None
if os.path.isdir(MODEL_DIR):
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkcls")]
    if model_files:
        selected_file = st.sidebar.selectbox("เลือกไฟล์โมเดล (.pkcls)", model_files)
        model_path = os.path.join(MODEL_DIR, selected_file)
    else:
        st.sidebar.warning(f"ไม่พบไฟล์ .pkcls ในโฟลเดอร์ '{MODEL_DIR}'")
else:
    st.sidebar.warning(f"ไม่พบโฟลเดอร์ '{MODEL_DIR}' กรุณาอัปโหลดไฟล์โมเดลแทน")

# ทางเลือกเสริม: อัปโหลดไฟล์โมเดลเอง (เผื่อไม่มีในโฟลเดอร์ models)
uploaded_model = st.sidebar.file_uploader("หรืออัปโหลดไฟล์โมเดล (.pkcls)", type=["pkcls"])
if uploaded_model is not None:
    temp_model_path = "temp_uploaded_model.pkcls"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path = temp_model_path

if model_path is None:
    st.info("กรุณาเลือกหรืออัปโหลดไฟล์โมเดลก่อนเริ่มใช้งาน")
    st.stop()

# --- 3.2 โหลดโมเดลจำแนกโรค + โมเดล SqueezeNet สำหรับสกัด embedding ---
model = load_model(model_path)
embedder = load_embedder()
st.sidebar.success(f"โหลดโมเดลสำเร็จ: {os.path.basename(model_path)}")

# ตรวจสอบว่าโมเดลรับ feature 1000 ตัวจริงตามที่คาดไว้ (ป้องกันกรณีเลือกโมเดลผิดชนิด)
n_features_expected = len(model.domain.attributes)
if n_features_expected != 1000:
    st.warning(
        f"โมเดลนี้คาดหวัง feature จำนวน {n_features_expected} ตัว "
        "ซึ่งไม่ตรงกับ embedding 1000 มิติที่แอปนี้สร้างให้ "
        "โปรดตรวจสอบว่าเลือกไฟล์โมเดลถูกต้อง"
    )

# --- 3.3 ให้ผู้ใช้อัปโหลดภาพ X-ray ---
st.subheader("อัปโหลดภาพ X-ray")
uploaded_image = st.file_uploader("เลือกไฟล์ภาพ (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # --- 3.4 ปุ่มทำนายผล ---
    if st.button("ทำนายผล"):
        with st.spinner("กำลังประมวลผลภาพและทำนาย..."):
            try:
                embedding = image_to_embedding(image, embedder)
                predicted_label, prob_dict = predict_disease(model, embedding)

                th_label = LABEL_MAP_TH.get(predicted_label, predicted_label)
                confidence = prob_dict[predicted_label] * 100

                st.success(f"ผลการทำนาย: **{th_label}** (ความมั่นใจ {confidence:.2f}%)")

                # แสดงความน่าจะเป็นของทุกคลาสเป็นตาราง/กราฟแท่งให้เข้าใจง่าย
                st.write("ความน่าจะเป็นของแต่ละคลาส:")
                st.bar_chart(prob_dict)

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
                st.write(
                    "สาเหตุที่พบบ่อย: ไฟล์โมเดลที่เลือกไม่ใช่โมเดลชนิดเดียวกับที่ฝึกด้วย "
                    "SqueezeNet embedding 1000 มิติ กรุณาตรวจสอบไฟล์โมเดลที่เลือก"
                )
else:
    st.info("กรุณาอัปโหลดภาพ X-ray เพื่อเริ่มการทำนาย")
