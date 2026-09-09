"""
app.py
โปรแกรมจำแนกภาพ Chest X-ray ด้วย Streamlit

โมเดล .pkcls ถูกฝึกด้วย Orange Data Mining
โดยใช้ Image Embedding: SqueezeNet
เพื่อแปลงภาพเป็น Feature จำนวน 1000 ค่า (n0-n999)
ก่อนนำไปทำนายด้วยโมเดลที่สร้างจาก Orange
"""

# =========================================================
# IMPORT
# =========================================================

import os
import tempfile

import streamlit as st
import numpy as np
from PIL import Image
import joblib
import Orange

# SqueezeNet ที่ใช้แบบเดียวกับ Orange Image Embedding
from ndf.example_models import squeezenet


# =========================================================
# 1. CONFIG
# =========================================================

MODEL_DIR = "models"

# Orange SqueezeNet ใช้ภาพขนาด 227x227
TARGET_IMAGE_SIZE = (227, 227)

# ImageNet Mean แบบ BGR
MEAN_PIXEL = np.array(
    [104.006, 116.669, 122.679],
    dtype=np.float32
)


# =========================================================
# 2. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Chest X-ray Classification",
    page_icon="🩻",
    layout="centered"
)


# =========================================================
# 3. โหลด SqueezeNet
# =========================================================

@st.cache_resource(
    show_spinner="กำลังโหลด SqueezeNet..."
)
def load_embedder():

    model = squeezenet(
        include_softmax=False
    )

    return model


# =========================================================
# 4. โหลดโมเดล Orange
# =========================================================

@st.cache_resource(
    show_spinner="กำลังโหลดโมเดลจำแนก..."
)
def load_model(model_path):

    model = joblib.load(model_path)

    return model


# =========================================================
# 5. แปลงภาพเป็น Embedding
# =========================================================

def image_to_embedding(image, embedder):

    """
    แปลงภาพเป็น Feature 1000 ค่า
    สำหรับส่งเข้าโมเดล Orange
    """

    # แปลงเป็น RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(
        TARGET_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    # Image -> numpy
    arr = np.asarray(
        image,
        dtype=np.float32
    )

    # เพิ่ม Batch Dimension
    # (227,227,3)
    # ->
    # (1,227,227,3)

    arr = arr[None, ...]

    # RGB -> BGR
    bgr = arr[..., ::-1].copy()

    # ลบค่าเฉลี่ย ImageNet
    bgr -= MEAN_PIXEL

    # ส่งเข้า SqueezeNet
    result = embedder.predict(
        [bgr]
    )

    embedding = np.asarray(
        result[0][0],
        dtype=np.float64
    )

    # ตรวจสอบขนาด
    if embedding.size != 1000:

        raise ValueError(
            f"SqueezeNet สร้าง Feature ได้ "
            f"{embedding.size} ค่า "
            "แต่ระบบต้องการ 1000 ค่า"
        )

    # (1000,)
    # ->
    # (1,1000)

    embedding = embedding.reshape(
        1,
        1000
    )

    return embedding


# =========================================================
# 6. ฟังก์ชันทำนาย
# =========================================================

def predict_disease(
    model,
    embedding
):

    """
    ส่ง Embedding เข้าโมเดล Orange
    """

    # จำนวน Feature ที่โมเดลต้องการ
    expected_features = len(
        model.domain.attributes
    )

    # ตรวจสอบ Feature
    if expected_features != embedding.shape[1]:

        raise ValueError(
            f"โมเดลต้องการ "
            f"{expected_features} Features "
            f"แต่ได้รับ "
            f"{embedding.shape[1]} Features"
        )

    # ทำนาย
    prediction, probabilities = model(
        embedding,
        ret=Orange.classification.Model.ValueProbs
    )

    # ชื่อ Class
    class_names = list(
        model.domain.class_var.values
    )

    # index ของผลลัพธ์
    prediction_array = np.asarray(
        prediction
    ).reshape(-1)

    predicted_index = int(
        prediction_array[0]
    )

    # ชื่อ class
    predicted_label = class_names[
        predicted_index
    ]

    # probabilities
    probabilities = np.asarray(
        probabilities
    )[0]

    probability_dict = {}

    for class_name, probability in zip(
        class_names,
        probabilities
    ):

        probability_dict[
            class_name
        ] = float(probability)

    return (
        predicted_label,
        probability_dict
    )


# =========================================================
# 7. ภาษาไทย
# =========================================================

LABEL_MAP_TH = {

    "covid":
        "พบรูปแบบที่โมเดลจัดอยู่ในกลุ่ม COVID-19",

    "normal":
        "โมเดลจัดภาพอยู่ในกลุ่มปกติ (Normal)",

    "pneumonia":
        "พบรูปแบบที่โมเดลจัดอยู่ในกลุ่ม Pneumonia"
}


def get_thai_label(label):

    label_text = str(label)

    return LABEL_MAP_TH.get(
        label_text.lower(),
        label_text
    )


# =========================================================
# 8. HEADER
# =========================================================

st.title(
    "🩻 โปรแกรมจำแนกภาพ Chest X-ray"
)

st.write(
    """
อัปโหลดภาพเอกซเรย์ทรวงอก (Chest X-ray)
จากนั้นระบบจะใช้ **SqueezeNet**
แปลงภาพเป็น Feature จำนวน **1000 ค่า**

แล้วส่งข้อมูลเข้าโมเดล Machine Learning
ที่สร้างด้วย **Orange Data Mining**
เพื่อจำแนกภาพ
"""
)

st.warning(
    """
ระบบนี้จัดทำเพื่อการศึกษาและสาธิต
Machine Learning เท่านั้น

ผลจากโมเดลไม่ควรใช้แทนการวินิจฉัย
จากแพทย์หรือบุคลากรทางการแพทย์
"""
)


# =========================================================
# 9. SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ เลือกโมเดล"
)

model_path = None


# =========================================================
# 10. ตรวจสอบโฟลเดอร์ models
# =========================================================

if os.path.isdir(MODEL_DIR):

    model_files = []

    for file in os.listdir(
        MODEL_DIR
    ):

        if file.lower().endswith(
            ".pkcls"
        ):

            model_files.append(file)

    model_files.sort()


    if len(model_files) > 0:

        selected_file = (
            st.sidebar.selectbox(
                "เลือกไฟล์โมเดล (.pkcls)",
                model_files
            )
        )

        model_path = os.path.join(
            MODEL_DIR,
            selected_file
        )

    else:

        st.sidebar.warning(
            f"ไม่พบไฟล์ .pkcls "
            f"ในโฟลเดอร์ '{MODEL_DIR}'"
        )

else:

    st.sidebar.warning(
        f"ไม่พบโฟลเดอร์ "
        f"'{MODEL_DIR}'"
    )


# =========================================================
# 11. อัปโหลดโมเดลเอง
# =========================================================

uploaded_model = (
    st.sidebar.file_uploader(
        "หรืออัปโหลดไฟล์โมเดล (.pkcls)",
        type=["pkcls"]
    )
)


if uploaded_model is not None:

    temp_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pkcls"
        )
    )

    temp_file.write(
        uploaded_model.getbuffer()
    )

    temp_file.close()

    model_path = temp_file.name


# =========================================================
# 12. ถ้าไม่มีโมเดล
# =========================================================

if model_path is None:

    st.info(
        "กรุณาเลือกหรืออัปโหลดโมเดลก่อนเริ่มใช้งาน"
    )

    st.stop()


# =========================================================
# 13. โหลดโมเดล
# =========================================================

try:

    model = load_model(
        model_path
    )

except Exception as e:

    st.error(
        "❌ ไม่สามารถโหลดโมเดลได้"
    )

    st.code(
        f"{type(e).__name__}: {e}"
    )

    st.stop()


# =========================================================
# 14. ตรวจสอบโมเดล Orange
# =========================================================

try:

    n_features_expected = len(
        model.domain.attributes
    )

except Exception as e:

    st.error(
        "❌ ไฟล์นี้อาจไม่ใช่โมเดล Orange "
        "หรือโครงสร้างโมเดลไม่ถูกต้อง"
    )

    st.code(
        f"{type(e).__name__}: {e}"
    )

    st.stop()


# =========================================================
# 15. แสดงสถานะโมเดล
# =========================================================

display_model_name = os.path.basename(
    model_path
)

st.sidebar.success(
    f"โหลดโมเดลสำเร็จ:\n\n"
    f"{display_model_name}"
)

st.sidebar.write(
    f"จำนวน Features: "
    f"**{n_features_expected}**"
)


# =========================================================
# 16. ตรวจ Feature
# =========================================================

if n_features_expected != 1000:

    st.error(
        f"""
โมเดลนี้ต้องการ Feature จำนวน
**{n_features_expected} ค่า**

แต่ระบบ SqueezeNet นี้สร้าง
**1000 ค่า**

กรุณาตรวจสอบว่าเลือกโมเดลถูกต้อง
"""
    )

    st.stop()


# =========================================================
# 17. UPLOAD IMAGE
# =========================================================

st.subheader(
    "📤 อัปโหลดภาพ X-ray"
)

uploaded_image = st.file_uploader(
    "เลือกไฟล์ภาพ (jpg, jpeg, png)",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# 18. ยังไม่อัปโหลดภาพ
# =========================================================

if uploaded_image is None:

    st.info(
        "กรุณาอัปโหลดภาพ X-ray "
        "เพื่อเริ่มการทำนาย"
    )

    st.stop()


# =========================================================
# 19. เปิดภาพ
# =========================================================

try:

    image = Image.open(
        uploaded_image
    )

except Exception as e:

    st.error(
        "❌ ไม่สามารถเปิดไฟล์ภาพได้"
    )

    st.code(
        f"{type(e).__name__}: {e}"
    )

    st.stop()


# =========================================================
# 20. แสดงภาพ
# =========================================================

st.image(
    image,
    caption="ภาพที่อัปโหลด",
    use_container_width=True
)


# =========================================================
# 21. ข้อมูลภาพ
# =========================================================

with st.expander(
    "ℹ️ ดูข้อมูลภาพ"
):

    st.write(
        "ขนาดภาพ:",
        image.size
    )

    st.write(
        "รูปแบบภาพ:",
        image.format
    )

    st.write(
        "Color Mode:",
        image.mode
    )


# =========================================================
# 22. ปุ่มทำนาย
# =========================================================

predict_button = st.button(
    "🔍 ทำนายผล",
    type="primary",
    use_container_width=True
)


# =========================================================
# 23. ทำนาย
# =========================================================

if predict_button:

    try:

        # ---------------------------------------------
        # โหลด SqueezeNet
        # ---------------------------------------------

        with st.spinner(
            "กำลังโหลด SqueezeNet..."
        ):

            embedder = load_embedder()


        # ---------------------------------------------
        # สร้าง Embedding
        # ---------------------------------------------

        with st.spinner(
            "กำลังประมวลผลภาพ..."
        ):

            embedding = (
                image_to_embedding(
                    image,
                    embedder
                )
            )


        # ---------------------------------------------
        # ทำนาย
        # ---------------------------------------------

        with st.spinner(
            "กำลังทำนายผล..."
        ):

            (
                predicted_label,
                probability_dict

            ) = predict_disease(
                model,
                embedding
            )


        # ---------------------------------------------
        # ภาษาไทย
        # ---------------------------------------------

        thai_label = get_thai_label(
            predicted_label
        )


        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        confidence = (
            probability_dict[
                predicted_label
            ]
            * 100
        )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader(
            "📊 ผลการทำนาย"
        )

        st.success(
            f"**{thai_label}**"
        )

        st.metric(
            label="ความมั่นใจของโมเดล",
            value=f"{confidence:.2f}%"
        )


        # =================================================
        # Probability
        # =================================================

        st.subheader(
            "ความน่าจะเป็นของแต่ละคลาส"
        )


        # เปลี่ยนชื่อเป็นภาษาไทย
        chart_data = {}

        for (
            class_name,
            probability

        ) in probability_dict.items():

            thai_name = get_thai_label(
                class_name
            )

            chart_data[
                thai_name
            ] = probability


        st.bar_chart(
            chart_data
        )


        # =================================================
        # แสดงเปอร์เซ็นต์
        # =================================================

        for (
            class_name,
            probability

        ) in probability_dict.items():

            thai_name = get_thai_label(
                class_name
            )

            percent = probability * 100

            st.write(
                f"**{thai_name}** "
                f": {percent:.2f}%"
            )

            st.progress(
                min(
                    max(
                        float(probability),
                        0.0
                    ),
                    1.0
                )
            )


        # =================================================
        # TECHNICAL INFO
        # =================================================

        with st.expander(
            "🛠️ ข้อมูลทางเทคนิค"
        ):

            st.write(
                "โมเดล:",
                display_model_name
            )

            st.write(
                "Embedding Shape:",
                embedding.shape
            )

            st.write(
                "Model Features:",
                n_features_expected
            )

            st.write(
                "Predicted Class:",
                predicted_label
            )

            st.write(
                "Target Image Size:",
                TARGET_IMAGE_SIZE
            )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.warning(
            """
ผลลัพธ์นี้เป็นการทำนายจากโมเดล
Machine Learning เพื่อการศึกษาเท่านั้น

ไม่ควรนำผลลัพธ์ไปใช้วินิจฉัย
หรือใช้แทนคำแนะนำจากแพทย์
"""
        )


    except Exception as e:

        st.error(
            "❌ เกิดข้อผิดพลาดระหว่างการทำนาย"
        )

        st.code(
            f"{type(e).__name__}: {e}"
        )

        st.info(
            """
ตรวจสอบว่า:

1. โมเดลถูกสร้างจาก SqueezeNet Embedding
2. โมเดลมี Feature 1000 ค่า
3. ไฟล์ .pkcls ไม่เสียหาย
4. ภาพที่อัปโหลดเป็น JPG / JPEG / PNG
"""
        )
