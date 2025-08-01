import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# タイトル
st.title("画像分類アプリ")
st.write("画像をアップロードすると、自動で分類されます。")

# 画像アップロード
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model = torch.load(r'C:\Users\t1aok\Downloads\result (2).pt',weights_only=False,map_location=torch.device('cpu'))
model.eval()


# 推論
if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    # 推論の実行
    from transformers import pipeline

    classifier = pipeline(
    "image-classification",
    image_processor=image_processor,
    model=model,device_map='auto'
    )

    results = classifier(image)

    # 結果の表示
    st.subheader("分類結果（上位5件）:")
    for result in results:
        label = result["label"]
        score = result["score"]
        st.write(f"🔹 {label}: {score:.4f}")