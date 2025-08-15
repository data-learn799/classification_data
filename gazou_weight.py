<<<<<<< HEAD
import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
from torch.quantization import quantize_dynamic

# タイトル
st.title("画像分類アプリ")
st.write("画像をアップロードすると、自動で分類されます。")

# 画像アップロード
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

from transformers import AutoImageProcessor,AutoModelForImageClassification
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

labels = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

id2label = {k:v for k, v in enumerate(labels)}
label2id = {k:v for v, k in enumerate(labels)}

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id)

for param in model.parameters():
    param.requires_grad = False

for param in model.vit.encoder.layer[-1].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
state_dict = torch.load(r'C:\Users\t1aok\Downloads\quantized_model.pth',map_location=torch.device('cpu'))
quantized_model.load_state_dict(state_dict)
quantized_model.eval()


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
    model=quantized_model,device_map='auto'
    )

    results = classifier(image)

    # 結果の表示
    st.subheader("分類結果（上位5件）:")
    for result in results:
        label = result["label"]
        score = result["score"]
=======
import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
from torch.quantization import quantize_dynamic

# タイトル
st.title("画像分類アプリ")
st.write("画像をアップロードすると、自動で分類されます。")

# 画像アップロード
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

from transformers import AutoImageProcessor,AutoModelForImageClassification
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

labels = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

id2label = {k:v for k, v in enumerate(labels)}
label2id = {k:v for v, k in enumerate(labels)}

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id)

for param in model.parameters():
    param.requires_grad = False

for param in model.vit.encoder.layer[-1].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
state_dict = torch.load(r'C:\Users\t1aok\Downloads\quantized_model.pth',map_location=torch.device('cpu'))
quantized_model.load_state_dict(state_dict)
quantized_model.eval()


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
    model=quantized_model,device_map='auto'
    )

    results = classifier(image)

    # 結果の表示
    st.subheader("分類結果（上位5件）:")
    for result in results:
        label = result["label"]
        score = result["score"]
>>>>>>> d52df76edcc69d048a4379dacb30d8275cb2b3e9
        st.write(f"🔹 {label}: {score:.4f}")