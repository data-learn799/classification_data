import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision import transforms
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# 画像アップロード
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

# データの読み込み
img = Image.open(uploaded_file).convert('RGB')

# ResNet18の特徴抽出器を用意
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # 最後の全結合層を除去
model = model.to(device)
model.eval()

def extract_deep_feature(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform1 = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(transform1).squeeze().cpu().numpy()
    return feat

#色ヒストグラム
def extract_color_hist(image, bins=(8, 8, 8)):
    rgb = image.convert("RGB")
    hist = np.histogramdd(np.array(rgb).reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)])[0]
    return hist.flatten() / hist.sum()  # 正規化

def last_feature(image):
    img = image.convert("L").resize((128, 128))  # グレースケール + リサイズ
    img_np = np.array(img)
    img_np = StandardScaler().fit_transform(img_np)
    hog_features = hog(img_np, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

color_features = extract_color_hist(img)

deep_features = extract_deep_feature(img)

hog_features = last_feature(img)

features = np.concatenate([hog_features, color_features,deep_features])

#モデルのロード
model = joblib.load(r"C:\Users\t1aok\Downloads\random_forest_model.pkl")

st.write(uploaded_file) #画像の出力

# 推論
if uploaded_file is not None:
    # 推論の実行
    img_np = np.array(img)
    result = model.predict(features.reshape(1,-1))
    result = result.tolist()
    label = {4:'Strawberry',3:'Mango',2:'Grape',1:'Banana',0:'Apple'}
    st.write(label[result[0]])
