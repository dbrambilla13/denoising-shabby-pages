import torch
import streamlit as st
import os
import torchvision
from DenoisingNet import DenoisingNet
import torchvision.transforms.functional as F


# model_name = 'mild-snowflake-30-epoch=157-mse_val=0.018.ckpt'
model_name = 'pretty-elevator-27-epoch=236-mse_val=0.017.ckpt'

trained_model = DenoisingNet.load_from_checkpoint(model_name).eval().cpu()

st.title("Denoising Shabby Pages")

# split='train'
# split='validate'
split='test'


input_folder = os.path.join(split, split, f'{split}_shabby')
target_folder = os.path.join(split, split, f'{split}_cleaned')

st.header('Input Selection')
img = st.selectbox("input image",options=os.listdir(input_folder))


input_imgpath = os.path.join(input_folder, img)
target_imgpath = os.path.join(target_folder, img)



x = torchvision.io.read_image(input_imgpath).to(dtype=torch.float32)/255.0


image = F.to_pil_image(x.squeeze(0), mode="L")


st.image(image)

st.header('Denoising Model Output')

y_hat = trained_model(x.unsqueeze(0))

col1, col2 = st.columns(2)
with col1:

    st.image(F.to_pil_image(y_hat.squeeze(0), mode="L"))

with col2:
    y = torchvision.io.read_image(target_imgpath).to(dtype=torch.float32)/255.0

    st.image(F.to_pil_image(y.squeeze(0), mode="L"))