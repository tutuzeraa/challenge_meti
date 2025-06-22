import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# --- Definição da Arquitetura do Modelo ---
# A classe do modelo precisa ser definida novamente para que o Streamlit
# possa carregar os pesos no objeto de modelo correto.
# Certifique-se de que esta definição seja IDÊNTICA à do script de treinamento.

# Parâmetros (devem ser os mesmos do treinamento)
IMG_SIZE = 28
NUM_CLASSES = 10
LATENT_DIM = 10
EMBED_DIM = 10
device = torch.device("cpu") # Para o deploy, é mais seguro usar CPU

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE + EMBED_DIM, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, LATENT_DIM)
        self.fc_log_var = nn.Linear(256, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM + EMBED_DIM, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, IMG_SIZE * IMG_SIZE)

    def encode(self, x, c):
        c_embedded = self.label_embedding(c)
        inputs = torch.cat([x.view(-1, IMG_SIZE * IMG_SIZE), c_embedded], dim=-1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c_embedded = self.label_embedding(c)
        inputs = torch.cat([z, c_embedded], dim=-1)
        h = F.relu(self.fc3(inputs))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, c)
        return reconstruction, mu, log_var

# --- Carregamento do Modelo ---
@st.cache_resource # Cacheia o modelo para melhor performance
def load_model():
    model = CVAE().to(device)
    # Carrega os pesos salvos no estado do modelo
    model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
    model.eval() # Coloca o modelo em modo de avaliação
    return model

model = load_model()

# --- Função de Geração de Imagens ---
def generate_images(digit, num_images=5):
    generated_images = []
    with torch.no_grad():
        for _ in range(num_images):
            # 1. Amostra um vetor aleatório do espaço latente
            z = torch.randn(1, LATENT_DIM).to(device)
            # 2. Cria o rótulo (condição) para o dígito desejado
            label = torch.LongTensor([digit]).to(device)
            # 3. Gera a imagem usando o decoder
            output = model.decode(z, label)
            # 4. Converte o tensor para uma imagem
            img_tensor = output.view(IMG_SIZE, IMG_SIZE).cpu()
            # Converte para formato de imagem PIL (escala de cinza)
            pil_img = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8), 'L')
            generated_images.append(pil_img)
    return generated_images

# --- Interface do Streamlit ---
st.set_page_config(page_title="Digit Generator", layout="wide")
st.title("✍️ Manuscrit digit generator with CVAE")

st.markdown("""
This app generates digits (0-9) utilizing **Conditional Variational Autoencoder (CVAE)** trained on MNIST dataset.
Select a digit and click in 'Generate' to see 5 examples.
""")

st.sidebar.header("Configurations")
selected_digit = st.sidebar.selectbox("Select a digit:", options=list(range(10)))

if st.sidebar.button("Generate images"):
    st.subheader(f"Generate images for the digit: {selected_digit}")
    
    images = generate_images(selected_digit, num_images=5)
    
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"#{i+1}", width=150)
else:
    st.info("Select a digit and click in 'Generate images' in the side bar.")