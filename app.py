import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# --- Model Architecture Definition ---
# The model class needs to be redefined so that Streamlit
# can load the weights into the correct model object.
# Ensure this definition is IDENTICAL to the one in the training script.

# Parameters (must be the same as in training)
IMG_SIZE = 28
NUM_CLASSES = 10
LATENT_DIM = 10
EMBED_DIM = 10
device = torch.device("cpu") # For deployment, it's safer to use CPU

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

# --- Model Loading ---
@st.cache_resource # Caches the model for better performance
def load_model():
    model = CVAE().to(device)
    # Loads the saved weights into the model's state
    model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
    model.eval() # Sets the model to evaluation mode
    return model

model = load_model()

# --- Image Generation Function ---
def generate_images(digit, num_images=5):
    generated_images = []
    with torch.no_grad():
        for _ in range(num_images):
            # 1. Sample a random vector from the latent space
            z = torch.randn(1, LATENT_DIM).to(device)
            # 2. Create the label (condition) for the desired digit
            label = torch.LongTensor([digit]).to(device)
            # 3. Generate the image using the decoder
            output = model.decode(z, label)
            # 4. Convert the tensor to an image
            img_tensor = output.view(IMG_SIZE, IMG_SIZE).cpu()
            # Convert to PIL image format (grayscale)
            pil_img = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8), 'L')
            generated_images.append(pil_img)
    return generated_images

# --- Streamlit Interface (in English) ---
st.set_page_config(page_title="Handwritten Digit Generator", layout="wide")
st.title("✍️ Handwritten Digit Generator with CVAE")

st.markdown("""
This web application generates images of handwritten digits (0-9) using a **Conditional Variational Autoencoder (CVAE)** trained on the MNIST dataset.
Select a digit and click 'Generate Images' to see 5 examples created by the model.
""")

st.divider() # Adds a horizontal line to separate blocks

# --- Main Screen Controls ---

# The selectbox for choosing the digit
selected_digit = st.selectbox("Choose a digit:", options=list(range(10)))

# The generation button
if st.button("Generate Images"):
    st.subheader(f"Generated Images for Digit: {selected_digit}")
    
    # Generate 5 images
    images = generate_images(selected_digit, num_images=5)
    
    # Display the images in 5 columns
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Generated #{i+1}", width=150)
else:
    # The instruction text
    st.info("Select a digit and click the button above to start.")