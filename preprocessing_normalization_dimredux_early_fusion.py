import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X = np.load("multimodal_dataset/X_fused.npy")
# split ranges based on dims (text_dim=768, audio_dim=768, image_dim=512)
t_dim, a_dim, i_dim = 768, 768, 512
text = X[:, :t_dim]
audio = X[:, t_dim:t_dim+a_dim]
image = X[:, t_dim+a_dim:]

#scale
sc_t = StandardScaler().fit(text)
sc_a = StandardScaler().fit(audio)
sc_i = StandardScaler().fit(image)

text_s = sc_t.transform(text)
audio_s = sc_a.transform(audio)
image_s = sc_i.transform(image)

X_norm = np.concatenate([text_s, audio_s, image_s], axis=1)

#PCA
pca_t = PCA(n_components=128).fit(text_s)
pca_a = PCA(n_components=128).fit(audio_s)
pca_i = PCA(n_components=128).fit(image_s)

text_p = pca_t.transform(text_s)
audio_p = pca_a.transform(audio_s)
image_p = pca_i.transform(image_s)

X_proj = np.concatenate([text_p, audio_p, image_p], axis=1)



# create mask columns from DataFrame
import pandas as pd
df = pd.read_parquet("multimodal_dataset/multimodal_dataset.parquet")
masks = df[['mask_text','mask_audio','mask_image']].astype(int).values

# final fused input: 128+128+128+3 = 387 dims
X_final = np.concatenate([X_proj, masks], axis=1)

# save correct fusion
np.save("X_fused.npy", X_final)

print("Saved fused embeddings to X_fused.npy")
