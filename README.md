# Metal-Anamoly-Detection
<p align="center">
X-ray Image (any metal region) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓ <br>
Global preprocessing (contrast + denoise) <br>
↓ <br>
CNN feature extraction (local patches) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓ <br>
Patch-wise memory of NORMAL metal <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓ <br>
Nearest-neighbor distance <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓ <br>
Anomaly heatmap <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓ <br>
Score + threshold <br>
</p>



# 1. Install Requirements
        pip install torch torchvision opencv-python numpy scikit-learn matplotlib

# 2. Image Loader & Preprocessing
        import cv2
        import numpy as np
        import os
        
        IMG_SIZE = 256
        
        def load_and_preprocess(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.equalizeHist(img)
            img = cv2.GaussianBlur(img, (3,3), 0)
            img = img.astype(np.float32) / 255.0
            return img
    
# 3. Feature Extractor (Backbone)
        import torch
        import torchvision.models as models
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = models.wide_resnet50_2(pretrained=True)
        model.to(device)
        model.eval()

## We extract features from these layers
        FEATURE_LAYERS = ["layer2", "layer3"]
        
        features = {}
        
        def hook(module, input, output):
            features[module] = output
        
        hooks = []
        for name, layer in model.named_children():
            if name in FEATURE_LAYERS:
                hooks.append(layer.register_forward_hook(hook))

# 4. Extract Patch Features
        def extract_features(img):
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
            img = img.repeat(1, 3, 1, 1).to(device)
        
            features.clear()
            with torch.no_grad():
                model(img)
        
            feats = []
            for f in features.values():
                f = torch.nn.functional.interpolate(
                    f, size=(32, 32), mode="bilinear", align_corners=False
                )
                feats.append(f)
        
            feat = torch.cat(feats, dim=1)
            feat = feat.squeeze(0).permute(1, 2, 0)
            return feat.reshape(-1, feat.shape[-1]).cpu().numpy()
# 5. Build Memory Bank (TRAIN)
        from sklearn.random_projection import SparseRandomProjection
        
        train_dir = "dataset/train"
        memory_bank = []
        
        for img_name in os.listdir(train_dir):
            path = os.path.join(train_dir, img_name)
            img = load_and_preprocess(path)
            patches = extract_features(img)
            memory_bank.append(patches)
        
        memory_bank = np.concatenate(memory_bank, axis=0)

## Dimensionality reduction (important for small data)
        rp = SparseRandomProjection(n_components=256)
        memory_bank = rp.fit_transform(memory_bank)
        
        print("Memory bank shape:", memory_bank.shape)

# 6. Nearest Neighbor Anomaly Scoring
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(memory_bank)

# 7. Inference + Heatmap
        import matplotlib.pyplot as plt
        
        def anomaly_detection(img):
            patches = extract_features(img)
            patches = rp.transform(patches)
        
            distances, _ = nn.kneighbors(patches)
            score_map = distances.reshape(32, 32)
        
            score_map = cv2.resize(score_map, (IMG_SIZE, IMG_SIZE))
            anomaly_score = np.max(score_map)
        
            return anomaly_score, score_map
    
# 8. Test & Visualize Results
        test_dir = "dataset/test"
        
        for img_name in os.listdir(test_dir):
            path = os.path.join(test_dir, img_name)
            img = load_and_preprocess(path)
        
            score, heatmap = anomaly_detection(img)
        
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                0.6, heatmap, 0.4, 0
            )
        
            print(f"{img_name} → Anomaly Score: {score:.4f}")
        
            plt.imshow(overlay)
            plt.axis("off")
            plt.show()
    
# 9. Threshold (No Labels)
## After collecting scores on TRAIN images
        threshold = np.percentile(train_scores, 99.5)
