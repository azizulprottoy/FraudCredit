import os
import numpy as np
import torch
import joblib
import json
from sklearn.preprocessing import StandardScaler
from train_engine import train_vae, train_wgan, initialize_ensemble

def bootstrap():
    print("=== CreditCardFraudRnD Synthetic Bootstrap ===")
    
    # 1. Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/artifacts', exist_ok=True)
    
    # 2. Generate Synthetic Data (432 features as per VAE requirement)
    print("[1/4] Generating synthetic preprocessed data...")
    n_samples = 1000
    n_features = 432
    
    X = np.random.randn(n_samples, n_features)
    # Make some "fraud" samples distinct
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, 50, replace=False)
    y[fraud_indices] = 1
    X[fraud_indices] += 2.0 # Simple shift for fraud
    
    np.save('data/X_processed.npy', X)
    np.save('data/y_processed.npy', y)
    
    # 3. Generate Dummy Scaler
    print("[2/4] Creating dummy scaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, 'models/artifacts/scaler.pkl')
    
    # 4. Train Minimal Models
    print("[3/4] Training minimal models (VAE & Ensemble)...")
    
    # Train VAE on legitimate data
    X_legit = X[y == 0]
    vae_model = train_vae(X_legit, epochs=5, batch_size=64)
    torch.save(vae_model.state_dict(), 'models/artifacts/vae.pth')
    
    # Train WGAN on fraud data
    X_fraud = X[y == 1]
    g_model = train_wgan(X_fraud, epochs=5, batch_size=32)
    torch.save(g_model.state_dict(), 'models/artifacts/generator.pth')
    
    # Initialize Ensemble
    ensemble = initialize_ensemble(X, y)
    joblib.dump(ensemble, 'models/artifacts/river_ensemble.pkl')
    
    # 5. Generate Sandbox Profiles
    print("[4/4] Mapping sandbox profiles...")
    idx_normal = np.where(y == 0)[0][0]
    idx_fraud = np.where(y == 1)[0][0]
    
    db = {
        '4000123456789010': {'features': X[idx_normal].tolist(), 'type': 'Normal'},
        '5000987654321098': {'features': X[idx_fraud].tolist(), 'type': 'Standard Fraud'},
        '4444555566667777': {'features': X[np.where(y == 1)[0][1]].tolist(), 'type': 'VPN / IP Anomaly'},
        '4111222233334444': {'features': X[np.where(y == 1)[0][2]].tolist(), 'type': 'Synthetic Identity'}
    }
    
    with open('models/artifacts/sandbox_database.json', 'w') as f:
        json.dump(db, f, indent=4)
        
    print("\n[SUCCESS] Bootstrap complete! All artifacts generated.")
    print("You can now run 'run.bat' to start the application.")

if __name__ == '__main__':
    bootstrap()
