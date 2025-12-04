import sys
import os
import pandas as pd
import numpy as np
import time

# --- Setup Path agar bisa import dari folder src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# [FIX] Tambahkan folder 'src' ke path secara eksplisit
# Ini wajib agar 'from core.base_model' di dalam decision_tree.py bisa jalan
sys.path.append(os.path.join(project_root, 'src'))

from src.decision_tree import ID3DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

def run_test():
    print("\n" + "="*60)
    print("   PROGRAM PENGUJIAN & PERBANDINGAN ID3 DECISION TREE")
    print("="*60)

    # 1. Setup File Paths
    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path  = os.path.join(project_root, 'data', 'test.csv')
    models_dir = os.path.join(project_root, 'models')
    
    # Buat folder models jika belum ada
    if not os.path.exists(models_dir): os.makedirs(models_dir)

    # 2. Load Data Train
    print(f"\n[1/4] Loading Dataset...")
    if not os.path.exists(train_path):
        print(f"[ERROR] File train.csv tidak ditemukan di {train_path}")
        return

    df_train = pd.read_csv(train_path)
    
    # --- [PENTING] MEMBERSIHKAN DATA (Preprocessing Awal) ---
    # Kita harus membuang kolom ID karena tidak relevan untuk prediksi
    # dan bisa menyebabkan kebocoran data (model menghafal ID).
    cols_to_drop = ['Target']
    if 'Student_ID' in df_train.columns: cols_to_drop.append('Student_ID')
    if 'id' in df_train.columns: cols_to_drop.append('id')
    
    # Pisahkan Fitur (X) dan Target (y)
    X = df_train.drop(cols_to_drop, axis=1) # Drop Target & ID
    y = df_train['Target'].values

    print(f"      Total Data: {len(df_train)} baris")
    print(f"      Fitur yang digunakan: {len(X.columns)} fitur")
    # (Opsional) Tampilkan nama fitur untuk memastikan ID sudah hilang
    # print(f"      List Fitur: {X.columns.tolist()}")

    # 3. Konfigurasi User
    try:
        inp = input("\n[INPUT] Masukkan Max Depth (Enter = Unlimited): ").strip()
        MAX_DEPTH = int(inp) if inp else None
    except:
        MAX_DEPTH = None
    print(f"      Using Max Depth: {MAX_DEPTH}")

    # =================================================================
    # TAHAP 1: KOMPARASI (SCRATCH vs SKLEARN) - Split 80:20
    # =================================================================
    print("\n" + "-"*60)
    print("[2/4] Melakukan Komparasi (Split Train 80% - Val 20%)")
    print("-"*60)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- A. ID3 From Scratch ---
    print("   A. Melatih ID3 From Scratch...")
    start_time = time.time()
    
    model_scratch = ID3DecisionTree(max_depth=MAX_DEPTH)
    model_scratch.fit(X_train, y_train)
    
    # Menggunakan function score() bawaan BaseClassifier (F1-Macro)
    f1_scratch = model_scratch.score(X_val, y_val)
    
    # Hitung akurasi manual untuk info tambahan
    pred_scratch = model_scratch.predict(X_val)
    acc_scratch = accuracy_score(y_val, pred_scratch)
    
    time_scratch = time.time() - start_time
    print(f"      -> Selesai dalam {time_scratch:.4f} detik")

    # --- B. Scikit-Learn (Pembanding) ---
    print("   B. Melatih Scikit-Learn DecisionTree...")
    # Menggunakan criterion='entropy' agar logic mirip dengan ID3
    model_sklearn = DecisionTreeClassifier(criterion='entropy', max_depth=MAX_DEPTH, random_state=42)
    model_sklearn.fit(X_train, y_train)
    
    pred_sklearn = model_sklearn.predict(X_val)
    acc_sklearn = accuracy_score(y_val, pred_sklearn)
    f1_sklearn = f1_score(y_val, pred_sklearn, average='macro')

    # --- C. Hasil Komparasi ---
    print("\n   📊 HASIL PERBANDINGAN (VALIDASI):")
    print(f"   {'METRIC':<15} | {'SCRATCH':<15} | {'SKLEARN':<15}")
    print("   " + "-"*50)
    print(f"   {'Accuracy':<15} | {acc_scratch*100:.2f}%{'':<9} | {acc_sklearn*100:.2f}%")
    print(f"   {'F1-Macro':<15} | {f1_scratch*100:.2f}%{'':<9} | {f1_sklearn*100:.2f}%")
    print(f"   {'Info':<15} | {'(Via .score())':<15} | {'(Via metrics)':<15}")

    if acc_scratch >= acc_sklearn:
        print("\n   ✅ HEBAT! Model buatanmu setara atau lebih baik dari library!")
    else:
        diff = (acc_sklearn - acc_scratch) * 100
        print(f"\n   ⚠️ Model Scratch tertinggal {diff:.2f}% dari Scikit-Learn.")

    # =================================================================
    # TAHAP 2: FULL TRAINING & SAVING
    # =================================================================
    print("\n" + "-"*60)
    print("[3/4] Retraining Model Scratch (100% Data) & Saving")
    print("-"*60)

    # Latih ulang dengan SEMUA data (X, y) bukan cuma X_train
    final_model = ID3DecisionTree(max_depth=MAX_DEPTH)
    final_model.fit(X, y)
    
    # Simpan Model (.pkl)
    save_path = os.path.join(models_dir, 'dtl_model.pkl')
    final_model.save_model(save_path)
    
    # Simpan Struktur Pohon (.txt)
    txt_path = os.path.join(models_dir, 'dtl_structure.txt')
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            # Print tree ke file
            final_model.print_tree(max_depth=MAX_DEPTH, file=f)
        print(f"[INFO] Struktur pohon disimpan ke: {txt_path}")
    except Exception as e:
        print(f"[ERROR] Gagal simpan TXT: {e}")

    # =================================================================
    # TAHAP 3: PREDIKSI TEST.CSV
    # =================================================================
    print("\n" + "-"*60)
    print("[4/4] Memprediksi Data Test (test.csv)")
    print("-"*60)

    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        
        # Simpan ID untuk submission nanti
        ids = None
        if 'id' in df_test.columns:
            ids = df_test['id']
        elif 'Student_ID' in df_test.columns:
            ids = df_test['Student_ID']
        else:
            ids = range(len(df_test))
            
        # Bersihkan data test sama seperti data train (Buang ID)
        X_test = df_test.copy()
        if 'Student_ID' in X_test.columns: X_test = X_test.drop('Student_ID', axis=1)
        if 'id' in X_test.columns: X_test = X_test.drop('id', axis=1)
        
        print(f"      Memprediksi {len(df_test)} data baru...")
        
        # Lakukan prediksi
        preds_test = final_model.predict(X_test.values)
        
        # Buat File Submission
        # Format Kaggle biasanya butuh kolom ID (misal: 'id' atau 'Student_ID') dan 'Target'
        # Sesuaikan nama kolom ID dengan format submission yang diminta (contoh: 'id')
        submission = pd.DataFrame({'id': ids, 'Target': preds_test})
        
        sub_path = os.path.join(models_dir, 'submission_dtl.csv')
        submission.to_csv(sub_path, index=False)
        
        print(f"   ✅ SUCCESS! File submission siap: {sub_path}")
        print("      (Cek folder 'models' untuk melihat file output)")
    else:
        print("   [WARNING] File test.csv tidak ditemukan. Skip prediksi.")

if __name__ == "__main__":
    run_test()