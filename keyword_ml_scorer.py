#!/usr/bin/env python3
"""
Keyword ML Scorer — Embedding + LightGBM
=========================================
Train một lần từ file keyword_intelligence (có avg_ctr),
sau đó score keyword thô mới bất kỳ.

COMMANDS:
  # Bước 1: Train model từ data GA4
  python keyword_ml_scorer.py train \
      --data keyword_intelligence_2026-04-14.csv \
      --model model.pkl

  # Bước 2: Score keyword mới
  python keyword_ml_scorer.py score \
      --input new_keywords.csv \
      --model model.pkl \
      --output scored.csv

  # Train + score luôn trong 1 lệnh
  python keyword_ml_scorer.py train \
      --data keyword_intelligence_2026-04-14.csv \
      --model model.pkl \
      --score-input new_keywords.csv \
      --score-output scored.csv

REQUIREMENTS:
  pip install sentence-transformers lightgbm scikit-learn pandas numpy --break-system-packages
"""

import argparse
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM   = 384

# Niches và intents đã biết — dùng để handle unseen labels khi scoring
KNOWN_NICHES = [
    'Food/Recipe', 'Home Decor', 'Garden/Outdoor', 'Wedding/Craft',
    'Styling', 'Other', 'Tattoo', 'Hair/Beauty', 'Kitchen', 'Lifestyle',
]
KNOWN_INTENTS = [
    'general', 'room-ideas', 'food-baking', 'outfit-style', 'wedding-event',
    'diy-craft', 'product-specific', 'tattoo', 'food-recipe', 'hair-beauty',
    'pop-culture',
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_embedding_model():
    print(f'[→] Loading embedding model: {EMBEDDING_MODEL}')
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str], st_model) -> np.ndarray:
    print(f'[→] Embedding {len(texts):,} keywords...')
    t0 = time.time()
    vecs = st_model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f'    Done in {time.time()-t0:.1f}s')
    return vecs


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove garbage rows (pure numbers, page/N, empty)."""
    df = df.copy()
    df = df[~df['keyword'].astype(str).str.match(r'^\d+$|^page', na=False)]
    df = df[df['keyword'].notna() & (df['keyword'].astype(str).str.strip() != '')]
    df['keyword'] = df['keyword'].astype(str).str.lower().str.strip()
    return df.reset_index(drop=True)


def make_meta_features(df: pd.DataFrame,
                        niche_map: dict,
                        intent_map: dict) -> np.ndarray:
    """
    3 meta features: niche_enc, intent_enc, word_count.
    Unseen labels → -1 (LightGBM handles fine).
    """
    niche_enc  = df['niche'].map(niche_map).fillna(-1).astype(int).values.reshape(-1, 1)
    intent_enc = df['intent'].map(intent_map).fillna(-1).astype(int).values.reshape(-1, 1)
    word_count = df['keyword'].str.split().str.len().values.reshape(-1, 1)
    return np.hstack([niche_enc, intent_enc, word_count])


def save_model(path: str, payload: dict):
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f'[✓] Model saved: {path}  ({size_mb:.1f} MB)')


def load_model(path: str) -> dict:
    if not Path(path).exists():
        print(f'[ERROR] Model file not found: {path}')
        sys.exit(1)
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def cmd_train(args):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import classification_report, roc_auc_score

    # ── Load data ────────────────────────────────────────────
    print(f'[→] Loading training data: {args.data}')
    df = pd.read_csv(args.data)

    required = {'keyword', 'avg_ctr', 'niche', 'intent'}
    missing = required - set(df.columns)
    if missing:
        print(f'[ERROR] Missing columns: {missing}')
        sys.exit(1)

    df = clean_df(df)
    df['is_converter'] = (df['avg_ctr'] >= 3.0).astype(int)  # ✅ chỉ CTR thực sự tốt

    print(f'    {len(df):,} keywords | '
          f'{df["is_converter"].mean()*100:.1f}% converters | '
          f'{df["niche"].nunique()} niches | '
          f'{df["intent"].nunique()} intents')

    # ── Label encoders ────────────────────────────────────────
    niches  = sorted(df['niche'].unique().tolist())
    intents = sorted(df['intent'].unique().tolist())
    niche_map  = {n: i for i, n in enumerate(niches)}
    intent_map = {n: i for i, n in enumerate(intents)}

    # ── Features ─────────────────────────────────────────────
    st_model   = load_embedding_model()
    embeddings = embed(df['keyword'].tolist(), st_model)
    meta       = make_meta_features(df, niche_map, intent_map)
    X = np.hstack([embeddings, meta])
    y = df['is_converter'].values
    print(f'    Feature matrix: {X.shape}')

    # ── Cross-validation ──────────────────────────────────────
    print('\n[→] Cross-validating (5-fold)...')
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(lgb_clf, X, y, cv=cv, scoring='roc_auc')
    f1  = cross_val_score(lgb_clf, X, y, cv=cv, scoring='f1')

    print(f'\n── CV Results ──────────────────────────────────────')
    print(f'  ROC-AUC : {auc.mean():.4f} ± {auc.std():.4f}')
    print(f'  F1      : {f1.mean():.4f} ± {f1.std():.4f}')
    print(f'  vs rule-based baseline: AUC=0.70 | Δ={auc.mean()-0.70:+.4f}')

    y_pred = cross_val_predict(lgb_clf, X, y, cv=cv)
    print(f'\n── Classification Report ───────────────────────────')
    print(classification_report(y, y_pred, target_names=['No Convert', 'Convert']))

    # ── Final fit on full data ────────────────────────────────
    print('[→] Fitting final model on full dataset...')
    lgb_clf.fit(X, y)

    # ── Save ──────────────────────────────────────────────────
    payload = {
        'model':       lgb_clf,
        'st_model':    st_model,
        'niche_map':   niche_map,
        'intent_map':  intent_map,
        'auc_cv':      float(auc.mean()),
        'f1_cv':       float(f1.mean()),
        'trained_on':  len(df),
        'niches':      niches,
        'intents':     intents,
    }
    save_model(args.model, payload)

    # ── Optionally score right away ───────────────────────────
    if args.score_input:
        args.model_loaded = payload
        cmd_score(args)


# ─────────────────────────────────────────────────────────────
# SCORE
# ─────────────────────────────────────────────────────────────

def cmd_score(args):
    # ── Load model ────────────────────────────────────────────
    if hasattr(args, 'model_loaded'):
        payload = args.model_loaded
    else:
        print(f'[→] Loading model: {args.model}')
        payload = load_model(args.model)

    lgb_clf    = payload['model']
    st_model   = payload['st_model']
    niche_map  = payload['niche_map']
    intent_map = payload['intent_map']

    print(f'    Model trained on {payload["trained_on"]:,} keywords | '
          f'CV AUC={payload["auc_cv"]:.4f}')

    # ── Load input ────────────────────────────────────────────
    input_path = getattr(args, 'score_input', None) or getattr(args, 'input', None)
    print(f'\n[→] Loading keywords: {input_path}')
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
    try:
        raw = pd.read_csv(input_path, encoding=enc)
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError(f"Cannot decode {input_path} with any known encoding")
  
    # Auto-detect keyword column
    kw_col = None
    for c in raw.columns:
        if any(k in c.lower() for k in ['keyword', 'kw', 'chu de', 'topic', 'query', 'term']):
            kw_col = c
            break
    if kw_col is None:
        kw_col = raw.columns[0]

    # Build từ raw, giữ niche/intent trước khi dedup
    raw['_kw'] = raw[kw_col].astype(str).str.lower().str.strip()
    raw['_niche']  = raw['niche']  if 'niche'  in raw.columns else pd.Series(['Unknown'] * len(raw))
    raw['_intent'] = raw['intent'] if 'intent' in raw.columns else pd.Series(['general'] * len(raw))

    # Dedup theo keyword, giữ dòng đầu tiên
    raw_dedup = raw.drop_duplicates('_kw').copy()
    raw_dedup = raw_dedup[raw_dedup['_kw'].notna() & (raw_dedup['_kw'] != '') & (raw_dedup['_kw'] != 'nan')]
    raw_dedup = raw_dedup.reset_index(drop=True)

    df_score = pd.DataFrame({
        'keyword': raw_dedup['_kw'],
        'niche':   raw_dedup['_niche'],
        'intent':  raw_dedup['_intent'],
    })

    if 'niche' not in raw.columns:
        print('[WARNING] Thiếu cột niche — score kém chính xác ~20-30%')
    if 'intent' not in raw.columns:
        print('[WARNING] Thiếu cột intent — score kém chính xác ~20-30%')

    print(f'    {len(df_score):,} unique keywords to score')
    if 'niche' not in raw.columns or 'intent' not in raw.columns:
        print('    [!] Tip: Chạy reclassify_keywords.py trước để có niche/intent chính xác hơn')
    # ── Features ─────────────────────────────────────────────
    embeddings = embed(df_score['keyword'].tolist(), st_model)
    meta       = make_meta_features(df_score, niche_map, intent_map)
    X          = np.hstack([embeddings, meta])

    # ── Predict ───────────────────────────────────────────────
    print('[→] Scoring...')
    proba     = lgb_clf.predict_proba(X)[:, 1]
    predicted = lgb_clf.predict(X)

    df_score['convert_prob'] = (proba * 100).round(1)   # 0–100
    df_score['ml_predict']   = predicted                 # 0 or 1

    # ── Tier từ probability ───────────────────────────────────
    def prob_to_tier(p):
        if p >= 70:   return 'Tier1_High'
        elif p >= 50: return 'Tier2_Medium'
        elif p >= 35: return 'Tier3_Low'
        else:         return 'Tier4_Skip'

    df_score['tier'] = df_score['convert_prob'].apply(prob_to_tier)

    # ── Stats ─────────────────────────────────────────────────
    print(f'\n── Score Distribution ──────────────────────────────')
    tier_counts = df_score['tier'].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        pct = cnt / len(df_score) * 100
        bar = '█' * int(pct / 2)
        arrow = ' ← làm content ngay' if 'Tier1' in tier else (' ← tiềm năng' if 'Tier2' in tier else '')
        print(f'  {tier:<18} {cnt:>5,}  ({pct:4.1f}%)  {bar}{arrow}')

    print(f'\n── Top 15 — Highest convert probability ────────────')
    top = df_score.nlargest(15, 'convert_prob')
    for _, row in top.iterrows():
        bar = '█' * int(row['convert_prob'] / 5)
        print(f'  {row["convert_prob"]:>5.1f}%  {bar}  {row["keyword"]}')

    print(f'\n── Bottom 10 — Lowest (likely skip) ────────────────')
    bot = df_score.nsmallest(10, 'convert_prob')
    for _, row in bot.iterrows():
        print(f'  {row["convert_prob"]:>5.1f}%  {row["keyword"]}')

    # ── Save outputs ─────────────────────────────────────────
    output_path = getattr(args, 'score_output', None) or getattr(args, 'output', None)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Full file
    # Giữ lại volume nếu có trong file input
    extra_cols = []
    for col in ['volume', 'search_volume', 'avg_monthly_searches', 'monthly_searches']:
        if col in raw_dedup.columns:
            df_score[col] = raw_dedup[col].values
            extra_cols.append(col)
            break  # chỉ lấy 1 cột volume

    # Priority score = convert_prob × log10(volume+1) nếu có volume
    if extra_cols:
        vol_col = extra_cols[0]
        df_score['priority_score'] = (
            df_score['convert_prob'] * np.log10(df_score[vol_col].fillna(1) + 1)
        ).round(1)
        out_cols = ['keyword', 'niche', 'intent', vol_col, 'convert_prob', 'priority_score', 'tier']
        sort_col = 'priority_score'
        print(f'    [✓] Tìm thấy cột volume: {vol_col} → tính priority_score')
    else:
        out_cols = ['keyword', 'niche', 'intent', 'convert_prob', 'tier']
        sort_col = 'convert_prob'
        print('    [!] Không có cột volume — sort theo convert_prob')

    df_out = df_score[out_cols].sort_values(sort_col, ascending=False)
    df_out.to_csv(out, index=False)
    print(f'\n[✓] Full output: {out}  ({len(df_out):,} keywords)')

    # Split by tier
    for tier_label in ['Tier1_High', 'Tier2_Medium', 'Tier3_Low']:
        tier_cols = ['keyword', 'convert_prob', 'niche'] + extra_cols + (['priority_score'] if extra_cols else [])
        subset = df_out[df_out['tier'] == tier_label][[c for c in tier_cols if c in df_out.columns]]
        if len(subset) == 0:
            continue
        tier_path = out.parent / f'{out.stem}_{tier_label}.csv'
        subset.to_csv(tier_path, index=False)
        arrow = ' ← làm content ngay' if 'Tier1' in tier_label else (' ← tiềm năng' if 'Tier2' in tier_label else '')
        print(f'    {tier_label}: {tier_path.name}  ({len(subset):,} kw){arrow}')

    print('\n[✓] Done.')


# ─────────────────────────────────────────────────────────────
# INFO
# ─────────────────────────────────────────────────────────────

def cmd_info(args):
    payload = load_model(args.model)
    print(f'── Model Info ──────────────────────────────────────')
    print(f'  Trained on    : {payload["trained_on"]:,} keywords')
    print(f'  CV ROC-AUC    : {payload["auc_cv"]:.4f}')
    print(f'  CV F1         : {payload["f1_cv"]:.4f}')
    print(f'  Niches known  : {", ".join(payload["niches"])}')
    print(f'  Intents known : {", ".join(payload["intents"])}')


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Keyword ML Scorer — Embedding + LightGBM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # train
    p_train = sub.add_parser('train', help='Train model from keyword_intelligence CSV')
    p_train.add_argument('--data',         required=True, help='CSV có cột: keyword, avg_ctr, niche, intent')
    p_train.add_argument('--model',        default='keyword_model.pkl', help='Output model file (default: keyword_model.pkl)')
    p_train.add_argument('--score-input',  default=None, help='Nếu muốn score ngay sau khi train')
    p_train.add_argument('--score-output', default='scored_keywords.csv')

    # score
    p_score = sub.add_parser('score', help='Score keyword CSV mới')
    p_score.add_argument('--input',  required=True, help='CSV keyword thô cần score')
    p_score.add_argument('--model',  default='keyword_model.pkl')
    p_score.add_argument('--output', default='scored_keywords.csv')

    # info
    p_info = sub.add_parser('info', help='Xem thông tin model đã train')
    p_info.add_argument('--model', default='keyword_model.pkl')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'score':
        cmd_score(args)
    elif args.command == 'info':
        cmd_info(args)


if __name__ == '__main__':
    main()
