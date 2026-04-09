from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r"D:\NUS\Second sem\5015\group_report\Project")
DEST = ROOT / "data_balance_final_o"
SELECTED = DEST / "selected_results"
FIG_DIR = SELECTED / "figures"
TABLE_DIR = SELECTED / "tables"
FINAL_SRC_DIR = SELECTED / "final"
DBF_SRC_DIR = SELECTED / "data_balance_final"
W2V_SRC_DIR = SELECTED / "wav2vec"

for path in [FIG_DIR, TABLE_DIR, FINAL_SRC_DIR, DBF_SRC_DIR, W2V_SRC_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst_dir: Path, name: str | None = None):
    src = Path(src)
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (name or src.name)
    shutil.copy2(src, dst)
    return dst


def plot_metric_bars(df: pd.DataFrame, model_col: str, auc_col: str, acc_col: str, out_path: Path, title: str):
    plot_df = df[[model_col, auc_col, acc_col]].copy()
    plot_df = plot_df.sort_values(auc_col, ascending=True)
    y = np.arange(len(plot_df))
    h = 0.36
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(plot_df) + 1.5)))
    ax.barh(y - h / 2, plot_df[auc_col], height=h, label='AUROC', color='#1f77b4')
    ax.barh(y + h / 2, plot_df[acc_col], height=h, label='ACC', color='#ff7f0e')
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[model_col])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_wav2vec_summary(with_df: pd.DataFrame, without_df: pd.DataFrame, out_path: Path):
    w = with_df[['model', 'accuracy_mean', 'auc_mean']].copy()
    w['setting'] = 'with_aug'
    wo = without_df[['model', 'accuracy_mean', 'auc_mean']].copy()
    wo['setting'] = 'without_aug'
    combo = pd.concat([w, wo], ignore_index=True)
    models = sorted(combo['model'].unique())
    x = np.arange(len(models))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for idx, setting in enumerate(['without_aug', 'with_aug']):
        part = combo[combo['setting'] == setting].set_index('model').reindex(models)
        axes[0].bar(x + (idx - 0.5) * width, part['auc_mean'], width=width, label=setting)
        axes[1].bar(x + (idx - 0.5) * width, part['accuracy_mean'], width=width, label=setting)
    axes[0].set_title('Wav2Vec Frozen Features: Mean AUROC')
    axes[1].set_title('Wav2Vec Frozen Features: Mean ACC')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.25)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_overall_scatter(points: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    palette = {
        'final': '#1f77b4',
        'data_balance_final': '#d62728',
        'data_balance_final_enhanced': '#2ca02c',
        'wav2vec': '#9467bd',
    }
    for _, row in points.iterrows():
        ax.scatter(row['auc'], row['acc'], s=120, color=palette.get(row['source'], '#333333'))
        ax.annotate(row['label'], (row['auc'], row['acc']), textcoords='offset points', xytext=(6, 6), fontsize=9)
    ax.set_xlim(0.4, 0.85)
    ax.set_ylim(0.45, 0.82)
    ax.set_xlabel('AUROC')
    ax.set_ylabel('ACC')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles = []
    for key, color in palette.items():
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='', color=color, label=key, markersize=8))
    ax.legend(handles=handles, loc='lower right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def best_fold_from_roc_dir(model_dir: Path):
    best_fold = None
    best_auc = -1.0
    for csv_path in sorted(model_dir.glob('fold_*_roc_points.csv')):
        try:
            roc_df = pd.read_csv(csv_path)
            fpr = roc_df.iloc[:, 0].to_numpy(dtype=float)
            tpr = roc_df.iloc[:, 1].to_numpy(dtype=float)
            order = np.argsort(fpr)
            auc = float(np.trapezoid(tpr[order], fpr[order]))
            fold_name = csv_path.stem.split('_')[1]
            if auc > best_auc:
                best_auc = auc
                best_fold = fold_name
        except Exception:
            continue
    return best_fold, best_auc


# Source paths
final_model_summary = ROOT / 'final' / 'results' / 'nested_cv_binary_w1p0_h0p25' / 'model_summary.csv'
dbf_key = ROOT / 'data_balance_final' / 'results' / 'key_results_summary.csv'
dbf_sup = ROOT / 'data_balance_final' / 'results' / 'strict_supervised_benchmark' / 'strict_supervised_model_summary.csv'
dbf_unsup = ROOT / 'data_balance_final' / 'results' / 'strict_unsupervised_benchmark' / 'strict_unsupervised_model_summary.csv'
dbf_shap_top = ROOT / 'data_balance_final' / 'analysis' / 'figures' / 'catboost_selected_best_shap_top20.png'
dbf_shap_groups = ROOT / 'data_balance_final' / 'analysis' / 'figures' / 'catboost_selected_best_shap_groups.png'

enh_key = ROOT / 'data_balance_final' / 'enhanced' / 'results' / 'key_results_summary.csv'
enh_sup = ROOT / 'data_balance_final' / 'enhanced' / 'results' / 'strict_supervised_benchmark' / 'strict_supervised_model_summary.csv'
enh_unsup = ROOT / 'data_balance_final' / 'enhanced' / 'results' / 'strict_unsupervised_benchmark' / 'strict_unsupervised_model_summary.csv'
enh_shap_top = ROOT / 'data_balance_final' / 'enhanced' / 'analysis' / 'figures' / 'hybrid_catboost_gpu_shap_top20.png'
enh_shap_groups = ROOT / 'data_balance_final' / 'enhanced' / 'analysis' / 'figures' / 'hybrid_catboost_gpu_shap_groups.png'

w2v_with = ROOT / 'Wav2vec_o' / 'Wav2vec' / 'results_voiced_wav2vec2' / 'with_augmentation' / 'summary_metrics.csv'
w2v_without = ROOT / 'Wav2vec_o' / 'Wav2vec' / 'results_voiced_wav2vec2' / 'without_augmentation' / 'summary_metrics.csv'

# Copy key source files
copied = []
for src, dst_dir, out_name in [
    (final_model_summary, FINAL_SRC_DIR, 'final_model_summary.csv'),
    (dbf_key, DBF_SRC_DIR, 'base_key_results_summary.csv'),
    (dbf_sup, DBF_SRC_DIR, 'base_strict_supervised_model_summary.csv'),
    (dbf_unsup, DBF_SRC_DIR, 'base_strict_unsupervised_model_summary.csv'),
    (dbf_shap_top, DBF_SRC_DIR, 'base_catboost_selected_best_shap_top20.png'),
    (dbf_shap_groups, DBF_SRC_DIR, 'base_catboost_selected_best_shap_groups.png'),
    (enh_key, DBF_SRC_DIR, 'enhanced_key_results_summary.csv'),
    (enh_sup, DBF_SRC_DIR, 'enhanced_strict_supervised_model_summary.csv'),
    (enh_unsup, DBF_SRC_DIR, 'enhanced_strict_unsupervised_model_summary.csv'),
    (enh_shap_top, DBF_SRC_DIR, 'enhanced_hybrid_catboost_gpu_shap_top20.png'),
    (enh_shap_groups, DBF_SRC_DIR, 'enhanced_hybrid_catboost_gpu_shap_groups.png'),
    (w2v_with, W2V_SRC_DIR, 'with_augmentation_summary_metrics.csv'),
    (w2v_without, W2V_SRC_DIR, 'without_augmentation_summary_metrics.csv'),
]:
    out = copy_file(src, dst_dir, out_name)
    if out:
        copied.append({'source': str(src), 'copied_to': str(out)})

# Copy representative original Wav2Vec charts
w2v_candidates = [
    ('with_aug_auc_best', ROOT / 'Wav2vec_o' / 'Wav2vec' / 'results_voiced_wav2vec2' / 'with_augmentation' / 'SVM'),
    ('with_aug_acc_best', ROOT / 'Wav2vec_o' / 'Wav2vec' / 'results_voiced_wav2vec2' / 'with_augmentation' / 'RF'),
    ('without_aug_acc_best', ROOT / 'Wav2vec_o' / 'Wav2vec' / 'results_voiced_wav2vec2' / 'without_augmentation' / 'SVM'),
]
for tag, model_dir in w2v_candidates:
    fold, auc = best_fold_from_roc_dir(model_dir)
    if fold is None:
        continue
    roc_png = model_dir / f'fold_{fold}_roc.png'
    cm_png = model_dir / f'fold_{fold}_confusion_matrix.png'
    out1 = copy_file(roc_png, W2V_SRC_DIR, f'{tag}_fold_{fold}_roc.png')
    out2 = copy_file(cm_png, W2V_SRC_DIR, f'{tag}_fold_{fold}_confusion_matrix.png')
    if out1:
        copied.append({'source': str(roc_png), 'copied_to': str(out1), 'computed_auc': round(float(auc), 4)})
    if out2:
        copied.append({'source': str(cm_png), 'copied_to': str(out2), 'computed_auc': round(float(auc), 4)})

# Load tables
final_df = pd.read_csv(final_model_summary)
dbf_sup_df = pd.read_csv(dbf_sup)
dbf_unsup_df = pd.read_csv(dbf_unsup)
enh_key_df = pd.read_csv(enh_key)
enh_sup_df = pd.read_csv(enh_sup)
enh_unsup_df = pd.read_csv(enh_unsup)
w2v_with_df = pd.read_csv(w2v_with)
w2v_without_df = pd.read_csv(w2v_without)

# Save compact selected tables
final_selected = final_df[['model', 'aggregate_auc', 'aggregate_accuracy', 'aggregate_f1', 'aggregate_mcc']].copy()
final_selected.to_csv(TABLE_DIR / 'final_nested_cv_selected_metrics.csv', index=False)

dbf_sup_df[['model', 'test_auc', 'test_accuracy', 'test_f1', 'test_mcc']].to_csv(TABLE_DIR / 'data_balance_final_supervised_selected_metrics.csv', index=False)
dbf_unsup_df[['model', 'test_auc', 'test_accuracy', 'test_f1', 'test_mcc']].to_csv(TABLE_DIR / 'data_balance_final_unsupervised_selected_metrics.csv', index=False)
enh_sup_df[['model', 'test_auc', 'test_accuracy', 'test_f1', 'test_mcc']].to_csv(TABLE_DIR / 'data_balance_final_enhanced_supervised_selected_metrics.csv', index=False)
enh_unsup_df[['model', 'test_auc', 'test_accuracy', 'test_f1', 'test_mcc']].to_csv(TABLE_DIR / 'data_balance_final_enhanced_unsupervised_selected_metrics.csv', index=False)

# Figures
plot_metric_bars(final_df, 'model', 'aggregate_auc', 'aggregate_accuracy', FIG_DIR / 'final_nested_cv_supervised_auc_acc.png', 'final: Nested 5x3 Supervised Benchmark')
plot_metric_bars(dbf_sup_df, 'model', 'test_auc', 'test_accuracy', FIG_DIR / 'data_balance_final_supervised_auc_acc.png', 'data_balance_final: Strict Supervised Test Results')
plot_metric_bars(dbf_unsup_df, 'model', 'test_auc', 'test_accuracy', FIG_DIR / 'data_balance_final_unsupervised_auc_acc.png', 'data_balance_final: Strict Unsupervised Test Results')
plot_metric_bars(enh_sup_df, 'model', 'test_auc', 'test_accuracy', FIG_DIR / 'data_balance_final_enhanced_supervised_auc_acc.png', 'data_balance_final/enhanced: Supervised Test Results')
plot_metric_bars(enh_unsup_df, 'model', 'test_auc', 'test_accuracy', FIG_DIR / 'data_balance_final_enhanced_unsupervised_auc_acc.png', 'data_balance_final/enhanced: Unsupervised Test Results')
plot_wav2vec_summary(w2v_with_df, w2v_without_df, FIG_DIR / 'wav2vec_supervised_aug_vs_noaug.png')

# Overall selected comparisons
final_auc_best = final_df.sort_values('aggregate_auc', ascending=False).iloc[0]
final_acc_best = final_df.sort_values('aggregate_accuracy', ascending=False).iloc[0]
dbf_sup_best = dbf_sup_df.sort_values('test_auc', ascending=False).iloc[0]
dbf_unsup_auc = dbf_unsup_df.sort_values('test_auc', ascending=False).iloc[0]
dbf_unsup_acc = dbf_unsup_df.sort_values('test_accuracy', ascending=False).iloc[0]
enh_sup_best = enh_sup_df.sort_values('test_auc', ascending=False).iloc[0]
enh_unsup_best = enh_unsup_df.sort_values('test_auc', ascending=False).iloc[0]
w2v_with_auc_best = w2v_with_df.sort_values('auc_mean', ascending=False).iloc[0]
w2v_without_acc_best = w2v_without_df.sort_values('accuracy_mean', ascending=False).iloc[0]

supervised_points = pd.DataFrame([
    {'label': f"final {final_auc_best['model']}", 'auc': float(final_auc_best['aggregate_auc']), 'acc': float(final_auc_best['aggregate_accuracy']), 'source': 'final'},
    {'label': f"final {final_acc_best['model']}", 'auc': float(final_acc_best['aggregate_auc']), 'acc': float(final_acc_best['aggregate_accuracy']), 'source': 'final'},
    {'label': f"strict {dbf_sup_best['model']}", 'auc': float(dbf_sup_best['test_auc']), 'acc': float(dbf_sup_best['test_accuracy']), 'source': 'data_balance_final'},
    {'label': f"enhanced {enh_sup_best['model']}", 'auc': float(enh_sup_best['test_auc']), 'acc': float(enh_sup_best['test_accuracy']), 'source': 'data_balance_final_enhanced'},
    {'label': f"wav2vec+aug {w2v_with_auc_best['model']}", 'auc': float(w2v_with_auc_best['auc_mean']), 'acc': float(w2v_with_auc_best['accuracy_mean']), 'source': 'wav2vec'},
    {'label': f"wav2vec no-aug {w2v_without_acc_best['model']}", 'auc': float(w2v_without_acc_best['auc_mean']), 'acc': float(w2v_without_acc_best['accuracy_mean']), 'source': 'wav2vec'},
])
plot_overall_scatter(supervised_points, FIG_DIR / 'overall_supervised_selected_models.png', 'Overall Supervised Paths: AUROC vs ACC')
supervised_points.to_csv(TABLE_DIR / 'overall_supervised_selected_models.csv', index=False)

unsupervised_points = pd.DataFrame([
    {'label': f"strict {dbf_unsup_auc['model']}", 'auc': float(dbf_unsup_auc['test_auc']), 'acc': float(dbf_unsup_auc['test_accuracy']), 'source': 'data_balance_final'},
    {'label': f"strict {dbf_unsup_acc['model']}", 'auc': float(dbf_unsup_acc['test_auc']), 'acc': float(dbf_unsup_acc['test_accuracy']), 'source': 'data_balance_final'},
    {'label': f"enhanced {enh_unsup_best['model']}", 'auc': float(enh_unsup_best['test_auc']), 'acc': float(enh_unsup_best['test_accuracy']), 'source': 'data_balance_final_enhanced'},
])
plot_overall_scatter(unsupervised_points, FIG_DIR / 'overall_unsupervised_selected_models.png', 'Overall Unsupervised Paths: AUROC vs ACC')
unsupervised_points.to_csv(TABLE_DIR / 'overall_unsupervised_selected_models.csv', index=False)

manifest = pd.DataFrame(copied)
manifest.to_csv(SELECTED / 'copied_artifact_manifest.csv', index=False)

summary = {
    'final_auc_best': {'model': str(final_auc_best['model']), 'auc': float(final_auc_best['aggregate_auc']), 'acc': float(final_auc_best['aggregate_accuracy'])},
    'final_acc_best': {'model': str(final_acc_best['model']), 'auc': float(final_acc_best['aggregate_auc']), 'acc': float(final_acc_best['aggregate_accuracy'])},
    'strict_supervised_best': {'model': str(dbf_sup_best['model']), 'auc': float(dbf_sup_best['test_auc']), 'acc': float(dbf_sup_best['test_accuracy'])},
    'strict_unsupervised_auc_best': {'model': str(dbf_unsup_auc['model']), 'auc': float(dbf_unsup_auc['test_auc']), 'acc': float(dbf_unsup_auc['test_accuracy'])},
    'strict_unsupervised_acc_best': {'model': str(dbf_unsup_acc['model']), 'auc': float(dbf_unsup_acc['test_auc']), 'acc': float(dbf_unsup_acc['test_accuracy'])},
    'enhanced_supervised_best': {'model': str(enh_sup_best['model']), 'auc': float(enh_sup_best['test_auc']), 'acc': float(enh_sup_best['test_accuracy'])},
    'enhanced_unsupervised_best': {'model': str(enh_unsup_best['model']), 'auc': float(enh_unsup_best['test_auc']), 'acc': float(enh_unsup_best['test_accuracy'])},
    'wav2vec_aug_auc_best': {'model': str(w2v_with_auc_best['model']), 'auc': float(w2v_with_auc_best['auc_mean']), 'acc': float(w2v_with_auc_best['accuracy_mean'])},
    'wav2vec_noaug_acc_best': {'model': str(w2v_without_acc_best['model']), 'auc': float(w2v_without_acc_best['auc_mean']), 'acc': float(w2v_without_acc_best['accuracy_mean'])},
}
(SELECTED / 'project_summary_snapshot.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))

