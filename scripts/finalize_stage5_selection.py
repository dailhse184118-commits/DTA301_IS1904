from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parents[1]
TABLES=ROOT/'outputs/tables'; FIGURES=ROOT/'outputs/figures'; PROCESSED=ROOT/'data/processed'; MODELS=ROOT/'models'
for p in [TABLES,FIGURES,PROCESSED,MODELS]: p.mkdir(parents=True,exist_ok=True)
PCA_COLS=['PC1','PC2','PC3','PC4','PC5']
CORE=['mean_speed_mps','max_speed_mps','speed_std_mps','mean_long_acc_mps2','max_acceleration_mps2','max_deceleration_mps2','acceleration_std_mps2','mean_abs_jerk_mps3','observed_stop_transition_count','stopped_time_ratio']

pca=pd.read_csv(PROCESSED/'stage4_selected_pca_scores.csv')
features=pd.read_csv(PROCESSED/'sind_full_core_behavior_features.csv')
benchmark=pd.read_csv(TABLES/'stage5_model_benchmark.csv')
stability=pd.read_csv(TABLES/'stage5_kmeans_stability_summary.csv')
prod_seed=pd.read_csv(TABLES/'stage5_kmeans_production_seed_stability.csv')

merged=pca[['trajectory_uid','city','recording_id','track_id',*PCA_COLS]].merge(features,on=['trajectory_uid','city','recording_id','track_id'],validate='one_to_one')
X=merged[PCA_COLS].to_numpy(float)
model=KMeans(n_clusters=4,n_init=50,random_state=42)
raw=model.fit_predict(X)
merged['cluster_raw']=raw

# Profile mapping based on original-feature medians, documented and fixed.
profile_map={
  1:(1,'Smooth and Steady'),
  3:(2,'Stop-and-Go'),
  2:(3,'Dynamic Speed Adjustment'),
  0:(4,'Acceleration-Intensive'),
}
merged['profile_id']=merged['cluster_raw'].map(lambda x: profile_map[int(x)][0])
merged['profile_name']=merged['cluster_raw'].map(lambda x: profile_map[int(x)][1])
merged=merged.sort_values('trajectory_uid').reset_index(drop=True)

# Save assignments.
assignment_cols=['trajectory_uid','city','recording_id','track_id','cluster_raw','profile_id','profile_name',*PCA_COLS,'CrossType','Signal_Violation_Behavior']
merged[assignment_cols].to_csv(PROCESSED/'stage5_final_cluster_assignments.csv',index=False)

# Metrics.
metrics={
 'selected_model':'KMeans',
 'selected_k':4,
 'preprocessing_pipeline':'C_winsor_robust',
 'pca_components':5,
 'pca_retained_variance':0.9171,
 'n_init':50,
 'random_state':42,
 'n_trajectories':len(merged),
 'coverage':1.0,
 'silhouette_sample_5000':float(silhouette_score(X,raw,sample_size=5000,random_state=42)),
 'davies_bouldin':float(davies_bouldin_score(X,raw)),
 'calinski_harabasz':float(calinski_harabasz_score(X,raw)),
 'inertia':float(model.inertia_),
}
sel_stab=stability.loc[stability.config_name=='kmeans_k4'].iloc[0].to_dict()
sel_seed=prod_seed.loc[prod_seed.config_name=='kmeans_k4'].iloc[0].to_dict()
metrics.update({
 'production_seed_pairwise_ari_mean':float(sel_seed['pairwise_ari_mean']),
 'production_seed_pairwise_ari_min':float(sel_seed['pairwise_ari_min']),
 'bootstrap_ari_mean':float(sel_stab['bootstrap_ari_mean']),
 'bootstrap_ari_min':float(sel_stab['bootstrap_ari_min']),
 'leave_one_city_out_ari_mean':float(sel_stab['leave_one_city_out_ari_mean']),
 'leave_one_city_out_ari_min':float(sel_stab['leave_one_city_out_ari_min']),
 'recording_subsample_ari_mean':float(sel_stab['recording_subsample_ari_mean']),
 'recording_subsample_ari_min':float(sel_stab['recording_subsample_ari_min']),
})
(TABLES/'stage5_selected_model_config.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
joblib.dump(model,MODELS/'stage5_kmeans_k4_pca_model.joblib')

# Cluster sizes.
sizes=(merged.groupby(['profile_id','profile_name']).size().reset_index(name='trajectory_count'))
sizes['percentage']=sizes.trajectory_count/len(merged)*100
sizes.to_csv(TABLES/'stage5_final_cluster_sizes.csv',index=False)

# Profiles in original features.
profile_median=merged.groupby(['profile_id','profile_name'])[CORE].median().reset_index()
profile_mean=merged.groupby(['profile_id','profile_name'])[CORE].mean().reset_index()
profile_median.to_csv(TABLES/'stage5_cluster_profile_medians.csv',index=False)
profile_mean.to_csv(TABLES/'stage5_cluster_profile_means.csv',index=False)

global_med=merged[CORE].median(); global_iqr=(merged[CORE].quantile(.75)-merged[CORE].quantile(.25)).replace(0,1)
rob=merged.groupby(['profile_id','profile_name'])[CORE].median()
rob=(rob-global_med)/global_iqr
rob.reset_index().to_csv(TABLES/'stage5_cluster_profile_robust_scores.csv',index=False)

# PCA centroids.
centroids=pd.DataFrame(model.cluster_centers_,columns=PCA_COLS)
centroids['cluster_raw']=range(4)
centroids['profile_id']=centroids.cluster_raw.map(lambda x: profile_map[int(x)][0])
centroids['profile_name']=centroids.cluster_raw.map(lambda x: profile_map[int(x)][1])
centroids.sort_values('profile_id').to_csv(TABLES/'stage5_pca_cluster_centroids.csv',index=False)

# Feature eta-squared.
eta=[]
for f in CORE:
 vals=merged[f]; gm=vals.mean(); total=((vals-gm)**2).sum(); between=0
 for _,g in merged.groupby('profile_id'):
  between+=len(g)*(g[f].mean()-gm)**2
 eta.append({'feature':f,'eta_squared':float(between/total if total else 0)})
pd.DataFrame(eta).sort_values('eta_squared',ascending=False).to_csv(TABLES/'stage5_cluster_feature_separation.csv',index=False)

# City composition count and row percentages.
city_counts=pd.crosstab([merged.profile_id,merged.profile_name],merged.city).reset_index()
city_counts.to_csv(TABLES/'stage5_cluster_city_counts.csv',index=False)
city_pct=(pd.crosstab([merged.profile_id,merged.profile_name],merged.city,normalize='index')*100).reset_index()
city_pct.to_csv(TABLES/'stage5_cluster_city_percentages.csv',index=False)

# Recording composition.
rec_counts=merged.groupby(['profile_id','profile_name','city','recording_id']).size().reset_index(name='trajectory_count')
rec_counts['within_recording_percentage']=rec_counts.groupby(['city','recording_id']).trajectory_count.transform(lambda s:s/s.sum()*100)
rec_counts.to_csv(TABLES/'stage5_cluster_recording_composition.csv',index=False)

# Tianjin post-cluster metadata interpretation; clean whitespace only for reporting.
tj=merged.loc[merged.city=='Tianjin'].copy()
for col in ['CrossType','Signal_Violation_Behavior']:
 tj[col]=tj[col].astype('string').str.strip()
ct_count=pd.crosstab([tj.profile_id,tj.profile_name],tj.CrossType).reset_index(); ct_count.to_csv(TABLES/'stage5_tianjin_crosstype_counts.csv',index=False)
ct_pct=(pd.crosstab([tj.profile_id,tj.profile_name],tj.CrossType,normalize='index')*100).reset_index(); ct_pct.to_csv(TABLES/'stage5_tianjin_crosstype_percentages.csv',index=False)
vi_count=pd.crosstab([tj.profile_id,tj.profile_name],tj.Signal_Violation_Behavior).reset_index(); vi_count.to_csv(TABLES/'stage5_tianjin_violation_counts.csv',index=False)
vi_pct=(pd.crosstab([tj.profile_id,tj.profile_name],tj.Signal_Violation_Behavior,normalize='index')*100).reset_index(); vi_pct.to_csv(TABLES/'stage5_tianjin_violation_percentages.csv',index=False)

# Representative trajectories nearest PCA centroid.
reps=[]
for raw_id,center in enumerate(model.cluster_centers_):
 idx=np.where(raw==raw_id)[0]
 dist=np.linalg.norm(X[idx]-center,axis=1)
 for rank,local in enumerate(np.argsort(dist)[:10],start=1):
  i=idx[local]
  reps.append({'cluster_raw':raw_id,'profile_id':profile_map[raw_id][0],'profile_name':profile_map[raw_id][1],'representative_rank':rank,'trajectory_uid':merged.iloc[i].trajectory_uid,'city':merged.iloc[i].city,'recording_id':merged.iloc[i].recording_id,'track_id':merged.iloc[i].track_id,'distance_to_centroid':float(dist[local])})
pd.DataFrame(reps).sort_values(['profile_id','representative_rank']).to_csv(TABLES/'stage5_representative_trajectories.csv',index=False)

# Decision matrix using actual benchmark evidence.
def br(name):
 return benchmark.loc[benchmark.config_name==name].iloc[0]
def st(name):
 r=stability.loc[stability.config_name==name]
 return r.iloc[0] if len(r) else None
rows=[]
for name,label,decision,reason in [
 ('kmeans_k2','K-Means, k=2','Not selected','Best compactness but too coarse; recording-group stability was much weaker and it collapses distinct dynamic profiles.'),
 ('kmeans_k4','K-Means, k=4','Selected primary model','Best balance of full coverage, production seed stability, recording robustness, four interpretable profiles, and manageable complexity.'),
 ('kmeans_k5','K-Means, k=5','Sensitivity candidate','Higher feature separation but lower bootstrap, seed, city, and recording stability; fifth cluster adds a less robust split.'),
 ('minibatch_k4','MiniBatch K-Means, k=4','Not selected','Similar structure but slightly weaker internal metrics; scalability advantage is unnecessary for 19,948 trajectories.'),
 ('gmm_k4','Gaussian Mixture, k=4','Not selected','Flexible probabilistic model, but substantially lower silhouette and higher Davies–Bouldin score.'),
 ('agglomerative_sample_k4','Agglomerative, k=4 (sample)','Exploratory only','Interpretable sample structure, but full 19,948-row Ward clustering is not memory-scalable and has no direct prediction for held-out cities.'),
 ('dbscan_sample_ms15_q08','DBSCAN (sample best compactness)','Complementary only','High sample silhouette but one cluster is below 1% and 11.17% is noise; sensitive to density parameters.'),
 ('hdbscan_sample_mcs150_ms15','HDBSCAN (sample)','Complementary only','Finds a broad density structure but leaves 39.63% as noise, so it cannot serve as the primary full-coverage profiling model.'),
]:
 r=br(name)
 sr=st(name)
 rows.append({'config_name':name,'candidate':label,'decision':decision,'n_clusters':int(r.n_clusters),'coverage':float(r.coverage),'noise_percentage':float(r.noise_percentage),'silhouette':float(r.silhouette) if pd.notna(r.silhouette) else np.nan,'davies_bouldin':float(r.davies_bouldin) if pd.notna(r.davies_bouldin) else np.nan,'smallest_cluster_percentage':float(r.smallest_cluster_percentage),'cluster_entropy':float(r.cluster_entropy),'mean_feature_eta_squared':float(r.mean_feature_eta_squared),'bootstrap_ari_mean':float(sr.bootstrap_ari_mean) if sr is not None else np.nan,'leave_one_city_out_ari_mean':float(sr.leave_one_city_out_ari_mean) if sr is not None else np.nan,'recording_subsample_ari_mean':float(sr.recording_subsample_ari_mean) if sr is not None else np.nan,'reason':reason})
pd.DataFrame(rows).to_csv(TABLES/'stage5_model_decision_matrix.csv',index=False)

# Figures.
# K sweep metrics.
km=benchmark[benchmark.model_family=='KMeans'].sort_values('n_clusters')
for metric, ylabel in [('silhouette','Silhouette'),('davies_bouldin','Davies-Bouldin'),('mean_feature_eta_squared','Mean feature eta squared')]:
 fig,ax=plt.subplots(figsize=(8,5)); ax.plot(km.n_clusters,km[metric],marker='o'); ax.set_title(f'K-Means {ylabel} by number of clusters'); ax.set_xlabel('k'); ax.set_ylabel(ylabel); ax.set_xticks(km.n_clusters); plt.tight_layout(); plt.savefig(FIGURES/f'stage5_kmeans_{metric}.png',dpi=180,bbox_inches='tight'); plt.close(fig)
# Cluster sizes.
fig,ax=plt.subplots(figsize=(9,5)); ax.bar(sizes.profile_name,sizes.trajectory_count); ax.set_title('Final K-Means behavioral-profile sizes'); ax.set_xlabel('Profile'); ax.set_ylabel('Trajectories'); ax.tick_params(axis='x',rotation=20); plt.tight_layout(); plt.savefig(FIGURES/'stage5_final_cluster_sizes.png',dpi=180,bbox_inches='tight'); plt.close(fig)
# PCA scatter sample.
sample=merged.sample(n=min(6000,len(merged)),random_state=42)
fig,ax=plt.subplots(figsize=(9,6))
for name,g in sample.groupby('profile_name'):
 ax.scatter(g.PC1,g.PC2,s=8,alpha=.5,label=name)
ax.set_title('Final profiles in the first two PCA dimensions'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(); plt.tight_layout(); plt.savefig(FIGURES/'stage5_final_pca_scatter.png',dpi=180,bbox_inches='tight'); plt.close(fig)
# Profile heatmap.
rob_plot=rob.sort_index(level=0)
fig,ax=plt.subplots(figsize=(12,5)); im=ax.imshow(rob_plot.to_numpy(),aspect='auto',vmin=-2.5,vmax=2.5); ax.set_xticks(range(len(CORE))); ax.set_xticklabels(CORE,rotation=90); ax.set_yticks(range(len(rob_plot))); ax.set_yticklabels([idx[1] for idx in rob_plot.index]); fig.colorbar(im,ax=ax,label='Median difference / global IQR'); ax.set_title('Robust original-feature profiles'); plt.tight_layout(); plt.savefig(FIGURES/'stage5_cluster_profile_heatmap.png',dpi=180,bbox_inches='tight'); plt.close(fig)
# City composition stacked.
pct=city_pct.set_index(['profile_id','profile_name']); fig,ax=plt.subplots(figsize=(10,6)); pct.plot(kind='bar',stacked=True,ax=ax); ax.set_title('City composition within each behavioral profile'); ax.set_xlabel('Profile'); ax.set_ylabel('Percentage'); ax.tick_params(axis='x',rotation=20); plt.tight_layout(); plt.savefig(FIGURES/'stage5_cluster_city_composition.png',dpi=180,bbox_inches='tight'); plt.close(fig)

print('Selected model metrics:',json.dumps(metrics,indent=2))
print('\nCluster sizes:')
print(sizes.to_string(index=False))
print('\nProfile medians:')
print(profile_median.round(3).to_string(index=False))
