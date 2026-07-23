from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
ROOT = Path(__file__).resolve().parents[1]
pca=pd.read_csv(ROOT/'data/processed/stage4_selected_pca_scores.csv')
X=pca[['PC1','PC2','PC3','PC4','PC5']].to_numpy(float)
cities=pca.city.astype(str); recs=pca.recording_id.astype(str)
rng=np.random.default_rng(20260723); unique_recs=np.array(sorted(recs.unique()))
summary=[]; details=[]; assign=pca[['trajectory_uid','city','recording_id','track_id']].copy()
for k in [2,4,5]:
 name=f'kmeans_k{k}'
 ref=KMeans(n_clusters=k,n_init=5,random_state=42).fit_predict(X); assign[name]=ref
 seed_labels=[]
 for seed in [7,17,27]:
  labels=KMeans(n_clusters=k,n_init=1,random_state=seed).fit_predict(X); seed_labels.append(labels)
  details.append({'config_name':name,'test_type':'seed','test_id':seed,'ari':adjusted_rand_score(ref,labels)})
 pair=[adjusted_rand_score(seed_labels[i],seed_labels[j]) for i in range(3) for j in range(i+1,3)]
 boot=[]; boot_min=[]
 for rep in range(3):
  idx=rng.choice(len(X),size=int(.8*len(X)),replace=True)
  m=KMeans(n_clusters=k,n_init=1,random_state=1000+rep).fit(X[idx]); labels=m.predict(X)
  _,cnt=np.unique(labels,return_counts=True); ari=adjusted_rand_score(ref,labels)
  boot.append(ari); boot_min.append(cnt.min()/len(X)*100)
  details.append({'config_name':name,'test_type':'bootstrap','test_id':rep+1,'ari':ari,'smallest_cluster_percentage':cnt.min()/len(X)*100})
 city=[]; city_min=[]; present=[]
 for c in sorted(cities.unique()):
  train=(cities!=c).to_numpy(); m=KMeans(n_clusters=k,n_init=3,random_state=42).fit(X[train]); labels=m.predict(X); held=labels[~train]
  _,cnt=np.unique(held,return_counts=True); ari=adjusted_rand_score(ref,labels)
  city.append(ari); city_min.append(cnt.min()/len(held)*100); present.append(len(cnt))
  details.append({'config_name':name,'test_type':'leave_city_out','test_id':c,'ari':ari,'smallest_cluster_percentage':cnt.min()/len(held)*100,'clusters_present':len(cnt)})
 rec=[]
 for rep in range(3):
  chosen=rng.choice(unique_recs,size=int(.8*len(unique_recs)),replace=False); train=recs.isin(chosen).to_numpy()
  m=KMeans(n_clusters=k,n_init=1,random_state=3000+rep).fit(X[train]); labels=m.predict(X); ari=adjusted_rand_score(ref,labels); rec.append(ari)
  details.append({'config_name':name,'test_type':'recording_subsample','test_id':rep+1,'ari':ari})
 summary.append({'config_name':name,'model_family':'KMeans','k':k,'seed_pairwise_ari_mean':np.mean(pair),'seed_pairwise_ari_min':np.min(pair),'bootstrap_ari_mean':np.mean(boot),'bootstrap_ari_min':np.min(boot),'bootstrap_smallest_cluster_percentage_min':np.min(boot_min),'leave_one_city_out_ari_mean':np.mean(city),'leave_one_city_out_ari_min':np.min(city),'held_out_city_min_cluster_percentage_min':np.min(city_min),'all_clusters_present_each_city':all(v==k for v in present),'recording_subsample_ari_mean':np.mean(rec),'recording_subsample_ari_min':np.min(rec)})
 print('done',name)
out=ROOT/'outputs/tables'; out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(summary).to_csv(out/'stage5_kmeans_stability_summary.csv',index=False)
pd.DataFrame(details).to_csv(out/'stage5_kmeans_stability_details.csv',index=False)
assign.to_csv(ROOT/'data/processed/stage5_kmeans_candidate_assignments.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
