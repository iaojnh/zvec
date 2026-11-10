import argparse, hashlib, json, os, pathlib, platform, re, statistics, subprocess, time
p=argparse.ArgumentParser();p.add_argument('--baseline',type=pathlib.Path,required=True);p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--trials',type=int,default=5);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
protocol=pathlib.Path(__file__).resolve().parent
iswin=platform.system()=='Windows';suffix='.exe' if iswin else ''
reader='diskann_file_reader_windows_test' if iswin else 'diskann_file_reader_test'
builds={}
for label,source in [('baseline',a.baseline),('candidate',a.candidate)]:
 source=source.resolve();build=out/('build-'+label);builds[label]=build
 args=['cmake','-S',str(protocol),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DZVEC_SOURCE_DIR='+str(source),'-DENABLE_WERROR=ON']
 with (out/(label+'-build.log')).open('w') as f:
  subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['cmake','--build',str(build),'--parallel',str(min(4,os.cpu_count() or 2)),'--target',reader,'diskann_cache_test','diskann_read_bench'],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('BUILT',label,flush=True)
 for name in [reader,'diskann_cache_test']:
  with (out/(label+'-'+name+'.log')).open('w') as f:subprocess.run([str(build/'bin'/(name+suffix)),'--gtest_output=xml:'+str(out/(label+'-'+name+'.xml'))],cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 print('CORRECTNESS',label,'passed',flush=True)
def run(label,seed,budget,workers,cold,trial,prepare=False):
 env=os.environ.copy()
 env.update(ZVEC_READ_BENCH_DIR=str(out/'fixture'),ZVEC_READ_BENCH_SEED=str(seed),ZVEC_READ_BENCH_BUDGET=str(budget),ZVEC_READ_BENCH_WORKERS=str(workers),ZVEC_READ_BENCH_COLD=str(int(cold)),ZVEC_READ_BENCH_PREPARE=str(int(prepare)))
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[k]='1'
 path=out/('%s-%s-%s-%s-%s-%s.log'%(label,seed,budget,workers,cold,trial))
 with path.open('w') as f:subprocess.run([str(builds[label]/'bin'/('diskann_read_bench'+suffix))],env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
 if prepare:return
 lines=[s[len('READ_BENCH '):] for s in path.read_text(errors='replace').splitlines() if s.startswith('READ_BENCH ')]
 assert len(lines)==1,path
 result=json.loads(lines[0]);result.update(label=label,trial=trial,log=path.name)
 assert result['errors']==0 and result['recall']>=0.9,result
 return result
results=[]
# AB/BA ordering alternates within each matched seed/quota/worker/cache point.
for seed in [721,2027]:
 run('baseline',seed,64,1,False,'prepare',prepare=True)
 for budget,workers,cold in [(64,1,False),(64,4,False),(16,1,False),(16,4,False),(16,1,True)]:
  for trial in range(a.trials):
   pair={}
   for label in (['baseline','candidate'] if trial%2==0 else ['candidate','baseline']):
    pair[label]=run(label,seed,budget,workers,cold,trial);results.append(pair[label])
    (out/'measurements.json').write_text(json.dumps(results,indent=2))
   b,c=pair['baseline'],pair['candidate']
   assert b['digest']==c['digest'] and b['recall']==c['recall'],pair
   print('PAIR',seed,budget,workers,cold,trial,'qps_ratio',c['qps']/b['qps'],'p99_ratio',c['p99_us']/b['p99_us'],flush=True)
summary=[]
for seed in [721,2027]:
 for budget,workers,cold in [(64,1,False),(64,4,False),(16,1,False),(16,4,False),(16,1,True)]:
  rows=[x for x in results if (x['seed'],x['budget_bytes'],x['workers'],bool(x['cold_managed_cache']))==(seed,budget<<20,workers,cold)]
  b=[x for x in rows if x['label']=='baseline'];c=[x for x in rows if x['label']=='candidate']
  d={'seed':seed,'budget_mib':budget,'workers':workers,'cold_managed_cache':cold,'trials':a.trials}
  for metric in ['qps','p50_us','p99_us','peak_rss','pool_used','bypass_bytes','cache_hits','cache_misses']:
   d[metric+'_baseline']=statistics.median(x[metric] for x in b);d[metric+'_candidate']=statistics.median(x[metric] for x in c)
  d['qps_ratio']=d['qps_candidate']/d['qps_baseline'];d['p99_ratio']=d['p99_us_candidate']/d['p99_us_baseline'];summary.append(d)
provenance={'platform':platform.platform(),'machine':platform.machine(),'baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.baseline,text=True).strip(),'candidate':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.candidate,text=True).strip(),'binaries':{label:hashlib.sha256((build/'bin'/('diskann_read_bench'+suffix)).read_bytes()).hexdigest() for label,build in builds.items()},'limitations':['Synthetic clustered corpus, 20000 rows x 256 FP32; no real-data acceptance claim.','Cold means empty managed cache after open; OS caches are not dropped.','Peak RSS is whole-process high-water mark, including the fixed fixture.'],'merged':False}
(out/'report.json').write_text(json.dumps({'complete':True,'provenance':provenance,'summary':summary},indent=2))
print(json.dumps({'complete':True,'summary':summary}),flush=True)
