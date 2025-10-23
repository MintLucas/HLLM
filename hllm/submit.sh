# 脚本目录 10.30.56.62:/njfs/train-comment/example/gaolin3/hllm 

export ML_HOME=/data0/sina-ml-k8s/sina-ml-1.0-k8s
export PATH=$ML_HOME/bin:$ML_HOME/sbin:$PATH
export HADOOP_HOME=/data0/sina-ml-k8s/hadoop-2.7.3-dfsrouter

ml-submit \
   --app-type "PYTORCH" \
   --app-name "hllm-supertopic" \
   --worker-memory 64G \
   --env-paths LANG=C.UTF-8 \
   --worker-num 1 \
   --worker-cores 32 \
   --worker-gpu-cores 4 \
   --shm-memory 16g \
   --app-node-label A800 \
   --docker-image registry.nevis.sina.com.cn/train-comment/recformer:v1.1 \
   --mount /data1/localdisk/njfs/train-comment:/njfs/train-comment \
   --launch-cmd " bash /njfs/train-comment/example/gaolin3/hllm/start-job.sh"
