### 模型全流程文档（训练/离线推理/线上推理）



#### 一、训练


**1 样本格式**


```json
[
    {
        "user_id": "u3328884",

        "history": [
            "text1",
            "text2",
            "text3"
        ]
    },
    ...
]
```

**2. 模型代码**

    code/hllm.py

**3. 提交**

```
#10.30.56.62
sh submit.sh
```

#### 二、离线推理

    uid embedding代码： code/user_embedding_infer.py
    mid embedding代码： code/item_embedding_infer.py

**提交脚本：**

```
#10.30.56.62
#修改start-job.sh中的py文件
sh submit.sh

#torchrun --nproc_per_node=4 /njfs/train-comment/example/gaolin3/hllm/hllm.py
```

#### 三、线上推理

推理任务：
[links](http://dataworks.sina.com.cn/#/machine-learning/llmonline/task/detail?defID=6166B4A3-3FC4-E157-3B87-F2985664616D&projectID=b1466a87-3357-482b-bddd-a221863bad72&type=llmonline)

线上推理代码实现：
[links](https://git.intra.weibo.com/growth_intelligence_algo/hllm_infer/-/blob/master/model_repository/bert-embedding/1/model.py)

调用方式：

```
curl --request POST \
  --url llm-beixian.multimedia.wml.weibo.com/mm-wb-ml-mbdetail/customize-bert-wb-ml-mbdetail-495281/v2/models/bert-embedding/infer \
  --header 'Content-Type: application/json' \
  --data '{"id":"45", "inputs":[{"shape":[1],"datatype":"BYTES","name":"text_input","data":[ "北京哪里适合春游"]}]}'
  
  ```
