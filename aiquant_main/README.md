# workflow说明, 该内容提供了mlp和gru两个样例
## 各文件夹说明
1. aiquant：基类和通用功能位置，后续会成为pip包
2. cache：数据存放处，包含模型参数、预测结果、数据集缓存等
3. user：各用户个性化操作存放位置
4. workflows：规范化工作流
5. wheel:aiquant包
### 以users/xhn为例
* datasets:数据集构建
* features：特征生成
* models：模型代码
* schedulers：rolling分区
* utils：个性化函数


## 运行说明
1. 修改xhn/utils/config.py中，cache路径，dataset路径
2. xhn/features/zn_sample.py构建样本数据（样例数据已上传，不运行，因为需与csccta配合使用，尚未标准化，所以参考逻辑即可）
3. 运行workflow/ai_workflow_xhn_001.py进行训练（需修改其中sys路径）
4. workflow/ai_workflow_xhn_002.py为gru样例，dataset丰富了操作
5. 结果可在cache中查看