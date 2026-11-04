1. 预训练模型路径：data/PV-Ground_ScanRefer_single.pth
2. 预训练模型的评测结果在log/scanrefer/exp/1793811982/log.txt中，接下来所有的评测实验都应该在全量数据集上评测，和预训练模型评测结果比较。
3. 模型训练入口：tools/train_scanrefer_single.sh
4. 模型评测入口：tools/test_scanrefer_single.sh
5. 训练评测时尽可能多卡多batchsize，至少8个num_workers
6. 尽可能在原有的代码修改，非必要避免新增代码文件
7. 不得修改模型评测代码