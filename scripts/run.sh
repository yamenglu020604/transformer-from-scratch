# 设置随机种子以保证可复现性
export PYTHONHASHSEED=42

# 创建必要的目录
mkdir -p checkpoints
mkdir -p results
mkdir -p tokenizer

# 执行训练
# --config 参数指向配置文件，这是最佳实践
python src/train.py --config configs/base_config.yaml