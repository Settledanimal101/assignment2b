import json
import argparse
from utils.preprocess import load_data  # 确保导入的是这个函数
from estimate_time import estimate_travel_time
from search import find_top_k_paths
from train import train_and_evaluate

def main(config_file='config.json'):
    print("main() start")

    # 读取配置文件
    with open(config_file, 'r') as f:
        config = json.load(f)

    data_path = config.get('data_path', 'data/boroondara_traffic.csv')
    model_name = config.get('model_name', 'lstm')
    origin = config.get('origin', 2000)
    destination = config.get('destination', 3002)
    k = config.get('top_k', 3)
    run_gui = config.get('run_gui', False)
    site_id = config.get('site_id', 2000)

    print(f"[1] 加载数据并预处理：{data_path}, site_id={site_id}")
    scaler, X, y = load_data(data_path, site_id)


    print(f"[2] 训练模型：{model_name.upper()}")
    # 这里传入site_id，确保train_and_evaluate接受该参数
    model, scaler = train_and_evaluate(model_name=model_name, data_path=data_path, site_id=site_id)

    print("[3] 估算旅行时间... (示例略)")

    # 这里构造的graph只是示例，实际需要你根据数据生成真实的图结构
    graph = {
        origin: [(destination, 1000)],  # 假设一条直达路径，权重为1000秒
        # 更多边...
    }

    print(f"[4] 搜索从 {origin} 到 {destination} 的 Top-{k} 路径")
    top_k_routes = find_top_k_paths(origin, destination, k=k, graph=graph)

    for i, route in enumerate(top_k_routes):
        print(f"\nRoute {i+1}:")
        print("Path:", route['path'])
        print("Estimated Time:", route['total_time'], "seconds")

    if run_gui:
        print("[5] 启动图形界面...")
        import gui
        gui.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='路径配置文件')
    args = parser.parse_args()
    main(args.config)
