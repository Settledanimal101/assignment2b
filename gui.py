import tkinter as tk
from tkinter import ttk
import json
from train import train_and_evaluate
from search import find_top_k_paths

class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TBRGS Traffic Prediction")

        # 读取配置文件
        with open("config.json", "r") as f:
            self.config = json.load(f)

        ttk.Label(root, text="Origin SiteID").grid(row=0, column=0)
        self.origin_entry = ttk.Entry(root)
        self.origin_entry.grid(row=0, column=1)

        ttk.Label(root, text="Destination SiteID").grid(row=1, column=0)
        self.dest_entry = ttk.Entry(root)
        self.dest_entry.grid(row=1, column=1)

        self.model_var = tk.StringVar(value=self.config["models"][0])
        ttk.Label(root, text="Model").grid(row=2, column=0)
        ttk.Combobox(root, textvariable=self.model_var, values=self.config["models"]).grid(row=2, column=1)

        ttk.Button(root, text="Train & Predict", command=self.run_prediction).grid(row=3, columnspan=2)
        self.output = tk.Text(root, height=10, width=60)
        self.output.grid(row=4, columnspan=2)

    def run_prediction(self):
        try:
            origin = int(self.origin_entry.get())
            dest = int(self.dest_entry.get())
        except ValueError:
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, "请输入有效的数字SiteID。\n")
            return

        model_name = self.model_var.get().lower()
        data_path = self.config["data_path"]
        site_id = self.config["default_site_id"]

        model, scaler = train_and_evaluate(model_name=model_name, data_path=data_path, site_id=site_id)

        # TODO: 从文件或数据库载入真实图数据
        graph = {}  # 你需要实现加载图的函数或从文件读入

        paths = find_top_k_paths(origin, dest, k=self.config["top_k"], graph=graph)

        self.output.delete(1.0, tk.END)
        if not paths:
            self.output.insert(tk.END, "没有找到路径。\n")
        for i, route in enumerate(paths):
            self.output.insert(tk.END, f"Route {i+1}: {' → '.join(map(str, route['path']))} | ETA: {int(route['total_time'])} sec\n")
