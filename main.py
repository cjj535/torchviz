import sys

def main(train_model = 'None'):
    if train_model != 'None':
        # 劫持profiler函数
        from hijack_function.hijack_profiler import hijack_profiler
        hijack_profiler()

        # 跑训练过程，获取原始的json格式数据
        if train_model == 'simple':
            from sample.train import train
            train()
        elif train_model == 'resnet':
            from sample.resnet_train import train
            train()

    # 将原始数据转为dot文件
    from util.json2dot import transfer_json_to_dot
    transfer_json_to_dot()

    # 将dot文件可视化
    import subprocess
    dot_file = "./sample/sample_result/graph.dot"
    png_file = "./sample/sample_result/graph.png"
    subprocess.run(['dot', '-Tpng', '-Gdpi=300', dot_file, '-o', png_file], check=True)
    print("Generated ./sample/sample_result/graph.png")

if __name__ == '__main__':
    train_model = str(sys.argv[1]) if len(sys.argv) > 1 else 'None'
    main(train_model)