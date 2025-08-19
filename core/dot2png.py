import subprocess

def dot_to_png(dot_path: str, generate_tree: int):
    graph_dot_file = f"{dot_path}/graph.dot"
    graph_png_file = f"{dot_path}/graph.png"
    subprocess.run(['dot', '-Tpng', '-Gdpi=300', graph_dot_file, '-o', graph_png_file], check=True)
    print(f"Transfered {graph_dot_file} to {graph_png_file}.")

    if generate_tree:
        tree_dot_file = f"{dot_path}/tree.dot"
        tree_png_file = f"{dot_path}/tree.png"
        subprocess.run(['dot', '-Tpng', '-Gdpi=300', '-Nfontname=Helvetica', '-Nfontsize=10', tree_dot_file, '-o', tree_png_file], check=True)
        print(f"Transfered {tree_dot_file} to {tree_png_file}.")