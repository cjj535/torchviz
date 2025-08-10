import sample.train

import util.json2dot

import subprocess
dot_file = "./sample/sample_result/graph.dot"
png_file = "./sample/sample_result/graph.png"
subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)