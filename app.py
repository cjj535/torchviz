from flask import Flask, render_template
import argparse

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # app.run(debug=True)
    parser = argparse.ArgumentParser(description="ip")

    parser.add_argument("--ip", type=str, default='0.0.0.0', help="ip", required=False)

    args = parser.parse_args()

    app.run(host=args.ip, port=5000, debug=True)
