"""MNIST live drawing demo backed by the user's NumPy network in utils.py.

Run:
    python mnist_live.py --weights best_mnist.npz
Then open http://127.0.0.1:8765

Requires: numpy; your existing utils.py (and any dependencies it imports).
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = [784, 512, 10]


def load_network(weights_path: Path):
    # Prefer the utils.py sitting next to this app. If it lives elsewhere,
    # pass --utils-dir /path/to/project instead.
    from utils import network, ReLU, del_ReLU

    model = network(
        SPEC,
        activation=[ReLU],
        del_activation=[del_ReLU],
    )
    with np.load(weights_path, allow_pickle=False) as stored:
        for i, layer in enumerate(model.layers):
            w_key, b_key = f"W{i}", f"b{i}"
            if w_key not in stored or b_key not in stored:
                raise ValueError(f"Missing {w_key} or {b_key} in {weights_path.name}")
            w = stored[w_key]
            b = stored[b_key]
            if w.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError(
                    f"Layer {i}: expected W{layer.W.shape}, b{layer.b.shape}; "
                    f"found W{w.shape}, b{b.shape}"
                )
            layer.W = np.array(w, copy=True)
            layer.b = np.array(b, copy=True)
    return model


def predict(model, pixels):
    x = np.asarray(pixels, dtype=np.float64)
    if x.shape != (784,):
        raise ValueError("Expected 784 normalized pixels, in row-major order")
    if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Pixel values must be finite numbers between 0 and 1")

    # One example, shape (784, 1), as expected by your network.
    logits = np.asarray(model(x.reshape(784, 1))[-1], dtype=np.float64)
    if logits.shape != (10, 1):
        raise ValueError(f"Expected logits of shape (10, 1), got {logits.shape}")

    # Numerically stable softmax. The model output is *logits*, not probabilities.
    scores = logits[:, 0] - np.max(logits[:, 0])
    exponentials = np.exp(scores)
    p = exponentials / exponentials.sum()
    if not np.all(np.isfinite(p)):
        raise ValueError("The network produced non-finite probabilities")
    return {"probabilities": p.tolist(), "prediction": int(np.argmax(p))}


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, obj):
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                payload = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif route == "/health":
                self.send_json(200, {"ok": True, "spec": SPEC})
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/predict":
                return self.send_json(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 65_536:
                    return self.send_json(413, {"error": "Invalid request size"})
                data = json.loads(self.rfile.read(size))
                result = predict(model, data.get("pixels"))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                return self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                print("Prediction failed:", repr(exc), file=sys.stderr)
                return self.send_json(500, {"error": "Prediction failed; see server console"})
            self.send_json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Draw MNIST digits and query your NumPy NN live")
    parser.add_argument("--weights", default="best_mnist.npz", help="NPZ file containing W0/b0, W1/b1, W2/b2")
    parser.add_argument("--utils-dir", type=Path, default=HERE, help="Directory containing your utils.py")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    utils_dir = args.utils_dir.resolve()
    sys.path.insert(0, str(utils_dir))
    path = Path(args.weights)
    if not path.is_absolute() and not path.exists():
        path = HERE / path
    path = path.resolve()

    if not (utils_dir / "utils.py").exists():
        parser.error(f"Could not find utils.py in {utils_dir} (use --utils-dir)")
    if not path.is_file():
        parser.error(f"Could not find weights file: {path} (use --weights)")

    model = load_network(path)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model))
    print(f"Loaded {path.name} — architecture: {SPEC}")
    print(f"Open http://127.0.0.1:{args.port} in your browser")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
