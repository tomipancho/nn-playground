# Live MNIST digit recognition

An offline, local browser app that calls **your actual trained NumPy network** every ~1 second while you draw. It uses the `network`, `ReLU`, and `del_ReLU` definitions from **your** `utils.py` and loads weights saved with `np.savez`.

## Quick start

1. Put `mnist_live.py` and `index.html` in the same folder as `utils.py` and `best_mnist.npz`. (Or use `--utils-dir` / `--weights` to point to their locations.)
2. Ensure NumPy and your existing `utils.py` dependencies are installed (e.g. `pip install numpy scipy`). The browser app itself uses no third-party Python web framework.
3. Run:

   ```bash
   python mnist_live.py --weights best_mnist.npz
   ```

4. Open **http://127.0.0.1:8765** in a browser. Draw in white on black. Predictions update while you draw and immediately after you lift the pen.

## Saving weights in the expected format

Run this on a **trained** instance called `NN`:

```python
np.savez(
    "best_mnist.npz",
    **{f"W{i}": layer.W for i, layer in enumerate(NN.layers)},
    **{f"b{i}": layer.b for i, layer in enumerate(NN.layers)},
)
```

The app assumes architecture `[784, 128, 64, 10]`, ReLU on both hidden layers, and identity logits on the output. If you've changed those, edit `SPEC` and the activation lists in `load_network()`.

If `utils.py` is in your project but outside the app's directory:

```bash
python mnist_live.py --utils-dir /path/to/your/project --weights /path/to/best_mnist.npz
```

## How the drawing becomes a model input

- White strokes on black background, like MNIST (not black ink on white paper).
- By default, the app crops to the ink bounding box, resizes the longer edge to **20 pixels**, and shifts the intensity center of mass to the center of a **28×28** image. This approximates common MNIST preprocessing, but does not reproduce it exactly.
- Each grayscale pixel is normalized to `[0, 1]` and flattened row-by-row into shape `(784, 1)`.
- The Python server calls `NN(x)[-1]` to obtain 10 logits and converts them to probabilities using a stable softmax.
- You can uncheck **Center & scale** to send a plain downsampled 28×28 drawing instead.

**Important:** Model behavior depends on whether your MNIST training images were white-on-black and normalized to `[0, 1]`. If you trained with a different pixel convention (e.g. 0–255 or inverted), modify the input preprocessing accordingly.

The app only binds to `127.0.0.1`, so it does not expose the model on your network. No image data is sent to external services.
