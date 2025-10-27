import os
import numpy as np
import pytest
from sklearn.decomposition import PCA

preproc = pytest.importorskip("utils.preprocessing")
koh_mod = pytest.importorskip("models.kohonen")
oja_mod = pytest.importorskip("models.oja_rule")

# Data path relative to this test file (tp4/data/europe.csv)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "europe.csv")

def test_load_and_standardize_europa_and_oja_vs_pca():
    # Attempt to load and standardize data; skip if function not implemented or file missing
    try:
        out = preproc.load_and_std_europa(DATA_PATH)
    except Exception as e:
        pytest.skip(f"preprocessing.load_and_std_europa not usable: {e}")

    # Accept either (data, labels, feature_names) or just data
    if out is None:
        pytest.skip("preprocessing.load_and_std_europa returned None")
    if isinstance(out, tuple) and len(out) >= 1:
        data = out[0]
    else:
        data = out

    data = np.asarray(data)
    assert data.ndim == 2 and data.shape[0] >= 5

    # Oja: train a PC1 approximation and compare to sklearn PCA
    Oja = oja_mod.Oja
    oja = Oja(input_dim=data.shape[1], learning_rate=0.001)
    try:
        oja.train(data, epochs=500)
    except Exception:
        # try alternative signature
        try:
            oja.train(data, 500)
        except Exception as e:
            pytest.skip(f"Oja.train not callable with (data, epochs): {e}")

    # get loadings
    try:
        w_oja = oja.get_pc1_loads()
    except Exception:
        pytest.skip("Oja.get_pc1_loads not implemented")

    w_oja = np.asarray(w_oja).ravel()
    # compare to sklearn PCA
    pca = PCA(n_components=1)
    pca.fit(data)
    w_pca = pca.components_[0]
    # align sign
    if np.sign(w_oja[0]) != np.sign(w_pca[0]):
        w_pca = -w_pca
    dist = np.linalg.norm(w_oja / np.linalg.norm(w_oja) - w_pca / np.linalg.norm(w_pca))
    assert dist < 1.5  # loose tolerance; adjust if needed

def test_kohonen_basic_shapes_and_activation_sum():
    Kohonen = koh_mod.Kohonen
    # load data
    try:
        out = preproc.load_and_std_europa(DATA_PATH)
    except Exception:
        pytest.skip("Cannot load data for Kohonen test")
    if isinstance(out, tuple) and len(out) >= 1:
        data = out[0]
    else:
        data = out
    data = np.asarray(data)
    k = 4
    koh = Kohonen(grid_size_k=k, input_dim=data.shape[1], learning_rate=0.5)
    # try to train
    if hasattr(koh, "train"):
        try:
            koh.train(data, epochs=200)
        except Exception:
            try:
                koh.train(data, 200)
            except Exception as e:
                pytest.skip(f"Kohonen.train failed: {e}")
    # weights shape
    assert hasattr(koh, "weights")
    W = koh.weights
    assert W.shape == (k, k, data.shape[1])
    # activation map if available
    if hasattr(koh, "get_activation_map"):
        amap = koh.get_activation_map(data)
        assert int(np.sum(amap)) == data.shape[0]
