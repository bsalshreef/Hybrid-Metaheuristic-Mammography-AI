import numpy as np
from sklearn.linear_model import LogisticRegression


def test_smoke_pipeline():
    rng = np.random.default_rng(42)

    x0 = rng.normal(0.2, 0.05, size=(4, 10))
    x1 = rng.normal(0.8, 0.05, size=(4, 10))

    X = np.vstack([x0, x1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert X.shape == (8, 10)
    assert preds.shape == (8,)
    assert probs.shape == (8, 2)
    assert np.isfinite(probs).all()
