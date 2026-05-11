"""Tests for the FastAPI backend."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from app.api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Test basic API endpoints."""

    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestPredictEndpoint:
    """Test the /predict endpoint."""

    def test_predict_requires_image(self, client):
        """Missing image should return 422."""
        response = client.post("/predict", data={"text": "hello", "fusion": "early"})
        assert response.status_code == 422

    def test_predict_requires_text(self, client):
        """Missing text should return 422."""
        import io
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"image": ("test.png", buf, "image/png")},
            data={"fusion": "early"},
        )
        assert response.status_code == 422

    def test_predict_returns_json(self, client):
        """Valid request should return prediction JSON."""
        import io
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"image": ("test.png", buf, "image/png")},
            data={"text": "this is a test meme", "fusion": "early"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "offensive_prob" in data
        assert "non_offensive_prob" in data
        assert "label" in data
        assert "confidence" in data

    def test_predict_probabilities_sum_to_one(self, client):
        """offensive_prob + non_offensive_prob should ≈ 1."""
        import io
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"image": ("test.png", buf, "image/png")},
            data={"text": "hello world", "fusion": "early"},
        )
        data = response.json()
        total = data["offensive_prob"] + data["non_offensive_prob"]
        assert abs(total - 1.0) < 1e-5

    def test_predict_label_is_valid(self, client):
        """Label should be either 'Offensive' or 'Non-offensive'."""
        import io
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"image": ("test.png", buf, "image/png")},
            data={"text": "test", "fusion": "early"},
        )
        data = response.json()
        assert data["label"] in ("Offensive", "Non-offensive")

    @pytest.mark.parametrize("fusion", ["early", "cross_attention", "gated", "bilinear"])
    def test_all_fusion_types(self, client, fusion):
        """All fusion types should work."""
        import io
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"image": ("test.png", buf, "image/png")},
            data={"text": "test meme text", "fusion": fusion},
        )
        assert response.status_code == 200
