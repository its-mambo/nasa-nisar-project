"""Check that moved browser files still resolve through their public URLs."""
from html.parser import HTMLParser
from types import SimpleNamespace
import unittest

from nisar.app.paths import FRONTEND_ROOT
from nisar.app.server import create_app
from nisar.app.web import ICON_FILES, PAGES, STATIC_FILES
from tests.support import request


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "img"} and attrs.get("src", "").startswith("/"):
            self.urls.add(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.urls.add(attrs["href"])


class BrowserResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_pages_and_all_referenced_assets_use_feature_files(self):
        app = create_app(SimpleNamespace(), FRONTEND_ROOT, ())
        references = set()
        for url, relative_path in PAGES.items():
            with self.subTest(page=url):
                status, contents = await request(app, "GET", url)
                self.assertEqual(status, 200)
                self.assertEqual(contents, (FRONTEND_ROOT / relative_path).read_bytes())
                parser = AssetReferences()
                parser.feed(contents.decode())
                references.update(parser.urls)

        # Include icons set dynamically by JavaScript as well as HTML references.
        expected_assets = {"/" + name: FRONTEND_ROOT / path
                           for name, (path, _) in STATIC_FILES.items()}
        expected_assets.update({f"/icons/{name}.svg": FRONTEND_ROOT / "shared/icons" / f"{name}.svg"
                                for name in ICON_FILES})
        self.assertTrue(references <= expected_assets.keys(), references - expected_assets.keys())
        for url, path in expected_assets.items():
            with self.subTest(asset=url):
                status, contents = await request(app, "GET", url)
                self.assertEqual(status, 200)
                self.assertEqual(contents, path.read_bytes())

    async def test_repository_and_model_files_are_not_public(self):
        app = create_app(SimpleNamespace(), FRONTEND_ROOT, ())
        for url in ("/models/model.safetensors", "/nisar/app/cli.py", "/README.md",
                    "/icons/missing.svg", "/icons/../app/index.html"):
            with self.subTest(url=url):
                status, _ = await request(app, "GET", url)
                self.assertEqual(status, 404)
