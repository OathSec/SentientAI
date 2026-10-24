# Project Fuzzy site

Static site for systembridge.site (served by `~/apps/system-bridge-main/server.js` on keystone).

    python site/build.py worker/dist/FuzzyWorker.exe     # builds ../site-build (pages + the download + its SHA-256)

Then copy the contents of `site-build/` into `~/apps/system-bridge-main/public/`. Links use trailing
slashes (`/worker/`) because that server serves any slash-less path as the home page.
