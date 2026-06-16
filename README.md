# BACON — Project Page

Project page for **"Last But Not Least: Boundary Attention Calibration for Multimodal KV Cache Compression"**.

Once GitHub Pages is enabled for this branch, the page will be served at
`https://ryu1ion.github.io/official_BACON/`.

To enable GitHub Pages:
1. Repo → **Settings** → **Pages**.
2. Source: **Deploy from a branch**.
3. Branch: **`gh-pages`**, folder: **`/ (root)`**.

## Local preview

```bash
python3 -m http.server -d . 8000
# open http://localhost:8000
```

## Layout

```
.
├── index.html              # the project page
├── static/
│   ├── css/index.css       # custom styles (Bulma + custom)
│   ├── js/index.js         # smooth-scroll helper
│   └── images/             # paper figures (converted from PDF)
└── .nojekyll               # disable Jekyll processing on GitHub Pages
```
