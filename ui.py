import glob
import io
import os
import logging

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import utils5 as ut
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)-15s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("ui")

# Configuration de Streamlit
st.set_page_config(
    page_title="Perles à repasser",
    page_icon="🎯",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_IN = os.path.join(BASE_DIR, "data", "in")

# Fonctions lourdes mises en cache (relancées seulement si les entrées changent)
@st.cache_data(show_spinner=False)
def _build_model(img: np.ndarray, size: int, n_colors: int):
    """Resize canvas + quantification des couleurs -> (small, quant)."""
    return ut.build_quantized(img, size, n_colors)

@st.cache_data(show_spinner=False)
def _render(quant: np.ndarray, scale: int) -> Image.Image:
    """Dessine la grille de perles à l'échelle demandée."""
    return ut.render_beads(quant, scale)


tab1, tab2 = st.tabs(["appli", "readme"])

with tab1:
    st.title(":dart: Perles à repasser")
    st.caption("Transforme une photo en modèle de perles à repasser, couleur par couleur.")

    # ------------------------------------------------------------------ #
    # 1. Téléchargement photo                                             #
    # ------------------------------------------------------------------ #
    st.header("1 · Photo")
    src_col, prev_col = st.columns([3, 2])
    with src_col:
        mode = st.radio(
            "Source de l'image",
            ["Galerie (data/in)", "Téléverser un fichier"],
            horizontal=True,
            key="src_mode",
        )
        img, src_name = None, None
        if mode == "Galerie (data/in)":
            files = sorted(
                f for f in glob.glob(os.path.join(DATA_IN, "*"))
                if f.lower().endswith(ut.IMG_EXTENSIONS)
            )
            if not files:
                st.warning(f"Aucune image trouvée dans `{DATA_IN}`.")
            else:
                path = st.selectbox(
                    "Choisir une image", files,
                    format_func=lambda p: os.path.basename(p), key="galerie",
                )
                img = ut.load_image(path)
                src_name = os.path.basename(path)
        else:
            up = st.file_uploader(
                "Déposer une image ici",
                type=[e.lstrip(".") for e in ut.IMG_EXTENSIONS],
                key="uploader",
            )
            if up is not None:
                img = ut.load_image(up)
                src_name = up.name

        if img is None:
            st.info("Charge une photo pour commencer 🙂")
            st.stop()
        logger.info("image chargée : %s, shape=%s", src_name, img.shape)
    with prev_col:
        st.image(img, caption=f"{src_name} — {img.shape[1]}×{img.shape[0]} px", use_container_width=True)

    # ------------------------------------------------------------------ #
    # 2. Rotation photo                                                   #
    # ------------------------------------------------------------------ #
    st.header("2 · Rotation")
    k = st.radio(
        "Rotation (sens anti-horaire)",
        [0, 1, 2, 3],
        format_func=lambda v: f"{v * 90}°",
        horizontal=True,
        key="rotation",
    )
    img_rot = ut.rotate_image(img, k)
    if k:
        with st.expander("Aperçu après rotation", expanded=False):
            st.image(img_rot, caption=f"rotation {k * 90}° — {img_rot.shape[1]}×{img_rot.shape[0]} px",
                     use_container_width=True)

    # ------------------------------------------------------------------ #
    # 3. Taille du canvas                                                 #
    # ------------------------------------------------------------------ #
    st.header("3 · Taille du canvas et couleurs")
    c1, c2, c3 = st.columns(3)
    with c1:
        size = st.slider("Côté du canvas (nombre de perles)", 8, 120, 48, key="size")
    with c2:
        n_colors = st.slider("Nombre de couleurs", 2, 64, 12, key="n_colors")
    with c3:
        default_scale = max(4, min(16, 640 // size))
        scale = st.slider("Zoom d'affichage (px par perle)", 4, 16, default_scale, key="scale")

    # ------------------------------------------------------------------ #
    # 4. Génération du modèle perles à repasser                           #
    # ------------------------------------------------------------------ #
    st.header("4 · Génération du modèle")
    if st.button("🧵 Générer le modèle de perles", type="primary", use_container_width=True):
        with st.spinner("Recadrage, réduction au canvas, quantification des couleurs…"):
            small, quant = _build_model(img_rot, size, n_colors)
            logger.info("modèle généré : canvas %dx%d, %d couleurs demandées, %d utilisées",
                        size, size, n_colors, len(ut.palette(quant)))
        st.session_state["model"] = {"quant": quant, "size": size, "src": src_name}
        st.session_state.pop("last_click", None)
        st.success(f"Modèle prêt : {size}×{size} perles, {len(ut.palette(quant))} couleurs.")

    model = st.session_state.get("model")
    if model is not None:
        quant, size_m = model["quant"], model["size"]
        beads = _render(quant, scale)

        canvas_col, info_col = st.columns([3, 2])
        with canvas_col:
            st.caption("👉 Clique sur une perle du modèle pour connaître sa couleur.")
            click = streamlit_image_coordinates(source=beads, key="canvas")
            if click and click.get("x") is not None:
                st.session_state["last_click"] = (int(click["x"]), int(click["y"]))

        with info_col:
            last = st.session_state.get("last_click")
            if last:
                x, y = last
                col = min(size_m - 1, max(0, x // scale))
                row = min(size_m - 1, max(0, y // scale))
                rgb = quant[row, col]
                hexc = ut.to_hex(rgb)
                st.markdown(f"**Perle cliquée** : ligne {row + 1} · colonne {col + 1}")
                st.markdown(
                    f'<div style="display:inline-block;width:44px;height:44px;'
                    f'background:{hexc};border:2px solid #333;border-radius:50%;'
                    f'vertical-align:middle"></div>'
                    f' &nbsp; <code>{hexc}</code> — RGB{tuple(int(v) for v in rgb)}',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Aucune perle sélectionnée pour le moment.")

            with st.expander(f"🎨 Palette ({len(ut.palette(quant))} couleurs)", expanded=True):
                for idx, c in enumerate(ut.palette(quant), start=1):
                    st.markdown(
                        f'<div style="display:inline-block;width:16px;height:16px;'
                        f'background:{ut.to_hex(c)};border:1px solid #999;border-radius:50%;'
                        f'margin-right:6px;vertical-align:middle"></div>'
                        f"**{idx}** · `{ut.to_hex(c)}`",
                        unsafe_allow_html=True,
                    )

            with st.expander("💾 Exporter", expanded=False):
                buf = io.BytesIO()
                beads.save(buf, format="PNG")
                st.download_button(
                    "📥 PNG du modèle",
                    buf.getvalue(),
                    file_name=f"perles_{model['src'] or 'modele'}.png",
                    mime="image/png",
                )
                st.download_button(
                    "📥 CSV de la grille (ligne, colonne, couleur)",
                    ut.grid_to_csv(quant),
                    file_name="grille_perles.csv",
                    mime="text/csv",
                )

with tab2:
    # readme
    st.header("📖 Readme — comment ça marche ?")
    st.markdown(
        """
        Cette appli génère un **canvas en perles à repasser** à partir d'une photo :
        chaque perle du modèle correspond à un pixel de l'image transformée.

        ## Pipeline (onglet « appli »)

        1. **Photo** — choisie dans la galerie `data/in/` ou téléversée
           (png, jpg, jpeg, bmp, webp, gif).
        2. **Rotation** — 0°, 90°, 180° ou 270° (`numpy.rot90`).
        3. **Taille du canvas** — l'image est recadrée au carré (centre), réduite par
           **max-pooling 2×2 répété** (équivalent `MaxPool2D` du notebook
           `notebooks/main_v0.ipynb`) jusqu'au côté demandé, puis ajustée exactement
           à `size × size`. Le nombre de couleurs est réduit par **K-Means**
           (scikit-learn, avec repli sur la quantification PIL si sklearn est absent).
        4. **Génération** — chaque pixel devient une **perle** : disque de la couleur
           quantifiée avec un petit trou gris au centre, comme dans le notebook.

        ## Interaction

        - **Clique sur une perle** du modèle pour afficher sa position
          (ligne, colonne) et sa couleur (hex + RGB).
        - La **palette** liste toutes les couleurs utilisées, numérotées.
        - **Exporter** : PNG du modèle et CSV de la grille complète
          (`ligne,colonne,hex,r,g,b`) pour compter les perles couleur par couleur.

        ## Lancer l'appli

        ```bash
        pip install streamlit pillow numpy streamlit-image-coordinates scikit-learn
        streamlit run ui.py
        ```

        ## Structure du projet

        ```
        perles_à_repasser/
        ├── ui.py                  # appli Streamlit (ce fichier)
        ├── utils5.py              # pipeline image (utils5)
        ├── notebooks/main_v0.ipynb  # prototype du pipeline
        ├── data/in/               # photos d'entrée (galerie)
        └── data/out/              # rendus générés
        ```
        """
    )
