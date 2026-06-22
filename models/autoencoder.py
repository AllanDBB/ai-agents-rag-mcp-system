"""
Autoencoder convolucional para reconstrucción de imágenes.

El número de epochs NO está fijo: EarlyStopping decide cuándo detener el
entrenamiento, monitoreando val_loss con paciencia configurable. Fijar
manualmente epochs bajos (8-10) produce reconstrucciones de mala calidad;
dejar que el callback lo decida garantiza convergencia adecuada.

Uso:
    from models.autoencoder import build_autoencoder, train

    autoencoder, encoder, decoder = build_autoencoder(input_shape=(128, 128, 3))
    history = train(autoencoder, x_train, x_val)
"""

import numpy as np


def build_autoencoder(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    latent_dim: int = 64,
) -> tuple:
    """Construye encoder, decoder y autoencoder completo.

    Returns:
        (autoencoder, encoder, decoder) — modelos Keras compilados.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # ── Encoder ──────────────────────────────────────────────────────────────
    inp = keras.Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)

    shape_before_flatten = x.shape[1:]
    x = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    encoder = keras.Model(inp, encoded, name="encoder")

    # ── Decoder ──────────────────────────────────────────────────────────────
    lat_inp = keras.Input(shape=(latent_dim,), name="latent_input")
    x = layers.Dense(int(np.prod(shape_before_flatten)), activation="relu")(lat_inp)
    x = layers.Reshape(shape_before_flatten)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2DTranspose(
        input_shape[-1], 3, activation="sigmoid", padding="same", name="reconstruction"
    )(x)

    decoder = keras.Model(lat_inp, decoded, name="decoder")

    # ── Autoencoder completo ──────────────────────────────────────────────────
    autoencoder = keras.Model(inp, decoder(encoder(inp)), name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return autoencoder, encoder, decoder


def train(
    autoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience: int = 15,
    min_delta: float = 1e-5,
    checkpoint_path: str | None = None,
):
    """Entrena el autoencoder con EarlyStopping.

    El training se detiene automáticamente cuando val_loss deja de mejorar
    por `patience` epochs consecutivos, con los mejores pesos restaurados.
    No se usa un máximo fijo bajo de epochs — 500 es un techo de seguridad,
    no el objetivo.

    Args:
        x_train / x_val: arrays float32 normalizados en [0, 1].
        patience: epochs sin mejora antes de detener.
        min_delta: mejora mínima considerada significativa.
        checkpoint_path: si se provee, guarda el mejor modelo en esa ruta.

    Returns:
        keras History object.
    """
    from tensorflow.keras import callbacks as cb

    cbs = [
        cb.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        cb.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    if checkpoint_path:
        cbs.append(
            cb.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    history = autoencoder.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=cbs,
        shuffle=True,
    )
    return history


def evaluate_reconstruction(autoencoder, x_test: np.ndarray) -> dict:
    """Evalúa calidad de reconstrucción. MSE < 0.01 indica buena calidad."""
    import tensorflow as tf

    reconstructed = autoencoder.predict(x_test, verbose=0)
    mse = float(tf.reduce_mean(tf.square(x_test - reconstructed)).numpy())
    mae = float(tf.reduce_mean(tf.abs(x_test - reconstructed)).numpy())
    psnr = float(tf.reduce_mean(tf.image.psnr(x_test, reconstructed, max_val=1.0)).numpy())
    ssim = float(tf.reduce_mean(tf.image.ssim(x_test, reconstructed, max_val=1.0)).numpy())
    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": ssim}
