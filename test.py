import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from preprocessing import (
    create_dataset,
    frames_extraction,
    load_labels,
    SEQUENCE_LENGTH,
    DATASET_DIR,
    seed_constant,
)


def get_classes_list(dataset_dir=DATASET_DIR):
    """
    Rebuilds the same classes_list used during training. create_dataset() in
    preprocessing.py derives it as sorted(set(label_map.values())), which is
    deterministic - so calling load_labels() again on the same dataset_dir
    gives back the identical index -> word mapping the model was trained
    with, without needing convlstm.py to have saved it separately.

    IMPORTANT: this only works if dataset_dir still points at the same
    dataset (same videos/labels.csv) used at training time. If the dataset
    has since changed (files added/removed/renamed), this mapping can drift
    from what the model actually learned.
    """
    label_map = load_labels(dataset_dir)
    if not label_map:
        raise ValueError(
            f"Tidak bisa menentukan daftar kelas: tidak ada video/labels.csv "
            f"ditemukan di '{dataset_dir}'. Pastikan path ini sama dengan "
            f"DATASET_DIR yang dipakai saat training."
        )
    return sorted(set(label_map.values()))


def predict_on_video(video_path, model_path="convlstm_model.h5", dataset_dir=DATASET_DIR):
    """
    Loads the trained ConvLSTM model and predicts the class of a single
    video file, using the exact same frame extraction (frames_extraction)
    as training so preprocessing stays consistent between train and test.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    classes_list = get_classes_list(dataset_dir)
    print(f"Classes ({len(classes_list)}): {classes_list}")

    model = load_model(model_path)

    frames = frames_extraction(video_path)
    if len(frames) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Gagal mengekstrak frame dari video ini (dapat {len(frames)}, "
            f"butuh {SEQUENCE_LENGTH}). Video mungkin corrupt atau kosong."
        )

    input_tensor = np.expand_dims(np.array(frames), axis=0)  # shape: [1, T, H, W, 3]

    predicted_probabilities = model.predict(input_tensor)[0]
    predicted_index = int(np.argmax(predicted_probabilities))
    predicted_label = classes_list[predicted_index]
    confidence = float(predicted_probabilities[predicted_index])

    print(f"\nVideo: {video_path}")
    print(f"Predicted class: {predicted_label} ({confidence * 100:.2f}% confidence)")

    print("\nFull probabilities:")
    ranked = sorted(zip(classes_list, predicted_probabilities), key=lambda x: -x[1])
    for label, prob in ranked:
        print(f"  {label:<15s} {prob * 100:6.2f}%")

    return predicted_label, confidence


def plot_confusion_matrix(
    model_path="convlstm_model.h5",
    dataset_dir=DATASET_DIR,
    output_path="confusion_matrix.png",
    test_size=0.25,
):
    """
    Evaluates the model on the same deterministic holdout split used in
    training and saves a confusion-matrix plot.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not 0 < test_size < 1:
        raise ValueError("test_size harus berada di antara 0 dan 1.")

    features, labels, classes_list = create_dataset(dataset_dir)
    _, features_test, _, labels_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        shuffle=True,
        random_state=seed_constant,
    )

    model = load_model(model_path)
    probabilities = model.predict(features_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    matrix = confusion_matrix(
        labels_test,
        predictions,
        labels=np.arange(len(classes_list)),
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes_list,
    )

    figure_size = max(8, len(classes_list) * 0.7)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    accuracy = float(np.mean(predictions == labels_test))
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Confusion matrix saved to: {output_path}")

    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the BISINDO word class of a single video using a trained ConvLSTM (.h5) model."
    )
    parser.add_argument("--video", help="Path to the video file to predict.")
    parser.add_argument("--model", default="convlstm_model.h5", help="Path to the trained .h5 model file.")
    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIR,
        help="Dataset folder used at training time (needed to reconstruct the class list).",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Evaluate the holdout split and save its confusion-matrix plot.",
    )
    parser.add_argument(
        "--output",
        default="confusion_matrix.png",
        help="Output path for the confusion-matrix plot.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Holdout fraction used for the confusion matrix (default: 0.25).",
    )
    args = parser.parse_args()

    if args.confusion_matrix:
        plot_confusion_matrix(
            model_path=args.model,
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            test_size=args.test_size,
        )
    elif args.video:
        predict_on_video(args.video, model_path=args.model, dataset_dir=args.dataset_dir)
    else:
        parser.error("Provide --video or use --confusion-matrix.")