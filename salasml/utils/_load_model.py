import pickle


def load_model(model_path):
    """
    Load model dari file pickle.

    == Contoh ==
    model = load_model("model/my_model.pkl")

    == Parameter ==
    model_path: path ke file .pkl
    """
    model = pickle.load(open(model_path, "rb"))
    return model
