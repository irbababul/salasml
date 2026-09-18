import os
import pickle


def save_model(model, file_name, folder_name="model"):
    """
    Simpan model ke file pickle.

    == Contoh ==
    save_model(model, "my_model.pkl")
    save_model(model.best_estimator_, "my_model.pkl")

    == Parameter ==
    model      : object Python (sklearn pipeline, GridSearchCV, dll)
    file_name  : nama file, contoh "model.pkl"
    folder_name: folder tujuan (default: "model")
    """
    os.makedirs(folder_name, exist_ok=True)
    pickle.dump(model, open(f"{folder_name}/{file_name}", "wb"))
    print(f"Model tersimpan di {folder_name}/{file_name}")
