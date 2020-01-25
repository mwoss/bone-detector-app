from os import path
from zipfile import ZipFile, ZIP_DEFLATED

from flask import Flask, render_template, redirect, flash, request, send_file
from werkzeug.utils import secure_filename

from bone_masker import BoneOpeningModel

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "super secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = BoneOpeningModel()


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def zipped(*args) -> str:
    zip_name = path.join(app.config["UPLOAD_FOLDER"], "prediction_result.zip")
    with ZipFile(zip_name, "w", ZIP_DEFLATED) as zip_file:
        for file_path in args:
            zip_file.write(file_path)

    return zip_name


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_path = path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(file_path)
            prediction, masked_image = model.predict(file_path)
            zipped_output = zipped(prediction, masked_image)
            return send_file(
                zipped_output,
                mimetype="zip",
                attachment_filename="prediction_result.zip",
                as_attachment=True
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run()
