const reader = new FileReader();

function readURL(input) {
    if (input.files && input.files[0]) {
        reader.readAsDataURL(input.files[0]);
    }
}

reader.onload = function (e) {
    document.getElementById('imagePreview').src = e.target.result;
};

document.getElementById('imageInput').addEventListener('change', function () {
    readURL(this);
});