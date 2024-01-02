document.addEventListener('change', function(event) {
    let target = event.target;

    if (target.classList.contains('file-input')) {
        let filesCount = target.files.length;
        let textbox = target.previousElementSibling;

        if (filesCount === 1)
            textbox.textContent = target.value.split('\\').pop();
        else
            textbox.textContent = filesCount + ' files selected';
    }
});

function upload() {
    let formData = new FormData();
    let form = document.getElementById('file-drop-area');
    let fileInput = form.getElementsByClassName("file-input")[0];
    let files = fileInput.files;

    for (let i = 0; i < files.length; i++)
        formData.append('files', files[i]);

    let xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload');
    xhr.responseType = 'blob';

    xhr.onload = function() {
        if (xhr.status !== 200)
            return alert(`Error ${xhr.status}: ${xhr.statusText}`);

        let blob;
        // Create a link element and trigger a click to download the image
        let link = document.createElement('a');

        if(xhr.response.type === "application/zip") {
            blob = new Blob([xhr.response], { type: 'application/zip' });
            link.download  = 'images.zip';
        }
        else if (xhr.response.type === "image/png"){
            blob = new Blob([xhr.response], { type: 'image/png' });
            link.download = fileInput.files[0].name;
        }
        else
          return alert("Unknown response");

        link.href = window.URL.createObjectURL(blob);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

    };

    xhr.send(formData);
}