function read_file(filename) {
    var reader = new FileReader();
    reader.onload = load_handler;
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function load_handler(event) {
    var b64string = event.target.result;
    var file_size_MB = format_bytes(format_b64_chars(b64string.length));


    if (source.data['server_mode'][0]){
        // It is assumed that in server mode, the upload limit problem is fixed
    }
    else{
        if (b64string.length > 10*1000*1000) {
            alert(input.files[0].name+' is '+file_size_MB+'. If the upload '+
            'does not work, please contact an administrator (and/or refer to '+
            'the known issues in dataexplorer.py).');
        }
    }
    // Spinning mouse wheel animation
    document.body.style.cursor = "wait";

    // Perform the "upload"
    source.data = {'contents' : [b64string], 'name':[input.files[0].name]};
    source.change.emit()
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.onchange = function(){
    if (window.FileReader) {
        read_file(input.files[0]);
    } else {
        alert('FileReader is not supported in this browser');
    }
}
input.click();

// Function for converting the number of chars in the b64string to bytes
function format_b64_chars(base64_chars) {
    var bytes = base64_chars*6.0/8.0
    return bytes
}

// Function for converting bytes to an appropriate string representation
function format_bytes(bytes,decimals) {
   if(bytes == 0) return '0 Bytes';
   var k = 1024,
       dm = decimals || 2,
       sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'],
       i = Math.floor(Math.log(bytes) / Math.log(k));
   return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
