// Initiate the download of a file with a given name, text (content) and type

// Get the information handed to this script in 'source.data', which is
// formatted as a dictionary:
var filename = source.data['filename'][0];
var filetext = source.data['filetext'][0];
var filetype = source.data['filetype'][0];
// Create the 'blob' that can be downloaded:
var blob = new Blob([filetext], { type: filetype });

// Treat InternetExplorer in a special way:
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}
// All other browsers:
else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}