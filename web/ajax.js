//AJAX function to interface with php socket page
function AjaxResponse(message, data){
    data = typeof(data) != 'undefined' ? data : '';        
    try {
        var xhttp;
        try{    
            xhttp=new XMLHttpRequest();// Firefox, Opera 8.0+, Safari
        }catch (e){
            try{
                xhttp=new ActiveXObject("Msxml2.XMLHTTP"); // Internet Explorer
            }catch (e){
                try{
                    xhttp=new ActiveXObject("Microsoft.XMLHTTP");
                }catch (e){
                    return -1;
                }
            }
        }
        params = "m=" + message + "&d=" + data;
        xhttp.open("POST", "ajax.html?" + Math.random(), false);
        xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        xhttp.setRequestHeader("Content-length", params.length);
        xhttp.setRequestHeader("Connection", "close");
        xhttp.send(params);
        xmlDoc = xhttp.responseText;
        xhttp.abort()
        return xmlDoc; 
    } catch (err) {
        return -1;
    }
}
